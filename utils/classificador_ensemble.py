# utils/classificador_ensemble.py
# Combina CNN + FFT + Heurísticas para classificação mais robusta
# Reduz significativamente os Falsos Negativos (fakes passando como reais)

import numpy as np
import cv2
import tensorflow as tf
import os
import sys
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.70):
    def loss(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.cast(y_pred, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -(y_true * tf.math.log(y_pred)
                    + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessor          import PreProcessador
from utils.analisador_frequencia import AnalisadorFrequencia
from utils.seguranca             import VerificadorSeguranca


class ClassificadorEnsemble:
    """
    Combina três fontes de evidência para classificar REAL vs FAKE:

      1. CNN (MobileNetV2)     — padrões visuais profundos    peso: 60%
      2. Análise FFT           — artefatos de frequência GAN  peso: 25%
      3. Heurísticas visuais   — simetria, qualidade          peso: 15%

    A decisão final é uma média ponderada dos três scores.
    Isso reduz significativamente os falsos negativos.
    """

    def __init__(
        self,
        caminho_modelo="models/melhor_modelo.keras",
        limiar=0.45,
        pesos=None
    ):
        """
        Parâmetros:
            caminho_modelo: caminho para o modelo Keras treinado
            limiar:         probabilidade de corte para REAL/FAKE
            pesos:          dict com pesos de cada componente (opcional)
        """
        self.limiar = limiar
        self.pesos  = pesos or {
            "cnn":         0.60,
            "fft":         0.25,
            "heuristicas": 0.15,
        }

        self._carregar_modelo(caminho_modelo)
        self._inicializar_componentes()

    # ─────────────────────────────────────
    # INICIALIZAÇÃO
    # ─────────────────────────────────────

    def _carregar_modelo(self, caminho):
        """Carrega o modelo Keras."""
        if not os.path.exists(caminho):
            print(f"[Ensemble] AVISO: modelo não encontrado em '{caminho}'")
            print("[Ensemble] Rodando sem CNN — apenas FFT + heurísticas.")
            self.modelo = None
            self.pesos["fft"]         = 0.60
            self.pesos["heuristicas"] = 0.40
            self.pesos["cnn"]         = 0.00
            return

        print(f"[Ensemble] Carregando CNN: {caminho}")
        self.modelo = tf.keras.models.load_model(
    caminho,
    custom_objects={"loss": focal_loss()}
)
        print("[Ensemble] CNN carregada.")

    def _inicializar_componentes(self):
        """Inicializa os componentes do ensemble."""
        print("[Ensemble] Inicializando componentes...")
        self.prep        = PreProcessador(usar_mediapipe=True)
        self.analisador  = AnalisadorFrequencia(limiar_score=0.30)
        self.verificador = VerificadorSeguranca()

        print(f"[Ensemble] Pronto!")
        print(f"  Pesos: CNN={self.pesos['cnn']:.0%} | "
              f"FFT={self.pesos['fft']:.0%} | "
              f"Heur={self.pesos['heuristicas']:.0%}")
        print(f"  Limiar de decisão: {self.limiar}")

    # ─────────────────────────────────────
    # SCORES INDIVIDUAIS
    # ─────────────────────────────────────

    def _score_cnn(self, rosto_norm):
        """Retorna probabilidade FAKE da CNN (0–1)."""
        if self.modelo is None:
            return 0.5
        entrada = np.expand_dims(rosto_norm, axis=0)
        return float(self.modelo.predict(entrada, verbose=0)[0][0])

    def _score_fft(self, imagem_bgr):
        """Retorna score de suspeita FFT (0–1)."""
        try:
            resultado = self.analisador.analisar_imagem(imagem_bgr)
            return float(resultado["score_fft"])
        except Exception:
            return 0.5  # neutro se falhar

    def _score_heuristicas(self, rosto_norm):
        """
        Retorna score de suspeita heurística (0–1).
        Score alto = suspeito de fake.
        """
        try:
            resultado = self.verificador.score_combinado(rosto_norm)
            # Inverte: heurísticas retornam score alto = legítimo
            # Precisamos de score alto = suspeito para o ensemble
            return 1.0 - float(resultado["score_final"])
        except Exception:
            return 0.5  # neutro se falhar

    # ─────────────────────────────────────
    # MÉTODO PRINCIPAL: Classificar
    # ─────────────────────────────────────

    def classificar(self, imagem_bgr, verbose=False):
        """
        Classifica uma imagem BGR como REAL ou FAKE.

        Parâmetros:
            imagem_bgr: frame BGR do OpenCV (imagem completa, não só o rosto)
            verbose:    se True, exibe scores individuais no terminal

        Retorna:
            dict com label, prob_fake, confianca, bbox e scores
        """
        if imagem_bgr is None or imagem_bgr.size == 0:
            return self._resultado_sem_rosto()

        # Detecta e pré-processa o rosto
        rosto_norm, bbox = self.prep.recortar_rosto(imagem_bgr)

        if rosto_norm is None:
            return self._resultado_sem_rosto()

        # Recorta a região do rosto para análise FFT
        rosto_bgr = self._recortar_bbox(imagem_bgr, bbox)

        # Calcula scores individuais
        s_cnn  = self._score_cnn(rosto_norm)
        s_fft  = self._score_fft(rosto_bgr)
        s_heur = self._score_heuristicas(rosto_norm)

        # Score ensemble ponderado
        score_final = (
            s_cnn  * self.pesos["cnn"]         +
            s_fft  * self.pesos["fft"]         +
            s_heur * self.pesos["heuristicas"]
        )

        label     = "FAKE" if score_final >= self.limiar else "REAL"
        confianca = abs(score_final - 0.5) * 2.0  # 0=incerto, 1=muito confiante

        if verbose:
            print(f"\n[Ensemble] {label} ({score_final:.3f})")
            print(f"  CNN:          {s_cnn:.3f} (peso {self.pesos['cnn']:.0%})")
            print(f"  FFT:          {s_fft:.3f} (peso {self.pesos['fft']:.0%})")
            print(f"  Heurísticas:  {s_heur:.3f} (peso {self.pesos['heuristicas']:.0%})")
            print(f"  Final:        {score_final:.3f} | Confiança: {confianca:.1%}")

        return {
            "label":     label,
            "prob_fake": float(score_final),
            "confianca": float(confianca),
            "bbox":      bbox,
            "scores": {
                "cnn":         float(s_cnn),
                "fft":         float(s_fft),
                "heuristicas": float(s_heur),
                "final":       float(score_final),
            },
        }

    # ─────────────────────────────────────
    # AUXILIARES
    # ─────────────────────────────────────

    def _recortar_bbox(self, imagem_bgr, bbox):
        """Recorta a região do rosto com segurança."""
        if bbox is None:
            return imagem_bgr
        x, y, w, h    = bbox
        alt, larg      = imagem_bgr.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, larg - x)
        h = min(h, alt  - y)
        recorte = imagem_bgr[y:y+h, x:x+w]
        return recorte if recorte.size > 0 else imagem_bgr

    def _resultado_sem_rosto(self):
        """Retorna resultado padrão quando nenhum rosto é detectado."""
        return {
            "label":     "SEM_ROSTO",
            "prob_fake": 0.5,
            "confianca": 0.0,
            "bbox":      None,
            "scores":    {
                "cnn":         0.5,
                "fft":         0.5,
                "heuristicas": 0.5,
                "final":       0.5,
            },
        }


# ─────────────────────────────────────────
# TESTE RÁPIDO
# ─────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("Testando ClassificadorEnsemble...\n")

    ensemble = ClassificadorEnsemble(
        caminho_modelo="models/melhor_modelo.keras",
        limiar=0.45
    )

    # Testa com imagem aleatória (simula frame da webcam)
    img_teste = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    resultado = ensemble.classificar(img_teste, verbose=True)

    print(f"\nResultado:")
    print(f"  Label:     {resultado['label']}")
    print(f"  Prob FAKE: {resultado['prob_fake']:.4f}")
    print(f"  Confiança: {resultado['confianca']:.4f}")
    print("\n✅ ClassificadorEnsemble funcionando!")