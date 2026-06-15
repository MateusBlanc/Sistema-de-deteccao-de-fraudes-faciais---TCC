# main.py
# Sistema de detecção de fraudes faciais em tempo real
# Versão completa com Ensemble CNN + FFT + Heurísticas
# Execute: python main.py

import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import time
from collections import deque

from utils.preprocessor import PreProcessador
from utils.evaluator import focal_loss

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Lê limiar otimizado se existir, senão usa padrão
def _ler_limiar():
    caminho = "models/melhor_limiar.txt"
    if os.path.exists(caminho):
        with open(caminho) as f:
            limiar = float(f.read().strip())
        print(f"[Config] Limiar otimizado carregado: {limiar}")
        return limiar
    return 0.56

CONFIG = {
    "caminho_modelo":    "models/melhor_modelo.keras",
    "limiar":            _ler_limiar(),
    "janela_suavizacao": 15,
    "largura_cam":       1280,
    "altura_cam":        720,
    "mostrar_painel":    True,
    "indice_cam":        0,
    "usar_ensemble":     True,   # False = usa só a CNN
}

# ─────────────────────────────────────────
# PALETA DE CORES (BGR)
# ─────────────────────────────────────────

CORES = {
    "real":        (80,  200,  80),
    "fake":        (60,   60, 220),
    "neutro":      (180, 180, 180),
    "branco":      (255, 255, 255),
    "preto":       (0,     0,   0),
    "amarelo":     (0,   220, 220),
    "azul_painel": (45,   45,  45),
    "barra_fundo": (60,   60,  60),
}


# ─────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────

class SistemaDeteccaoFraude:
    """
    Sistema completo de detecção de fraudes faciais em tempo real.
    Usa ensemble CNN + FFT + Heurísticas quando disponível,
    com fallback automático para CNN simples se algo falhar.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG

        self.historico_probs = deque(maxlen=self.config["janela_suavizacao"])
        self.fps_historico   = deque(maxlen=30)
        self.total_frames    = 0
        self.total_real      = 0
        self.total_fake      = 0
        self.tempo_inicio    = time.time()

        self.usar_ensemble   = False
        self.ensemble        = None
        self.modelo          = None
        self.prep            = None

        self._inicializar()

    # ─────────────────────────────────────
    # INICIALIZAÇÃO
    # ─────────────────────────────────────

    def _inicializar(self):
        """Inicializa ensemble ou CNN simples com fallback automático."""

        # Tenta carregar o ensemble
        if self.config["usar_ensemble"]:
            try:
                from utils.classificador_ensemble import ClassificadorEnsemble
                print("[Sistema] Iniciando ensemble CNN + FFT + Heurísticas...")
                self.ensemble = ClassificadorEnsemble(
                    caminho_modelo=self.config["caminho_modelo"],
                    limiar=self.config["limiar"]
                )
                self.usar_ensemble = True
                print("[Sistema] Ensemble ativo.\n")
                return
            except Exception as e:
                print(f"[Sistema] Ensemble indisponível ({e})")
                print("[Sistema] Usando CNN simples como fallback.\n")

        # Fallback: CNN simples
        self._carregar_cnn_simples()

    def _carregar_cnn_simples(self):
        """Carrega só a CNN e o preprocessador."""
        print("[Sistema] Inicializando CNN simples...")

        # Preprocessador
        from utils.preprocessor import PreProcessador
        self.prep = PreProcessador(usar_mediapipe=True)

        # Modelo
        caminho = self.config["caminho_modelo"]
        if os.path.exists(caminho):
            modelo = tf.keras.models.load_model(
                caminho,
                custom_objects={"loss": focal_loss()}
            )
            print(f"[Sistema] Modelo carregado: {caminho}")
        else:
            print("[Sistema] Modelo não encontrado — modo demonstração ativo.")
            self.modelo = None

        self.usar_ensemble = False

    # ─────────────────────────────────────
    # PREDIÇÃO
    # ─────────────────────────────────────

    def _classificar_frame(self, frame):
        """
        Classifica um frame completo.
        Usa ensemble se disponível, CNN simples caso contrário.

        Retorna dict com label, prob_fake, confianca, bbox, scores.
        """
        if self.usar_ensemble and self.ensemble is not None:
            return self.ensemble.classificar(frame)

        # Fallback: CNN simples
        rosto_norm, bbox = self.prep.recortar_rosto(frame)

        if rosto_norm is None:
            return {
                "label":     "SEM_ROSTO",
                "prob_fake": 0.5,
                "confianca": 0.0,
                "bbox":      None,
                "scores":    {},
            }

        if self.modelo is None:
            prob = float(np.random.random())
        else:
            entrada = np.expand_dims(rosto_norm, axis=0)
            prob    = float(self.modelo.predict(entrada, verbose=0)[0][0])

        label     = "FAKE" if prob >= self.config["limiar"] else "REAL"
        confianca = abs(prob - 0.5) * 2.0

        return {
            "label":     label,
            "prob_fake": prob,
            "confianca": confianca,
            "bbox":      bbox,
            "scores":    {"cnn": prob},
        }

    def _suavizar_prob(self, prob_atual):
        """
        Suavização robusta com média ponderada e descarte de outliers.
        Frames recentes pesam mais; o frame mais extremo é descartado
        para eliminar saltos causados por uma única leitura ruim.
        """
        self.historico_probs.append(prob_atual)
        vals = np.array(self.historico_probs, dtype=np.float32)

        if len(vals) >= 4:
            # Descarta o valor mais distante da mediana (outlier)
            mediana = np.median(vals)
            idx_outlier = np.argmax(np.abs(vals - mediana))
            vals = np.delete(vals, idx_outlier)

        # Média ponderada: frames recentes pesam mais
        pesos = np.linspace(0.5, 1.0, len(vals))
        return float(np.average(vals, weights=pesos))

    # ─────────────────────────────────────
    # DESENHO DA INTERFACE
    # ─────────────────────────────────────

    def _desenhar_bbox(self, frame, bbox, label, prob_suav, cor):
        """Retângulo ao redor do rosto com resultado e cantos decorativos."""
        x, y, w, h = bbox

        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)

        # Cantos decorativos
        tam   = 20
        esp   = 3
        for px, py in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            dx = 1 if px == x else -1
            dy = 1 if py == y else -1
            cv2.line(frame, (px, py), (px + dx*tam, py), cor, esp)
            cv2.line(frame, (px, py), (px, py + dy*tam), cor, esp)

        # Rótulo com percentual
        confianca    = prob_suav if label == "FAKE" else (1 - prob_suav)
        texto_label  = f"{label}  {confianca*100:.1f}%"
        (tw, th), _  = cv2.getTextSize(
            texto_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        pad  = 6
        ry1  = max(0, y - th - 2*pad)
        ry2  = y
        rx1  = x
        rx2  = x + tw + 2*pad

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), cor, -1)
        cv2.putText(
            frame, texto_label,
            (rx1 + pad, ry2 - pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            CORES["branco"], 2, cv2.LINE_AA
        )

    def _desenhar_barra_confianca(self, frame, prob_suav, x, y, largura):
        """Barra de progresso REAL ↔ FAKE."""
        altura       = 16
        preenchimento = int(prob_suav * largura)

        cv2.rectangle(frame, (x, y), (x+largura, y+altura), CORES["barra_fundo"], -1)

        if preenchimento > 0:
            cv2.rectangle(frame, (x, y), (x+preenchimento, y+altura), CORES["fake"], -1)

        resto = largura - preenchimento
        if resto > 0:
            cv2.rectangle(frame, (x+preenchimento, y), (x+largura, y+altura), CORES["real"], -1)

        x_limiar = x + int(self.config["limiar"] * largura)
        cv2.line(frame, (x_limiar, y-3), (x_limiar, y+altura+3), CORES["amarelo"], 2)

        cv2.putText(frame, "REAL", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES["real"], 1)
        cv2.putText(frame, "FAKE", (x+largura-35, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES["fake"], 1)

    def _desenhar_scores_ensemble(self, frame, scores, x_painel, y_inicio):
        """
        Exibe scores individuais do ensemble no painel lateral.
        Só aparece quando o ensemble está ativo.
        """
        if not scores:
            return

        y = y_inicio
        cv2.putText(frame, "─── Scores ───",
                    (x_painel + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CORES["neutro"], 1)

        nomes = {
            "cnn":         "CNN",
            "fft":         "FFT",
            "heuristicas": "Heur.",
            "final":       "Final",
        }

        for chave, nome in nomes.items():
            if chave not in scores:
                continue
            y += 22
            val  = scores[chave]
            cor  = CORES["fake"] if val >= self.config["limiar"] else CORES["real"]
            barra_w = 80
            barra_x = x_painel + 55
            barra_h = 8

            cv2.putText(frame, f"{nome}:",
                        (x_painel + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, CORES["branco"], 1)

            # Mini barra de score
            cv2.rectangle(frame,
                          (barra_x, y-8),
                          (barra_x+barra_w, y-8+barra_h),
                          CORES["barra_fundo"], -1)
            cv2.rectangle(frame,
                          (barra_x, y-8),
                          (barra_x+int(val*barra_w), y-8+barra_h),
                          cor, -1)
            cv2.putText(frame, f"{val:.2f}",
                        (barra_x + barra_w + 4, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CORES["branco"], 1)

        return y

    def _desenhar_painel(self, frame, prob_suav, fps, tem_rosto, scores=None):
        """Painel lateral semitransparente com informações do sistema."""
        h_frame, w_frame  = frame.shape[:2]
        largura_painel    = 230
        x_painel          = w_frame - largura_painel

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_painel, 0), (w_frame, h_frame),
                      CORES["azul_painel"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Título
        modo = "ENSEMBLE" if self.usar_ensemble else "CNN"
        cv2.putText(frame, f"ANTIFRAUDE [{modo}]",
                    (x_painel + 8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    CORES["amarelo"], 1, cv2.LINE_AA)
        cv2.line(frame, (x_painel+8, 36), (w_frame-8, 36), CORES["neutro"], 1)

        # Status
        y = 58
        status_txt = "Rosto detectado" if tem_rosto else "Aguardando rosto..."
        status_cor = CORES["real"] if tem_rosto else CORES["neutro"]
        cv2.putText(frame, status_txt, (x_painel+8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_cor, 1, cv2.LINE_AA)

        # Probabilidade + barra
        y += 30
        cv2.putText(frame, f"Prob FAKE: {prob_suav*100:.1f}%",
                    (x_painel+8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, CORES["branco"], 1, cv2.LINE_AA)
        y += 12
        self._desenhar_barra_confianca(
            frame, prob_suav, x_painel+8, y, largura_painel-16
        )

        # Scores do ensemble
        y += 35
        if self.usar_ensemble and scores:
            y_final = self._desenhar_scores_ensemble(frame, scores, x_painel, y)
            y = (y_final or y) + 15
        else:
            y += 10

        # Estatísticas da sessão
        cv2.putText(frame, "─── Sessão ───",
                    (x_painel+8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CORES["neutro"], 1)

        tempo    = int(time.time() - self.tempo_inicio)
        min_, s  = divmod(tempo, 60)

        infos = [
            f"Tempo:   {min_:02d}:{s:02d}",
            f"Frames:  {self.total_frames}",
            f"REAL:    {self.total_real}",
            f"FAKE:    {self.total_fake}",
            f"FPS:     {fps:.1f}",
            f"Limiar:  {self.config['limiar']}",
        ]
        for info in infos:
            y += 22
            cv2.putText(frame, info, (x_painel+8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CORES["branco"], 1, cv2.LINE_AA)

        # Controles
        y_ctrl = h_frame - 85
        cv2.line(frame, (x_painel+8, y_ctrl), (w_frame-8, y_ctrl), CORES["neutro"], 1)
        for ctrl in ["[Q] Sair", "[S] Screenshot", "[R] Reset"]:
            y_ctrl += 20
            cv2.putText(frame, ctrl, (x_painel+8, y_ctrl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, CORES["neutro"], 1, cv2.LINE_AA)

    def _salvar_screenshot(self, frame):
        """Salva o frame atual."""
        os.makedirs("screenshots", exist_ok=True)
        nome = f"screenshots/captura_{int(time.time())}.png"
        cv2.imwrite(nome, frame)
        print(f"  Screenshot salvo: {nome}")

    # ─────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────

    def executar(self):
        """Loop principal — captura, processa e exibe em tempo real."""

        cap = cv2.VideoCapture(self.config["indice_cam"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config["largura_cam"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["altura_cam"])
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("ERRO: Não foi possível abrir a webcam.")
            return

        modo = "ENSEMBLE (CNN + FFT + Heurísticas)" if self.usar_ensemble else "CNN simples"
        print("=" * 55)
        print(f"  Sistema de Detecção de Fraudes Faciais")
        print(f"  Modo: {modo}")
        print(f"  Limiar: {self.config['limiar']}")
        print("  [Q] Sair  [S] Screenshot  [R] Reset")
        print("=" * 55 + "\n")

        prob_suav  = 0.5
        tem_rosto  = False
        scores_atu = {}

        while True:
            t_inicio = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame.")
                break

            self.total_frames += 1

            # ── Classifica o frame ────────────────────────────────────────
            resultado  = self._classificar_frame(frame)
            tem_rosto  = resultado["label"] != "SEM_ROSTO"
            bbox       = resultado["bbox"]
            scores_atu = resultado.get("scores", {})

            if tem_rosto:
                prob_suav = self._suavizar_prob(resultado["prob_fake"])

                # Histerese: zona morta de ±0.05 ao redor do limiar
                # evita piscar REAL/FAKE quando a prob fica na fronteira
                limiar = self.config["limiar"]
                margem_hist = 0.05
                rotulo_anterior = getattr(self, "_ultimo_label", "REAL")

                if prob_suav >= limiar + margem_hist:
                    label = "FAKE"
                elif prob_suav <= limiar - margem_hist:
                    label = "REAL"
                else:
                    label = rotulo_anterior  # mantém na zona de incerteza

                self._ultimo_label = label
                cor = CORES["fake"] if label == "FAKE" else CORES["real"]

                # Contabiliza e desenha o rótulo no rosto
                if label == "FAKE":
                    self.total_fake += 1
                else:
                    self.total_real += 1

                self._desenhar_bbox(frame, bbox, label, prob_suav, cor)
            else:
                self.historico_probs.clear()
                prob_suav  = 0.5
                scores_atu = {}

            # ── HUD ───────────────────────────────────────────────────────
            t_frame   = max(time.time() - t_inicio, 1e-6)
            fps_atual = 1.0 / t_frame
            self.fps_historico.append(fps_atual)
            fps_medio = float(np.mean(self.fps_historico))

            cv2.putText(frame, f"FPS: {fps_medio:.1f}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, CORES["amarelo"], 2, cv2.LINE_AA)

            if self.modelo is None and not self.usar_ensemble:
                cv2.putText(frame, "MODO DEMO — sem modelo treinado",
                            (12, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, CORES["amarelo"], 1, cv2.LINE_AA)

            if self.config["mostrar_painel"]:
                self._desenhar_painel(
                    frame, prob_suav, fps_medio,
                    tem_rosto, scores_atu
                )

            cv2.imshow("Deteccao de Fraudes Faciais | Q = sair", frame)

            # ── Teclas ───────────────────────────────────────────────────
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                print("\nEncerrando sistema...")
                break
            elif tecla == ord("s"):
                self._salvar_screenshot(frame)
            elif tecla == ord("r"):
                self.total_frames = 0
                self.total_real   = 0
                self.total_fake   = 0
                self.tempo_inicio = time.time()
                self.historico_probs.clear()
                print("  Estatísticas resetadas.")

        # ── Finalização ───────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()

        tempo  = int(time.time() - self.tempo_inicio)
        min_, s = divmod(tempo, 60)
        print("\n" + "=" * 45)
        print("  Sessão encerrada")
        print(f"  Frames:   {self.total_frames}")
        print(f"  REAL:     {self.total_real}")
        print(f"  FAKE:     {self.total_fake}")
        print(f"  Tempo:    {min_:02d}:{s:02d}")
        print("=" * 45 + "\n")


# ─────────────────────────────────────────
# PONTO DE ENTRADA
# ─────────────────────────────────────────

if __name__ == "__main__":
    sistema = SistemaDeteccaoFraude()
    sistema.executar()