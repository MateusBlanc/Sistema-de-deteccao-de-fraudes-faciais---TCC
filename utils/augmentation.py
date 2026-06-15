# utils/augmentation.py
# Gera variações das imagens de treino para o modelo aprender
# padrões mais robustos e não decorar os dados (overfitting)

import numpy as np
import cv2
import os
import random
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAÇÕES DE AUGMENTATION
# ─────────────────────────────────────────

# Probabilidade de cada transformação ser aplicada (0.0 a 1.0)
CONFIG = {
    "flip_horizontal":   0.5,   # espelhar horizontalmente
    "rotacao":           0.4,   # rotacionar levemente
    "brilho":            0.4,   # clarear ou escurecer
    "zoom":              0.3,   # aproximar levemente
    "ruido_gaussiano":   0.3,   # adicionar ruído (simula baixa qualidade)
    "desfoque":          0.2,   # borrar levemente (simula câmera tremida)
    "contraste":         0.3,   # ajustar contraste
}

# Limites das transformações
LIMITES = {
    "rotacao_max_graus":  12,    # máximo de graus para rotacionar
    "brilho_variacao":    0.25,  # variação de brilho (+ ou -)
    "zoom_max":           0.15,  # zoom máximo de 15%
    "ruido_intensidade":  0.03,  # intensidade do ruído gaussiano
    "desfoque_kernel":    3,     # tamanho do kernel de desfoque (deve ser ímpar)
    "contraste_variacao": 0.3,   # variação de contraste (+ ou -)
}


# ─────────────────────────────────────────
# CLASSE PRINCIPAL DE AUGMENTATION
# ─────────────────────────────────────────

class AugmentadorFacial:
    """
    Aplica transformações aleatórias em imagens de rostos para aumentar
    o dataset e melhorar a generalização do modelo.

    Todas as funções trabalham com imagens no formato:
    numpy array float32, shape (224, 224, 3), valores entre 0.0 e 1.0
    """

    def __init__(self, config=None, limites=None, seed=None):
        """
        Parâmetros:
            config:  dicionário com probabilidades (usa CONFIG padrão se None)
            limites: dicionário com limites das transformações
            seed:    semente para reprodutibilidade (None = aleatório)
        """
        self.config  = config  or CONFIG
        self.limites = limites or LIMITES

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    # ─────────────────────────────────────
    # TRANSFORMAÇÕES INDIVIDUAIS
    # ─────────────────────────────────────

    def _flip_horizontal(self, img):
        """Espelha a imagem horizontalmente (como um espelho)."""
        return np.fliplr(img)

    def _rotacionar(self, img):
        """Rotaciona a imagem levemente em torno do centro."""
        angulo = random.uniform(
            -self.limites["rotacao_max_graus"],
             self.limites["rotacao_max_graus"]
        )
        h, w = img.shape[:2]
        centro = (w // 2, h // 2)

        # Matriz de rotação 2D
        matriz = cv2.getRotationMatrix2D(centro, angulo, scale=1.0)

        # Aplica a rotação mantendo o mesmo tamanho da imagem
        rotada = cv2.warpAffine(
            img, matriz, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT  # preenche bordas com reflexo
        )
        return rotada

    def _ajustar_brilho(self, img):
        """Aumenta ou diminui o brilho da imagem."""
        variacao = random.uniform(
            -self.limites["brilho_variacao"],
             self.limites["brilho_variacao"]
        )
        # Soma a variação e garante que fique entre 0 e 1
        return np.clip(img + variacao, 0.0, 1.0)

    def _aplicar_zoom(self, img):
        """Aplica zoom central e recorta para o tamanho original."""
        fator = random.uniform(0.0, self.limites["zoom_max"])
        h, w  = img.shape[:2]

        # Calcula quanto recortar de cada lado
        crop_h = int(h * fator)
        crop_w = int(w * fator)

        # Recorta o centro da imagem
        recortada = img[crop_h:h-crop_h, crop_w:w-crop_w]

        # Redimensiona de volta para o tamanho original
        return cv2.resize(recortada, (w, h), interpolation=cv2.INTER_LINEAR)

    def _adicionar_ruido(self, img):
    
        intensidade = self.limites["ruido_intensidade"]
        # dtype=np.float32 direto evita alocação intermediária em float64
        ruido = np.random.normal(0, intensidade, img.shape)
        ruido = ruido.astype(np.float32, copy=False)
        return np.clip(img + ruido, 0.0, 1.0, out=img)

    def _aplicar_desfoque(self, img):
        """Aplica desfoque gaussiano leve (simula câmera levemente tremida)."""
        k = self.limites["desfoque_kernel"]
        # Converte para uint8 para usar o cv2.GaussianBlur
        img_uint8 = (img * 255).astype(np.uint8)
        desfocada = cv2.GaussianBlur(img_uint8, (k, k), sigmaX=1)
        return desfocada.astype(np.float32) / 255.0

    def _ajustar_contraste(self, img):
        """Aumenta ou diminui o contraste da imagem."""
        fator = 1.0 + random.uniform(
            -self.limites["contraste_variacao"],
             self.limites["contraste_variacao"]
        )
        # Aplica contraste em torno de 0.5 (meio do range normalizado)
        return np.clip((img - 0.5) * fator + 0.5, 0.0, 1.0)

    # ─────────────────────────────────────
    # MÉTODO PRINCIPAL: augmentar 1 imagem
    # ─────────────────────────────────────

    def augmentar(self, img):
        """
        Aplica transformações aleatórias em uma imagem.
        Cada transformação é aplicada conforme sua probabilidade em CONFIG.

        Parâmetro:
            img: numpy array float32, shape (H, W, 3), valores entre 0 e 1

        Retorna:
            imagem augmentada com mesmo shape e tipo
        """
        img = img.copy()

        # Dicionário que mapeia nome → função de transformação
        transformacoes = {
            "flip_horizontal": self._flip_horizontal,
            "rotacao":         self._rotacionar,
            "brilho":          self._ajustar_brilho,
            "zoom":            self._aplicar_zoom,
            "ruido_gaussiano": self._adicionar_ruido,
            "desfoque":        self._aplicar_desfoque,
            "contraste":       self._ajustar_contraste,
        }

        # Aplica cada transformação com sua probabilidade
        for nome, funcao in transformacoes.items():
            if random.random() < self.config.get(nome, 0):
                img = funcao(img)

        return img.astype(np.float32)

    # ─────────────────────────────────────
    # MÉTODO: Gerar múltiplas variações
    # ─────────────────────────────────────

    def gerar_variacoes(self, img, quantidade=5):
        """
        Gera N variações de uma mesma imagem.

        Útil quando o dataset é muito pequeno —
        cada imagem original vira 'quantidade' imagens diferentes.

        Retorna:
            lista de numpy arrays augmentados
        """
        return [self.augmentar(img) for _ in range(quantidade)]

    # ─────────────────────────────────────
    # MÉTODO: Aumentar dataset em disco
    # ─────────────────────────────────────

    def aumentar_pasta(self, pasta_entrada, pasta_saida, variacoes_por_imagem=3):
        """
        Lê todas as imagens de uma pasta, gera variações e salva na pasta de saída.
        Útil para aumentar um dataset pequeno antes do treinamento.

        Parâmetros:
            pasta_entrada:        onde estão as imagens originais
            pasta_saida:          onde salvar as imagens augmentadas
            variacoes_por_imagem: quantas cópias gerar de cada imagem
        """
        extensoes = {".jpg", ".jpeg", ".png"}
        arquivos  = [
            p for p in Path(pasta_entrada).iterdir()
            if p.suffix.lower() in extensoes
        ]

        os.makedirs(pasta_saida, exist_ok=True)

        total_geradas = 0
        print(f"\nAugmentando {len(arquivos)} imagens "
              f"({variacoes_por_imagem} variações cada)...")

        for i, arquivo in enumerate(arquivos):
            img = cv2.imread(str(arquivo))
            if img is None:
                continue

            # Normaliza para float32
            img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0

            # Salva a imagem original também
            nome_orig = os.path.join(pasta_saida, f"orig_{i:05d}.png")
            cv2.imwrite(nome_orig, img)
            total_geradas += 1

            # Gera e salva as variações
            variacoes = self.gerar_variacoes(img_norm, variacoes_por_imagem)
            for j, var in enumerate(variacoes):
                var_uint8 = (var * 255).astype(np.uint8)
                var_bgr   = cv2.cvtColor(var_uint8, cv2.COLOR_RGB2BGR)
                nome_var  = os.path.join(pasta_saida, f"aug_{i:05d}_{j}.png")
                cv2.imwrite(nome_var, var_bgr)
                total_geradas += 1

            if (i + 1) % 50 == 0:
                print(f"  → {i+1}/{len(arquivos)} imagens processadas...")

        print(f"  ✓ Total gerado: {total_geradas} imagens em '{pasta_saida}'")
        return total_geradas