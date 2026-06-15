# utils/analisador_frequencia.py
# Analisa o espectro de frequência da imagem para detectar
# artefatos de geração sintética invisíveis ao olho humano

import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AnalisadorFrequencia:
    """
    Deepfakes gerados por GAN têm uma assinatura característica
    no espectro de frequência — padrões de grade (grid artifacts)
    causados pela arquitetura de upsampling das redes geradoras.

    Esta classe extrai features do espectro FFT para complementar
    a classificação da CNN principal.
    """

    def __init__(self, limiar_score=0.35):
        """
        Parâmetro:
            limiar_score: score acima do qual considera suspeito de fake
        """
        self.limiar = limiar_score

    # ─────────────────────────────────────
    # MÉTODO: Extrair espectro FFT
    # ─────────────────────────────────────

    def extrair_espectro(self, imagem_bgr):
        """
        Extrai o espectro de magnitude FFT de uma imagem.

        Retorna:
            espectro normalizado (numpy array 2D)
        """
        cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

        h, w     = cinza.shape
        janela   = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        janelado = cinza.astype(np.float32) * janela

        fft       = np.fft.fft2(janelado)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1e-8)

        mag_min  = magnitude.min()
        mag_max  = magnitude.max()
        mag_norm = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

        return mag_norm

    # ─────────────────────────────────────
    # MÉTODO: Detectar artefatos de grade
    # ─────────────────────────────────────

    def detectar_artefatos_grade(self, imagem_bgr):
        """
        Detecta padrões de grade no espectro — assinatura clássica de GANs.

        GANs com transposed convolution produzem picos periódicos no
        espectro que se manifestam como checkerboard artifacts.

        Retorna:
            dict com score (0–1) e diagnóstico
        """
        espectro = self.extrair_espectro(imagem_bgr)
        h, w     = espectro.shape
        ch, cw   = h // 2, w // 2

        # ── Analisa simetria do espectro ──────────────────────────────────
        quad_tl = espectro[:ch, :cw]
        quad_br = espectro[ch:,  cw:]
        quad_tr = espectro[:ch,  cw:]
        quad_bl = espectro[ch:,  :cw]

        tamanho = (min(ch, cw), min(ch, cw))
        q_tl = cv2.resize(quad_tl, tamanho)
        q_br = cv2.resize(np.flipud(np.fliplr(quad_br)), tamanho)
        q_tr = cv2.resize(quad_tr, tamanho)
        q_bl = cv2.resize(np.fliplr(np.flipud(quad_bl)), tamanho)

        simetria_diag  = 1.0 - np.mean(np.abs(q_tl - q_br))
        simetria_anti  = 1.0 - np.mean(np.abs(q_tr - q_bl))
        simetria_media = (simetria_diag + simetria_anti) / 2.0

        # ── Detecta energia em altas frequências ──────────────────────────
        y_g, x_g    = np.ogrid[:h, :w]
        dist_centro = np.sqrt((y_g - ch)**2 + (x_g - cw)**2)

        # Remove componente DC (baixa frequência)
        raio_dc = min(h, w) // 20
        mascara_hf = dist_centro > (min(h, w) // 4)

        espectro_sem_dc = espectro.copy()
        espectro_sem_dc[dist_centro < raio_dc] = 0
        energia_hf = float(np.mean(espectro_sem_dc[mascara_hf]))

        # ── Score combinado ───────────────────────────────────────────────
        score_simetria = max(0.0, (simetria_media - 0.70) / 0.30)
        score_energia  = min(1.0, energia_hf / 0.5)
        score_final    = score_simetria * 0.6 + score_energia * 0.4

        suspeito = score_final >= self.limiar

        return {
            "score":       float(score_final),
            "simetria":    float(simetria_media),
            "energia_hf":  float(energia_hf),
            "suspeito":    suspeito,
            "diagnostico": "SUSPEITO: padrões de GAN detectados"
                           if suspeito else "OK: espectro dentro do esperado",
        }

    # ─────────────────────────────────────
    # MÉTODO: Análise completa
    # ─────────────────────────────────────

    def analisar_imagem(self, imagem_bgr):
        """
        Análise completa de uma imagem BGR.

        Retorna:
            dict com score_fft, suspeito, diagnostico e detalhes
        """
        if imagem_bgr is None or imagem_bgr.size == 0:
            return {
                "score_fft":   0.5,
                "suspeito":    False,
                "diagnostico": "Imagem inválida",
                "detalhes":    {},
            }

        # Garante tamanho mínimo para FFT
        h, w = imagem_bgr.shape[:2]
        if h < 32 or w < 32:
            imagem_bgr = cv2.resize(imagem_bgr, (64, 64))

        resultado = self.detectar_artefatos_grade(imagem_bgr)

        return {
            "score_fft":   resultado["score"],
            "suspeito":    resultado["suspeito"],
            "diagnostico": resultado["diagnostico"],
            "detalhes":    resultado,
        }


# ─────────────────────────────────────────
# TESTE RÁPIDO
# ─────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("Testando AnalisadorFrequencia...\n")
    analisador = AnalisadorFrequencia()

    # Imagem sintética simulando um deepfake (padrão periódico)
    img_fake = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 8):
        img_fake[i::16, :] = 128  # padrão de grade

    # Imagem natural (ruído aleatório)
    img_real = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)

    res_fake = analisador.analisar_imagem(img_fake)
    res_real = analisador.analisar_imagem(img_real)

    print(f"Imagem com padrão de grade (FAKE simulado):")
    print(f"  Score FFT: {res_fake['score_fft']:.4f}")
    print(f"  Suspeito:  {res_fake['suspeito']}")
    print(f"  {res_fake['diagnostico']}")

    print(f"\nImagem natural (REAL simulado):")
    print(f"  Score FFT: {res_real['score_fft']:.4f}")
    print(f"  Suspeito:  {res_real['suspeito']}")
    print(f"  {res_real['diagnostico']}")

    print("\n✅ AnalisadorFrequencia funcionando!")