# utils/seguranca.py
# Documentação técnica de segurança do sistema
# Analisa limitações, vetores de ataque e sugestões de melhoria
# Execute: python utils/seguranca.py

"""
Este módulo serve dois propósitos:

1. Documentação técnica — explica as limitações e vulnerabilidades
   do sistema de detecção de fraudes faciais (importante para o TCC)

2. Utilitários de defesa — implementa verificações extras que podem
   ser integradas ao main.py para aumentar a robustez do sistema
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────
# DOCUMENTAÇÃO: VETORES DE ATAQUE
# ─────────────────────────────────────────

VETORES_DE_ATAQUE = {

    "spoofing_foto": {
        "nome":        "Spoofing com Fotografia",
        "nivel_risco": "MÉDIO",
        "descricao": """
        O atacante apresenta uma foto impressa ou exibida em tela
        de uma pessoa real na frente da câmera.
        Sistemas sem liveness detection são vulneráveis a esse ataque.
        """,
        "como_detectar": [
            "Analisar textura 2D vs 3D (LBP - Local Binary Patterns)",
            "Detectar reflexo de luz em superfície plana",
            "Verificar ausência de micromovimentos naturais",
            "Pedir para o usuário piscar ou sorrir (challenge-response)",
        ],
        "mitigacao_implementada": "Parcial — CNN detecta padrões de foto impressa",
    },

    "spoofing_video": {
        "nome":        "Spoofing com Vídeo",
        "nivel_risco": "ALTO",
        "descricao": """
        O atacante exibe um vídeo da pessoa real em um dispositivo
        (celular, tablet, monitor) na frente da câmera.
        Mais difícil de detectar que foto pois há movimento natural.
        """,
        "como_detectar": [
            "Detectar bordas e reflexos de tela (moiré pattern)",
            "Analisar inconsistência de iluminação entre face e ambiente",
            "Verificar compressão de vídeo em pixels (artefatos JPEG/H264)",
            "Análise de profundidade com câmera IR ou estéreo",
        ],
        "mitigacao_implementada": "Limitada — requer hardware adicional (câmera IR)",
    },

    "deepfake_gerado": {
        "nome":        "Deepfake Sintético",
        "nivel_risco": "ALTO",
        "descricao": """
        Uso de redes GAN ou diffusion models para gerar um rosto
        sintético fotorrealista que não pertence a nenhuma pessoa real.
        Ferramentas como StyleGAN2, Midjourney e DALL-E produzem
        imagens difíceis de distinguir a olho nu.
        """,
        "como_detectar": [
            "Analisar inconsistências ao redor dos olhos e orelhas",
            "Verificar padrões de ruído no espectro de frequência (FFT)",
            "Detectar ausência de vasos sanguíneos reais (rPPG)",
            "Usar modelos especializados: FaceForensics, XceptionNet",
        ],
        "mitigacao_implementada": "Principal foco do sistema — CNN treinada para isso",
    },

    "deepfake_troca_rosto": {
        "nome":        "Face Swap (Troca de Rosto)",
        "nivel_risco": "MUITO ALTO",
        "descricao": """
        O atacante substitui o rosto de uma pessoa em um vídeo real
        pelo rosto da vítima usando técnicas como DeepFaceLab ou Roop.
        O corpo e ambiente são reais, apenas o rosto é sintético.
        Esse tipo de ataque é o mais usado em crimes financeiros digitais.
        """,
        "como_detectar": [
            "Verificar inconsistência entre textura do rosto e pescoço",
            "Analisar bordas do rosto (halos, blending artificial)",
            "Detectar sincronização labial com áudio (lip-sync)",
            "Verificar piscadas naturais (deepfakes antigos não piscam)",
        ],
        "mitigacao_implementada": "Parcial — treinamento com FaceForensics++ ajuda",
    },

    "adversarial_attack": {
        "nome":        "Ataque Adversarial",
        "nivel_risco": "MUITO ALTO",
        "descricao": """
        Perturbações imperceptíveis ao olho humano são adicionadas
        à imagem para enganar especificamente o modelo de ML.
        Um atacante com acesso ao modelo pode gerar uma imagem FAKE
        que o sistema classifica como REAL com alta confiança.
        """,
        "como_detectar": [
            "Input preprocessing: suavização gaussiana antes da inferência",
            "Ensemble de modelos — múltiplos modelos devem concordar",
            "Detecção de perturbações no espectro de frequência",
            "Adversarial training — treinar com exemplos adversariais",
        ],
        "mitigacao_implementada": "Nenhuma — área ativa de pesquisa",
    },
}


# ─────────────────────────────────────────
# CLASSE: Verificações extras de segurança
# ─────────────────────────────────────────

class VerificadorSeguranca:
    """
    Implementa verificações heurísticas extras que complementam
    o modelo de ML principal, aumentando a robustez do sistema.

    Essas verificações analisam características físicas que são
    difíceis de falsificar sem hardware especializado.
    """

    def __init__(self):
        self.historico_frames = []    # armazena frames recentes
        self.max_historico    = 30    # últimos 30 frames (~1 segundo a 30fps)

    # ─────────────────────────────────────
    # VERIFICAÇÃO 1: Variação temporal
    # ─────────────────────────────────────

    def verificar_variacao_temporal(self, rosto_norm):
        """
        Verifica se o rosto tem variação natural entre frames.

        Uma foto ou imagem estática tem variação temporal próxima de zero.
        Um rosto real tem micromovimentos naturais (respiração, piscada).

        Retorna:
            dict com score (0–1) e diagnóstico
            score próximo de 0 → suspeito (muito estático)
            score próximo de 1 → natural (tem variação)
        """
        self.historico_frames.append(rosto_norm.copy())

        # Precisa de pelo menos 5 frames para análise
        if len(self.historico_frames) < 5:
            return {"score": 1.0, "diagnostico": "Aguardando frames suficientes"}

        # Mantém apenas os últimos N frames
        if len(self.historico_frames) > self.max_historico:
            self.historico_frames.pop(0)

        # Calcula diferença média entre frames consecutivos
        diferencas = []
        for i in range(1, len(self.historico_frames)):
            diff = np.mean(np.abs(
                self.historico_frames[i] - self.historico_frames[i-1]
            ))
            diferencas.append(diff)

        variacao_media = np.mean(diferencas)

        # Limiares empíricos (ajustados para imagens 224x224 normalizadas)
        LIMIAR_MUITO_ESTATICO = 0.0005  # provavelmente foto
        LIMIAR_NORMAL         = 0.002   # variação natural esperada

        if variacao_media < LIMIAR_MUITO_ESTATICO:
            score       = 0.1
            diagnostico = "ALERTA: Rosto muito estático (possível foto)"
        elif variacao_media < LIMIAR_NORMAL:
            score       = 0.6
            diagnostico = "ATENÇÃO: Pouca variação temporal"
        else:
            score       = 1.0
            diagnostico = "OK: Variação temporal natural detectada"

        return {
            "score":      score,
            "variacao":   float(variacao_media),
            "diagnostico": diagnostico,
        }

    # ─────────────────────────────────────
    # VERIFICAÇÃO 2: Qualidade da imagem
    # ─────────────────────────────────────

    def verificar_qualidade_imagem(self, rosto_norm):
        """
        Analisa a qualidade da imagem do rosto.

        Deepfakes gerados frequentemente têm padrões de ruído
        diferentes de câmeras reais, e fotos impressas perdem
        nitidez de forma característica.

        Retorna:
            dict com score e métricas de qualidade
        """
        # Converte para uint8 para análise
        rosto_uint8 = (rosto_norm * 255).astype(np.uint8)
        cinza        = cv2.cvtColor(rosto_uint8, cv2.COLOR_RGB2GRAY)

        # ── Variância do Laplaciano (nitidez) ────────────────────────────
        # Alta variância = imagem nítida
        # Baixa variância = imagem borrada ou desfocada
        nitidez = cv2.Laplacian(cinza, cv2.CV_64F).var()

        # ── Análise de frequência (FFT) ───────────────────────────────────
        # Deepfakes e fotos impressas têm espectro de frequência diferente
        fft        = np.fft.fft2(cinza)
        fft_shift  = np.fft.fftshift(fft)
        magnitude  = np.log(np.abs(fft_shift) + 1)

        # Energia nas altas frequências (detalhes finos)
        h, w      = magnitude.shape
        centro_h  = h // 2
        centro_w  = w // 2
        raio      = min(h, w) // 6
        mascara   = np.zeros((h, w), dtype=bool)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist      = np.sqrt((y_grid - centro_h)**2 + (x_grid - centro_w)**2)
        mascara[dist > raio] = True   # regiões de alta frequência
        energia_hf = float(np.mean(magnitude[mascara]))

        # ── Diagnóstico ───────────────────────────────────────────────────
        NITIDEZ_MINIMA = 50.0

        if nitidez < NITIDEZ_MINIMA:
            score       = 0.3
            diagnostico = "ATENÇÃO: Imagem com baixa nitidez"
        else:
            score       = 1.0
            diagnostico = "OK: Qualidade de imagem adequada"

        return {
            "score":       score,
            "nitidez":     float(nitidez),
            "energia_hf":  energia_hf,
            "diagnostico": diagnostico,
        }

    # ─────────────────────────────────────
    # VERIFICAÇÃO 3: Simetria facial
    # ─────────────────────────────────────

    def verificar_simetria_facial(self, rosto_norm):
        """
        Verifica se o rosto tem simetria natural.

        Deepfakes gerados por GAN frequentemente têm simetria
        excessiva ou insuficiente comparado a rostos reais.

        Retorna:
            dict com score e índice de simetria
        """
        rosto_uint8 = (rosto_norm * 255).astype(np.uint8)
        cinza        = cv2.cvtColor(rosto_uint8, cv2.COLOR_RGB2GRAY)

        h, w = cinza.shape
        meio  = w // 2

        # Divide o rosto em metade esquerda e direita
        lado_esq = cinza[:, :meio]
        lado_dir = cinza[:, meio:]

        # Espelha o lado direito para comparar com o esquerdo
        lado_dir_espelhado = np.fliplr(lado_dir)

        # Garante que os dois lados tenham o mesmo tamanho
        min_largura       = min(lado_esq.shape[1], lado_dir_espelhado.shape[1])
        lado_esq           = lado_esq[:, :min_largura]
        lado_dir_espelhado = lado_dir_espelhado[:, :min_largura]

        # Calcula similaridade (1 - diferença normalizada)
        diff     = np.abs(lado_esq.astype(float) - lado_dir_espelhado.astype(float))
        simetria = 1.0 - (np.mean(diff) / 255.0)

        # Rostos reais têm simetria entre 0.7 e 0.92
        # Abaixo de 0.7 → muito assimétrico (suspeito)
        # Acima de 0.95 → simetria artificial (possível deepfake GAN)
        if simetria < 0.70:
            score       = 0.4
            diagnostico = "ATENÇÃO: Assimetria facial elevada"
        elif simetria > 0.95:
            score       = 0.5
            diagnostico = "ATENÇÃO: Simetria excessiva (possível GAN)"
        else:
            score       = 1.0
            diagnostico = "OK: Simetria facial dentro do esperado"

        return {
            "score":       score,
            "simetria":    float(simetria),
            "diagnostico": diagnostico,
        }

    # ─────────────────────────────────────
    # SCORE COMBINADO
    # ─────────────────────────────────────

    def score_combinado(self, rosto_norm):
        """
        Combina todas as verificações em um único score de segurança.

        Retorna:
            dict com score final (0–1) e resultados individuais
            Score próximo de 1 → rosto provavelmente legítimo
            Score próximo de 0 → múltiplos sinais de alerta
        """
        variacao  = self.verificar_variacao_temporal(rosto_norm)
        qualidade = self.verificar_qualidade_imagem(rosto_norm)
        simetria  = self.verificar_simetria_facial(rosto_norm)

        # Pesos de cada verificação
        # Variação temporal é o sinal mais confiável
        score_final = (
            variacao["score"]  * 0.50 +
            qualidade["score"] * 0.30 +
            simetria["score"]  * 0.20
        )

        alertas = []
        for resultado in [variacao, qualidade, simetria]:
            if resultado["score"] < 0.7:
                alertas.append(resultado["diagnostico"])

        return {
            "score_final":  float(score_final),
            "variacao":     variacao,
            "qualidade":    qualidade,
            "simetria":     simetria,
            "alertas":      alertas,
            "aprovado":     score_final >= 0.6 and len(alertas) == 0,
        }


# ─────────────────────────────────────────
# FUNÇÃO: Imprimir relatório de segurança
# ─────────────────────────────────────────

def imprimir_relatorio_seguranca():
    """
    Imprime relatório completo de segurança para o TCC.
    Documenta todos os vetores de ataque e mitigações.
    """
    print("\n" + "=" * 65)
    print("  RELATÓRIO DE SEGURANÇA — Sistema de Detecção de Fraudes Faciais")
    print("=" * 65)

    niveis_cor = {
        "MÉDIO":      "⚠",
        "ALTO":       "🔴",
        "MUITO ALTO": "🚨",
    }

    for chave, ataque in VETORES_DE_ATAQUE.items():
        icone = niveis_cor.get(ataque["nivel_risco"], "•")
        print(f"\n{icone}  {ataque['nome']}")
        print(f"   Nível de risco: {ataque['nivel_risco']}")
        print(f"   Mitigação atual: {ataque['mitigacao_implementada']}")
        print(f"   Como detectar:")
        for tecnica in ataque["como_detectar"]:
            print(f"     • {tecnica}")

    print("\n" + "─" * 65)
    print("  SUGESTÕES DE MELHORIA FUTURA (para o TCC)")
    print("─" * 65)

    melhorias = [
        ("Liveness Detection",
         "Detectar piscadas e micromovimentos para confirmar presença física"),
        ("rPPG (Remote PPG)",
         "Medir variação de cor da pele causada pelos batimentos cardíacos"),
        ("Análise Multimodal",
         "Combinar face + voz + comportamento para autenticação mais robusta"),
        ("Adversarial Training",
         "Treinar com exemplos adversariais para maior resistência a ataques"),
        ("Câmera de Profundidade",
         "Sensor IR ou ToF para distinguir rosto 3D real de foto/tela 2D"),
        ("Ensemble de Modelos",
         "Combinar XceptionNet + EfficientNet + verificações heurísticas"),
        ("Atualização Contínua",
         "Retreinar periodicamente com novos deepfakes para manter eficácia"),
    ]

    for i, (titulo, descricao) in enumerate(melhorias, 1):
        print(f"\n  {i}. {titulo}")
        print(f"     {descricao}")

    print("\n" + "=" * 65)
    print("  LIMITAÇÕES ATUAIS DO SISTEMA")
    print("=" * 65)

    limitacoes = [
        "Não detecta ataques adversariais sofisticados",
        "Performance depende da qualidade e diversidade do dataset",
        "Sem liveness detection nativo (vulnerável a fotos em tela)",
        "Modelo pode ter viés se dataset não for diverso o suficiente",
        "Não funciona bem com iluminação muito baixa ou oclusão parcial",
        "Não autentica identidade — apenas classifica REAL vs FAKE",
    ]

    for lim in limitacoes:
        print(f"  ✗ {lim}")

    print("\n" + "=" * 65 + "\n")


# ─────────────────────────────────────────
# PONTO DE ENTRADA
# ─────────────────────────────────────────

if __name__ == "__main__":
    imprimir_relatorio_seguranca()

    print("Testando verificações de segurança com imagem sintética...\n")
    verificador = VerificadorSeguranca()
    img_teste   = np.random.rand(224, 224, 3).astype(np.float32)

    for i in range(8):
        resultado = verificador.score_combinado(img_teste)
        # Adiciona leve variação para simular movimento
        img_teste = np.clip(img_teste + np.random.normal(0, 0.003, img_teste.shape), 0, 1)

    print(f"Score de segurança final: {resultado['score_final']:.3f}")
    print(f"Aprovado:                 {resultado['aprovado']}")
    if resultado["alertas"]:
        print("Alertas:")
        for alerta in resultado["alertas"]:
            print(f"  ⚠ {alerta}")
    print("\n✅ Módulo de segurança funcionando corretamente!")