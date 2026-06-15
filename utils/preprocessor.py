# utils/preprocessor.py
# Responsável por detectar, recortar e normalizar rostos
# Usado tanto no treinamento quanto no sistema em tempo real

import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────

# Tamanho padrão de entrada para o modelo (MobileNetV2 espera 224x224)
IMG_SIZE = (224, 224)

# Margem ao redor do rosto detectado (em pixels)
MARGEM = 30


# ─────────────────────────────────────────
# CLASSE PRINCIPAL DE PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────

class PreProcessador:
    """
    Detecta rostos em imagens/frames e os prepara para o modelo.
    Usa MediaPipe como detector principal e OpenCV como fallback.
    """

    def __init__(self, usar_mediapipe=True):
        """
        Parâmetros:
            usar_mediapipe: Se True, usa MediaPipe (mais preciso).
                            Se False, usa Haar Cascade do OpenCV (mais rápido).
        """
        self.usar_mediapipe = usar_mediapipe

        # Inicializa o detector MediaPipe
        if usar_mediapipe:
            self.mp_face = mp.solutions.face_detection
            self.detector = self.mp_face.FaceDetection(
                model_selection=0,       # 0 = curta distância (webcam), 1 = longa distância
                min_detection_confidence=0.6  # confiança mínima para aceitar a detecção
            )
            print("[PreProcessador] Usando MediaPipe para detecção de faces.")

        # Inicializa o Haar Cascade como alternativa
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("[PreProcessador] Haar Cascade carregado como fallback.")

    # ─────────────────────────────────────
    # MÉTODO: Detectar rosto com MediaPipe
    # ─────────────────────────────────────
    def _detectar_mediapipe(self, imagem_bgr):
        """
        Detecta rostos usando MediaPipe.
        Retorna lista de bounding boxes (x, y, w, h) em pixels.
        """
        # MediaPipe trabalha com RGB, OpenCV usa BGR — precisamos converter
        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        resultado = self.detector.process(imagem_rgb)

        boxes = []
        if resultado.detections:
            h, w = imagem_bgr.shape[:2]
            for deteccao in resultado.detections:
                bbox = deteccao.location_data.relative_bounding_box
                # Converte coordenadas relativas (0.0–1.0) para pixels absolutos
                x = int(bbox.xmin * w) - MARGEM
                y = int(bbox.ymin * h) - MARGEM
                larg = int(bbox.width * w) + 2 * MARGEM
                alt = int(bbox.height * h) + 2 * MARGEM
                boxes.append((x, y, larg, alt))

        return boxes

    # ─────────────────────────────────────
    # MÉTODO: Detectar rosto com Haar Cascade
    # ─────────────────────────────────────
    def _detectar_haar(self, imagem_bgr):
        """
        Detecta rostos usando Haar Cascade do OpenCV.
        Mais leve, porém menos preciso que o MediaPipe.
        """
        # Haar funciona melhor em escala de cinza
        cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            cinza,
            scaleFactor=1.1,   # quanto reduzir a imagem a cada escala
            minNeighbors=5,    # quantos vizinhos mínimos para confirmar detecção
            minSize=(60, 60)   # tamanho mínimo do rosto em pixels
        )
        return list(faces) if len(faces) > 0 else []

    # ─────────────────────────────────────
    # MÉTODO: Recortar e normalizar o rosto
    # ─────────────────────────────────────
    def recortar_rosto(self, imagem_bgr):
        """
        Detecta, recorta e normaliza o primeiro rosto encontrado.

        Retorna:
            numpy array de shape (224, 224, 3) normalizado entre 0 e 1,
            ou None se nenhum rosto for encontrado.
        """
        # Tenta com MediaPipe primeiro
        boxes = []
        if self.usar_mediapipe:
            boxes = self._detectar_mediapipe(imagem_bgr)

        # Se MediaPipe não encontrou, tenta Haar Cascade
        if len(boxes) == 0:
            boxes = self._detectar_haar(imagem_bgr)

        # Se ainda não encontrou rosto, retorna None
        if len(boxes) == 0:
            return None, None

        # Pega o primeiro rosto detectado
        x, y, w, h = boxes[0]

        # Garante que as coordenadas não saiam dos limites da imagem
        altura_img, largura_img = imagem_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, largura_img - x)
        h = min(h, altura_img - y)

        # Recorta o rosto da imagem original
        rosto = imagem_bgr[y:y+h, x:x+w]

        # Se o recorte ficou vazio por algum erro, retorna None
        if rosto.size == 0:
            return None, None

        # Redimensiona para o tamanho esperado pelo modelo
        rosto_redim = cv2.resize(rosto, IMG_SIZE)

        # Converte BGR (OpenCV) para RGB (padrão para redes neurais)
        rosto_rgb = cv2.cvtColor(rosto_redim, cv2.COLOR_BGR2RGB)

        # Normaliza os pixels de [0, 255] para [0.0, 1.0]
        rosto_norm = rosto_rgb.astype(np.float32) / 255.0

        # Retorna a imagem normalizada e o bounding box para desenhar na tela
        return rosto_norm, (x, y, w, h)

    # ─────────────────────────────────────
    # MÉTODO: Processar pasta de imagens
    # ─────────────────────────────────────
    def processar_pasta(self, caminho_entrada, caminho_saida, label, limite=None):
        """
        Processa todas as imagens de uma pasta, detectando e recortando rostos.
        Salva as imagens processadas em caminho_saida.

        Parâmetros:
            caminho_entrada: pasta com imagens brutas (ex: "dataset/raw/real")
            caminho_saida:   pasta de destino  (ex: "dataset/real")
            label:           "real" ou "fake" (apenas para log)
            limite:          número máximo de imagens a processar (None = todas)
        """
        # Extensões de imagem aceitas
        extensoes = {".jpg", ".jpeg", ".png"}

        # Lista todos os arquivos de imagem na pasta
        arquivos = [
            p for p in Path(caminho_entrada).iterdir()
            if p.suffix.lower() in extensoes
        ]

        if limite:
            arquivos = arquivos[:limite]

        os.makedirs(caminho_saida, exist_ok=True)

        processados = 0
        sem_rosto  = 0

        print(f"\n[{label.upper()}] Processando {len(arquivos)} imagens...")

        for i, arquivo in enumerate(arquivos):
            # Lê a imagem
            imagem = cv2.imread(str(arquivo))
            if imagem is None:
                continue

            # Detecta e normaliza o rosto
            rosto, _ = self.recortar_rosto(imagem)

            if rosto is None:
                sem_rosto += 1
                continue

            # Converte de volta para uint8 para salvar como PNG
            rosto_salvar = (rosto * 255).astype(np.uint8)
            rosto_bgr    = cv2.cvtColor(rosto_salvar, cv2.COLOR_RGB2BGR)

            # Salva com nome padronizado
            nome_saida = os.path.join(caminho_saida, f"{label}_{i:05d}.png")
            cv2.imwrite(nome_saida, rosto_bgr)
            processados += 1

            # Mostra progresso a cada 100 imagens
            if (i + 1) % 100 == 0:
                print(f"  → {i+1}/{len(arquivos)} processadas...")

        print(f"  ✓ Salvas: {processados} | Sem rosto detectado: {sem_rosto}")
        return processados


# ─────────────────────────────────────────
# FUNÇÃO AUXILIAR: carregar dataset em memória
# ─────────────────────────────────────────

def carregar_dataset(pasta_real, pasta_fake, limite_por_classe=None):
    """
    Carrega imagens processadas das pastas real e fake.
    Retorna arrays numpy prontos para o modelo.

    Retorna:
        X: array de imagens, shape (N, 224, 224, 3), valores entre 0 e 1
        y: array de labels, 0 = REAL, 1 = FAKE
    """
    extensoes = {".jpg", ".jpeg", ".png"}
    X, y = [], []

    for pasta, label in [(pasta_real, 0), (pasta_fake, 1)]:
        arquivos = [
            p for p in Path(pasta).iterdir()
            if p.suffix.lower() in extensoes
        ]
        if limite_por_classe:
            arquivos = arquivos[:limite_por_classe]

        nome_label = "REAL" if label == 0 else "FAKE"
        print(f"Carregando {len(arquivos)} imagens [{nome_label}]...")

        for arquivo in arquivos:
            img = cv2.imread(str(arquivo))
            if img is None:
                continue
            # Converte e normaliza
            img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
            X.append(img_norm)
            y.append(label)

    print(f"\nDataset carregado: {len(X)} imagens total")
    print(f"  → REAL: {y.count(0)} | FAKE: {y.count(1)}")

    return np.array(X), np.array(y)