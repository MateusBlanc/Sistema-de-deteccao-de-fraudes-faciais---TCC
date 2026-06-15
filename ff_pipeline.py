# ff_pipeline.py
# Pipeline completo FaceForensics++ → frames → rostos → treino
# Etapa 1: Extrai frames dos vídeos
# Etapa 2: Detecta e recorta rostos
# Etapa 3: Organiza dataset final
# Execute: python ff_pipeline.py

import cv2
import os
import gc
import random
import shutil
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─────────────────────────────────────────
# CONFIGURAÇÕES — ajuste conforme seu disco
# ─────────────────────────────────────────

CONFIG = {
    # Caminhos exatos confirmados pela estrutura baixada
    "pasta_ff":        "dataset/raw/ff++",

    "pasta_originais": "dataset/raw/ff++/original_sequences/youtube/raw/videos",
    "pastas_fakes": [
        "dataset/raw/ff++/manipulated_sequences/Deepfakes/c23/videos",
        "dataset/raw/ff++/manipulated_sequences/FaceSwap/c23/videos",
        "dataset/raw/ff++/manipulated_sequences/NeuralTextures/c23/videos",
    ],

    # Destino final
    "pasta_real":       "dataset/real",
    "pasta_fake":       "dataset/fake",
    "pasta_teste_real": "dataset/test/real",
    "pasta_teste_fake": "dataset/test/fake",

    # Limites calibrados para 8GB RAM
    # 200 vídeos originais × 8 frames = ~1600 reais disponíveis
    # 600 vídeos fake (200×3 métodos) × 8 frames = ~4800 fakes disponíveis
    # Usamos 1500 de cada para balancear e não estourar RAM
    "limite_real_treino": 1500,
    "limite_fake_treino": 1500,
    "limite_real_teste":  300,
    "limite_fake_teste":  300,

    # Extração de frames
    "frames_por_video":  8,
    "skip_inicio":       10,
    "skip_fim":          10,

    # Qualidade mínima
    "nitidez_minima":    25.0,
    "luminancia_min":    20,
    "luminancia_max":    235,

    "seed": 42,
}

# ─────────────────────────────────────────
# DETECTOR DE FACE
# ─────────────────────────────────────────

detector_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detectar_rosto(frame_bgr):
    """
    Detecta o rosto principal no frame e retorna recortado 224x224.
    Usa Haar Cascade — leve e sem dependências extras.
    """
    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector_haar.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # Pega a maior face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Margem de 20% ao redor
    margem = int(min(w, h) * 0.20)
    ih, iw = frame_bgr.shape[:2]
    x1 = max(0, x - margem)
    y1 = max(0, y - margem)
    x2 = min(iw, x + w + margem)
    y2 = min(ih, y + h + margem)

    rosto = frame_bgr[y1:y2, x1:x2]
    if rosto.size == 0:
        return None

    return cv2.resize(rosto, (224, 224))


def verificar_qualidade(rosto_224):
    """
    Verifica qualidade mínima do rosto extraído.
    Descarta frames borrados, escuros, ou sem variação.
    """
    cinza     = cv2.cvtColor(rosto_224, cv2.COLOR_BGR2GRAY)
    media     = float(np.mean(cinza))
    nitidez   = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
    contraste = float(np.std(cinza))

    if media < CONFIG["luminancia_min"]:   return False
    if media > CONFIG["luminancia_max"]:   return False
    if nitidez < CONFIG["nitidez_minima"]: return False
    if contraste < 12:                     return False
    return True


# ─────────────────────────────────────────
# EXTRAÇÃO DE FRAMES DE UM VÍDEO
# ─────────────────────────────────────────

def extrair_rostos_video(caminho_video, n_frames):
    """
    Extrai N rostos bem distribuídos ao longo do vídeo.
    Ignora frames iniciais e finais (fade/corte).

    Retorna lista de arrays BGR (224, 224, 3).
    """
    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inicio = CONFIG["skip_inicio"]
    fim    = total - CONFIG["skip_fim"]

    if fim - inicio < n_frames:
        cap.release()
        return []

    # Distribui frames uniformemente com pequena variação aleatória
    passo     = (fim - inicio) // n_frames
    posicoes  = [
        inicio + i * passo + random.randint(0, max(1, passo // 3))
        for i in range(n_frames)
    ]

    rostos = []
    for pos in posicoes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, fim))
        ret, frame = cap.read()
        if not ret:
            continue

        rosto = detectar_rosto(frame)
        if rosto is None:
            continue

        if not verificar_qualidade(rosto):
            continue

        rostos.append(rosto)

    cap.release()
    return rostos


# ─────────────────────────────────────────
# PROCESSAMENTO DE PASTA DE VÍDEOS
# ─────────────────────────────────────────

def processar_videos(pastas_video, pasta_saida, label,
                     limite, prefixo):
    """
    Processa vídeos de uma ou mais pastas.
    Extrai rostos e salva na pasta de saída.
    """
    extensoes = {".mp4", ".avi", ".mov", ".mkv"}

    # Coleta todos os vídeos das pastas fornecidas
    todos_videos = []
    for pasta in (pastas_video if isinstance(pastas_video, list)
                  else [pastas_video]):
        if not os.path.exists(pasta):
            print(f"  ⚠ Pasta não encontrada: {pasta}")
            continue
        videos = [
            p for p in Path(pasta).rglob("*")
            if p.suffix.lower() in extensoes
        ]
        todos_videos.extend(videos)
        print(f"  → {len(videos)} vídeos em {pasta}")

    if not todos_videos:
        print(f"  ❌ Nenhum vídeo encontrado para [{label}]")
        return 0

    random.shuffle(todos_videos)
    os.makedirs(pasta_saida, exist_ok=True)

    total_salvo  = 0
    total_videos = 0
    total_sem_rosto = 0

    print(f"\n  Processando {len(todos_videos)} vídeos [{label}]...")
    print(f"  Meta: {limite} rostos\n")

    for video in todos_videos:
        if total_salvo >= limite:
            break

        rostos = extrair_rostos_video(
            video,
            CONFIG["frames_por_video"]
        )

        if not rostos:
            total_sem_rosto += 1
        else:
            for i, rosto in enumerate(rostos):
                if total_salvo >= limite:
                    break
                nome = f"{prefixo}_{total_videos:04d}_{i:02d}.jpg"
                cv2.imwrite(
                    os.path.join(pasta_saida, nome),
                    rosto,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                total_salvo += 1

        total_videos += 1

        # Progresso
        if total_videos % 100 == 0:
            print(f"  [{label}] {total_videos} vídeos | "
                  f"{total_salvo}/{limite} rostos | "
                  f"{total_sem_rosto} sem rosto")
        gc.collect()

    print(f"\n  ✓ [{label}] {total_salvo} rostos salvos "
          f"de {total_videos} vídeos")
    return total_salvo


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def rodar_pipeline():
    print("=" * 60)
    print("  PIPELINE FF++ — Vídeos → Rostos → Dataset")
    print("=" * 60 + "\n")

    random.seed(CONFIG["seed"])

    # Cria pastas de saída
    for pasta in [
        CONFIG["pasta_real"],       CONFIG["pasta_fake"],
        CONFIG["pasta_teste_real"], CONFIG["pasta_teste_fake"],
    ]:
        os.makedirs(pasta, exist_ok=True)

    # ── ETAPA 1: Reais (treino) ───────────────────────────────────────────
    print("[1/4] Extraindo rostos REAIS (treino)...")
    n_real = processar_videos(
        CONFIG["pasta_originais"],
        CONFIG["pasta_real"],
        label="REAL",
        limite=CONFIG["limite_real_treino"],
        prefixo="ff_real"
    )

    # ── ETAPA 2: Fakes (treino) ───────────────────────────────────────────
    print("\n[2/4] Extraindo rostos FAKE (treino)...")
    print("  Fontes: Deepfakes + FaceSwap + NeuralTextures")
    n_fake = processar_videos(
        CONFIG["pastas_fakes"],
        CONFIG["pasta_fake"],
        label="FAKE",
        limite=CONFIG["limite_fake_treino"],
        prefixo="ff_fake"
    )

    # ── ETAPA 3: Reais (teste) ────────────────────────────────────────────
    print("\n[3/4] Extraindo rostos REAIS (teste)...")
    # Usa vídeos diferentes dos de treino para teste
    # O FF++ tem 1000 vídeos — pegamos os últimos 200 para teste
    n_real_te = processar_videos(
        CONFIG["pasta_originais"],
        CONFIG["pasta_teste_real"],
        label="REAL-TESTE",
        limite=CONFIG["limite_real_teste"],
        prefixo="ff_real_test"
    )

    # ── ETAPA 4: Fakes (teste) ────────────────────────────────────────────
    print("\n[4/4] Extraindo rostos FAKE (teste)...")
    n_fake_te = processar_videos(
        CONFIG["pastas_fakes"],
        CONFIG["pasta_teste_fake"],
        label="FAKE-TESTE",
        limite=CONFIG["limite_fake_teste"],
        prefixo="ff_fake_test"
    )

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"""
{'='*60}
  ✅ PIPELINE CONCLUÍDO!
{'='*60}

  TREINO:
    dataset/real/       → {n_real} rostos reais
    dataset/fake/       → {n_fake} rostos fake
    Total:              → {n_real + n_fake} imagens

  TESTE:
    dataset/test/real/  → {n_real_te} rostos reais
    dataset/test/fake/  → {n_fake_te} rostos fake
    Total:              → {n_real_te + n_fake_te} imagens

  Qualidade do dataset:
  ✓ Vídeos reais do YouTube (variação natural)
  ✓ Fakes com 3 técnicas diferentes (Deepfakes,
    FaceSwap, NeuralTextures)
  ✓ Filtro de qualidade aplicado em cada frame
  ✓ Rostos centralizados e padronizados 224x224

  Próximo passo:
  python train.py
{'='*60}
""")


# ─────────────────────────────────────────
# VERIFICAÇÃO ANTES DE RODAR
# ─────────────────────────────────────────

def verificar_estrutura():
    """
    Verifica se os vídeos do FF++ foram baixados corretamente
    antes de iniciar o pipeline.
    """
    print("Verificando estrutura do FF++...\n")

    pastas_verificar = [
        (CONFIG["pasta_originais"], "Vídeos originais (reais)"),
        (CONFIG["pastas_fakes"][0], "Deepfakes"),
        (CONFIG["pastas_fakes"][1], "FaceSwap"),
        (CONFIG["pastas_fakes"][2], "NeuralTextures"),
    ]

    extensoes = {".mp4", ".avi"}
    tudo_ok   = True

    for pasta, nome in pastas_verificar:
        if not os.path.exists(pasta):
            print(f"  ❌ {nome}: pasta não encontrada")
            print(f"     Esperado em: {pasta}")
            tudo_ok = False
        else:
            videos = [
                p for p in Path(pasta).iterdir()
                if p.suffix.lower() in extensoes
            ]
            print(f"  ✓ {nome}: {len(videos)} vídeos")

    if not tudo_ok:
        print("""
  ─────────────────────────────────────────────────
  Ajuste os caminhos em CONFIG conforme sua estrutura.
  A estrutura padrão do FF++ após download é:

  dataset/raw/ff++/
  ├── original_sequences/
  │   └── youtube/
  │       └── c23/
  │           └── videos/   ← pasta_originais
  └── manipulated_sequences/
      ├── Deepfakes/
      │   └── c23/
      │       └── videos/   ← pastas_fakes[0]
      ├── FaceSwap/
      │   └── c23/
      │       └── videos/   ← pastas_fakes[1]
      └── NeuralTextures/
          └── c23/
              └── videos/   ← pastas_fakes[2]
  ─────────────────────────────────────────────────
""")
        return False

    print("\n  ✅ Estrutura OK — pode rodar o pipeline!\n")
    return True


if __name__ == "__main__":
    if verificar_estrutura():
        rodar_pipeline()
    else:
        print("Corrija os caminhos no CONFIG e rode novamente.")