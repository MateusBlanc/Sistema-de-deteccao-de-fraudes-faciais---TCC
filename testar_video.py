# testar_video.py
# Avalia o modelo em multiplos videos do FF++ por tecnica.
# - Usa MediaPipe (consistente com main.py e o ensemble)
# - Processa varios videos por tecnica e tira media/estatisticas
# - Salva UM video de demonstracao por tecnica
# - Gera CSV com os resultados para o TCC
# Execute: python testar_video.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import csv
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ─────────────────────────────────────────
# FOCAL LOSS (necessaria para carregar o modelo)
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# CONFIGURACOES
# ─────────────────────────────────────────

VIDEOS_TESTE = {
    "Deepfakes": {
        "pasta": r"C:\dev_C - base\dataset\raw\ff++\manipulated_sequences\Deepfakes\c23\videos",
        "esperado": "FAKE",
    },
    "FaceSwap": {
        "pasta": r"C:\dev_C - base\dataset\raw\ff++\manipulated_sequences\FaceSwap\c23\videos",
        "esperado": "FAKE",
    },
    "NeuralTextures": {
        "pasta": r"C:\dev_C - base\dataset\raw\ff++\manipulated_sequences\NeuralTextures\c23\videos",
        "esperado": "FAKE",
    },
    "Original": {
        "pasta": r"C:\dev_C - base\dataset\raw\ff++\original_sequences\youtube\raw\videos",
        "esperado": "REAL",
    },
}

# Quantos videos analisar por tecnica
VIDEOS_POR_TECNICA = 8

# Quantos frames amostrar por video
FRAMES_POR_VIDEO = 30

# Processa 1 a cada N frames no video de demonstracao salvo
PULO_DEMO = 3

LIMIAR = 0.56
if Path("models/melhor_limiar.txt").exists():
    LIMIAR = float(open("models/melhor_limiar.txt").read().strip())

random.seed(123)  # seed fixa para reprodutibilidade


# ─────────────────────────────────────────
# CARREGA MODELO E DETECTOR
# ─────────────────────────────────────────

print("Carregando modelo...")
modelo = tf.keras.models.load_model(
    "models/melhor_modelo.keras",
    custom_objects={"loss": focal_loss()}
)

mp_face = mp.solutions.face_detection
detector_mp = mp_face.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

os.makedirs("testes_tcc/videos", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ─────────────────────────────────────────
# DETECCAO DE ROSTO (MediaPipe + fallback Haar)
# ─────────────────────────────────────────

detector_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detectar_rosto(frame):
    """Retorna (x1, y1, x2, y2) do maior rosto, ou None."""
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = detector_mp.process(frame_rgb)

    if resultado.detections:
        melhor = max(resultado.detections, key=lambda d: d.score[0])
        bbox = melhor.location_data.relative_bounding_box
        x  = int(bbox.xmin * w)
        y  = int(bbox.ymin * h)
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)
        margem = int(min(fw, fh) * 0.2)
        x1 = max(0, x - margem); y1 = max(0, y - margem)
        x2 = min(w, x + fw + margem); y2 = min(h, y + fh + margem)
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)

    # Fallback Haar
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector_haar.detectMultiScale(cinza, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        margem = int(min(fw, fh) * 0.2)
        x1 = max(0, x - margem); y1 = max(0, y - margem)
        x2 = min(w, x + fw + margem); y2 = min(h, y + fh + margem)
        return (x1, y1, x2, y2)

    return None


def prever_rosto(frame, bbox):
    """Roda o modelo no rosto recortado. Retorna prob_fake ou None."""
    x1, y1, x2, y2 = bbox
    rosto = frame[y1:y2, x1:x2]
    if rosto.size == 0:
        return None
    rosto_224  = cv2.resize(rosto, (224, 224))
    rosto_rgb  = cv2.cvtColor(rosto_224, cv2.COLOR_BGR2RGB)
    rosto_norm = rosto_rgb.astype(np.float32) / 255.0
    return float(modelo.predict(np.expand_dims(rosto_norm, 0), verbose=0)[0][0])


# ─────────────────────────────────────────
# ANALISE DE UM VIDEO
# ─────────────────────────────────────────

def analisar_video(caminho_video, n_frames=FRAMES_POR_VIDEO):
    cap = cv2.VideoCapture(str(caminho_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5:
        cap.release()
        return []

    n = min(n_frames, total)
    posicoes = np.linspace(5, total - 5, n, dtype=int)
    predicoes = []

    for pos in posicoes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = detectar_rosto(frame)
        if bbox is None:
            continue
        prob = prever_rosto(frame, bbox)
        if prob is not None:
            predicoes.append(prob)

    cap.release()
    return predicoes


# ─────────────────────────────────────────
# GERA VIDEO DE DEMONSTRACAO (1 por tecnica)
# ─────────────────────────────────────────

def gerar_video_demo(caminho_video, tecnica, esperado):
    cap = cv2.VideoCapture(str(caminho_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nome_saida = f"testes_tcc/videos/{tecnica}_demo.mp4"
    writer = cv2.VideoWriter(
        nome_saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % PULO_DEMO != 0:
            writer.write(frame)
            continue

        bbox = detectar_rosto(frame)
        label = "SEM ROSTO"
        prob  = 0.5
        cor   = (180, 180, 180)

        if bbox is not None:
            prob = prever_rosto(frame, bbox)
            if prob is not None:
                label = "FAKE" if prob >= LIMIAR else "REAL"
                cor = (60, 60, 220) if label == "FAKE" else (80, 200, 80)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        cv2.rectangle(frame, (0, 0), (360, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Tecnica: {tecnica} (esperado: {esperado})",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{label}  FAKE: {prob*100:.1f}%",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    return nome_saida


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────

print("=" * 62)
print("  AVALIACAO EM VIDEO — FaceForensics++")
print(f"  {VIDEOS_POR_TECNICA} videos por tecnica | "
      f"{FRAMES_POR_VIDEO} frames por video | limiar {LIMIAR}")
print("=" * 62)

resultados = {}
linhas_csv = []

for tecnica, info in VIDEOS_TESTE.items():
    pasta = info["pasta"]
    esperado = info["esperado"]

    if not os.path.exists(pasta):
        print(f"\n  {tecnica}: pasta nao encontrada ({pasta})")
        continue

    videos = list(Path(pasta).glob("*.mp4"))
    if not videos:
        print(f"\n  {tecnica}: nenhum video")
        continue

    random.shuffle(videos)
    videos_usar = videos[:VIDEOS_POR_TECNICA]

    print(f"\n── {tecnica} (esperado: {esperado}) ──")
    print(f"  Analisando {len(videos_usar)} videos...")

    todas_probs = []
    acertos_video = 0

    for i, video in enumerate(videos_usar):
        probs = analisar_video(video)
        if not probs:
            continue
        media_v = float(np.mean(probs))
        pred_v  = "FAKE" if media_v >= LIMIAR else "REAL"
        ok      = pred_v == esperado
        if ok:
            acertos_video += 1
        todas_probs.extend(probs)

        linhas_csv.append({
            "tecnica":   tecnica,
            "video":     video.name,
            "esperado":  esperado,
            "predito":   pred_v,
            "prob_fake": f"{media_v*100:.1f}%",
            "correto":   "Sim" if ok else "Nao",
        })
        print(f"    {video.name:<22} {media_v*100:5.1f}% -> {pred_v} "
              f"{'OK' if ok else 'ERRO'}")

    if not todas_probs:
        print("  Nenhum rosto detectado nesta tecnica.")
        continue

    arr = np.array(todas_probs)
    fakes_frame = (arr >= LIMIAR).mean() * 100
    acc_video   = acertos_video / len(videos_usar) * 100

    resultados[tecnica] = {
        "esperado":     esperado,
        "prob_media":   arr.mean() * 100,
        "prob_mediana": np.median(arr) * 100,
        "desvio":       arr.std() * 100,
        "prob_min":     arr.min() * 100,
        "prob_max":     arr.max() * 100,
        "pct_frame_fake": fakes_frame,
        "acc_video":    acc_video,
        "n_videos":     len(videos_usar),
        "n_frames":     len(arr),
    }

    # Gera video de demonstracao com o primeiro video da tecnica
    print(f"  Gerando video de demonstracao...")
    demo = gerar_video_demo(videos_usar[0], tecnica, esperado)
    print(f"  Demo salvo: {demo}")

detector_mp.close()


# ─────────────────────────────────────────
# RELATORIO FINAL
# ─────────────────────────────────────────

print("\n" + "=" * 62)
print("  RESUMO POR TECNICA")
print("=" * 62)
print(f"\n  {'Tecnica':<16}{'Prob FAKE':<12}{'Desvio':<10}"
      f"{'Acerto/video':<14}{'Frames':<8}")
print("  " + "-" * 58)

for tecnica, r in resultados.items():
    print(f"  {tecnica:<16}{r['prob_media']:>6.1f}%     "
          f"{r['desvio']:>5.1f}%    "
          f"{r['acc_video']:>5.0f}%        "
          f"{r['n_frames']:>5}")

print("  " + "-" * 58)
print("\n  DETALHES POR TECNICA:")
for tecnica, r in resultados.items():
    print(f"\n  {tecnica} (esperado: {r['esperado']}):")
    print(f"    Prob FAKE media:   {r['prob_media']:.1f}% "
          f"(mediana {r['prob_mediana']:.1f}%)")
    print(f"    Faixa:             {r['prob_min']:.1f}% a {r['prob_max']:.1f}%")
    print(f"    Frames detectados FAKE: {r['pct_frame_fake']:.0f}%")
    print(f"    Acerto por video:  {r['acc_video']:.0f}% "
          f"({r['n_videos']} videos)")

# Salva CSV detalhado
with open("logs/teste_video_resultados.csv", "w", newline="",
          encoding="utf-8") as f:
    writer_csv = csv.DictWriter(f, fieldnames=[
        "tecnica", "video", "esperado", "predito", "prob_fake", "correto"
    ])
    writer_csv.writeheader()
    writer_csv.writerows(linhas_csv)

print("\n" + "=" * 62)
print("  CSV salvo em: logs/teste_video_resultados.csv")
print("  Videos de demonstracao em: testes_tcc/videos/")
print("=" * 62)