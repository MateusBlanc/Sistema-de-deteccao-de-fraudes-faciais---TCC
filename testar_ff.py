# testar_ff.py
# Testa o modelo com imagens extraídas do FF++
# Execute: python testar_ff.py

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

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
# CARREGA MODELO E LIMIAR
# ─────────────────────────────────────────

modelo = tf.keras.models.load_model(
    "models/melhor_modelo.keras",
    custom_objects={"loss": focal_loss()}
)

LIMIAR = 0.56
if Path("models/melhor_limiar.txt").exists():
    LIMIAR = float(open("models/melhor_limiar.txt").read().strip())

print(f"Modelo carregado. Limiar: {LIMIAR}\n")

# ─────────────────────────────────────────
# FUNÇÃO DE PREDIÇÃO
# ─────────────────────────────────────────

def predizer_imagem(caminho):
    img = cv2.imread(str(caminho))
    if img is None:
        return None
    img_rgb  = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    prob     = modelo.predict(
        np.expand_dims(img_norm, 0), verbose=0
    )[0][0]
    pred     = "FAKE" if prob >= LIMIAR else "REAL"
    return float(prob), pred

# ─────────────────────────────────────────
# TESTA REAIS
# ─────────────────────────────────────────

resultados = {
    "REAL": {"corretos": 0, "total": 0, "detalhes": []},
    "FAKE": {"corretos": 0, "total": 0, "detalhes": []},
}

print("=" * 60)
print("  TESTE COM IMAGENS FF++")
print("=" * 60)

print("\n--- ROSTOS REAIS ---")
for img_path in sorted(Path("testes_tcc/real").glob("*.jpg")):
    resultado = predizer_imagem(img_path)
    if resultado is None:
        continue
    prob, pred = resultado
    correto = "OK" if pred == "REAL" else "ERRO"
    resultados["REAL"]["total"] += 1
    if pred == "REAL":
        resultados["REAL"]["corretos"] += 1
    resultados["REAL"]["detalhes"].append({
        "arquivo": img_path.name,
        "prob_fake": prob,
        "pred": pred,
        "correto": pred == "REAL"
    })
    print(f"  [{correto}] {img_path.name:<35} "
          f"FAKE={prob*100:.1f}%  -> {pred}")

# ─────────────────────────────────────────
# TESTA FAKES
# ─────────────────────────────────────────

print("\n--- DEEPFAKES FF++ ---")
for img_path in sorted(Path("testes_tcc/fake").glob("*.jpg")):
    resultado = predizer_imagem(img_path)
    if resultado is None:
        continue
    prob, pred = resultado
    correto = "OK" if pred == "FAKE" else "ERRO"
    resultados["FAKE"]["total"] += 1
    if pred == "FAKE":
        resultados["FAKE"]["corretos"] += 1
    resultados["FAKE"]["detalhes"].append({
        "arquivo": img_path.name,
        "prob_fake": prob,
        "pred": pred,
        "correto": pred == "FAKE"
    })
    print(f"  [{correto}] {img_path.name:<35} "
          f"FAKE={prob*100:.1f}%  -> {pred}")

# ─────────────────────────────────────────
# RESUMO
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("  RESUMO FINAL")
print("=" * 60)

for classe, dados in resultados.items():
    total   = dados["total"]
    corretos = dados["corretos"]
    acc     = corretos / max(total, 1) * 100
    erros   = total - corretos
    print(f"\n  {classe}:")
    print(f"    Corretos:  {corretos}/{total} ({acc:.0f}%)")
    print(f"    Erros:     {erros}/{total}")

    if dados["detalhes"]:
        probs = [d["prob_fake"] for d in dados["detalhes"]]
        print(f"    Prob FAKE media: {sum(probs)/len(probs)*100:.1f}%")
        print(f"    Prob FAKE min:   {min(probs)*100:.1f}%")
        print(f"    Prob FAKE max:   {max(probs)*100:.1f}%")

total_corretos = (resultados["REAL"]["corretos"] +
                  resultados["FAKE"]["corretos"])
total_geral    = (resultados["REAL"]["total"] +
                  resultados["FAKE"]["total"])
acc_geral      = total_corretos / max(total_geral, 1) * 100

print(f"\n  ACURACIA GERAL: {total_corretos}/{total_geral} ({acc_geral:.0f}%)")
print("=" * 60)
print("\nSalvando resultados para o TCC...")

# Salva CSV com resultados detalhados
import csv
with open("logs/teste_ff_resultados.csv", "w", newline="",
          encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "arquivo", "classe_real", "prob_fake", "pred", "correto"
    ])
    writer.writeheader()
    for classe, dados in resultados.items():
        for d in dados["detalhes"]:
            writer.writerow({
                "arquivo":    d["arquivo"],
                "classe_real": classe,
                "prob_fake":  f"{d['prob_fake']*100:.1f}%",
                "pred":       d["pred"],
                "correto":    "Sim" if d["correto"] else "Nao"
            })

print("CSV salvo em: logs/teste_ff_resultados.csv")
print("Use essa tabela no TCC!")