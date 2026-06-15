# utils/otimizar_limiar.py
# Encontra o limiar ideal para maximizar F1 e reduzir falsos negativos
# Execute: python utils/otimizar_limiar.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, sys

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

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def otimizar_limiar():
    print("=" * 55)
    print("  OTIMIZAÇÃO DE LIMIAR")
    print("=" * 55 + "\n")

    # Carrega modelo e dados de teste
    caminho = "models/melhor_modelo.keras"
    if not os.path.exists(caminho):
        print("ERRO: modelo não encontrado.")
        return

    print("Carregando modelo e dados de teste...")
    modelo = tf.keras.models.load_model(
    caminho,
    custom_objects={"loss": focal_loss()}
)
    X_teste = np.load("models/X_teste.npy")
    y_teste = np.load("models/y_teste.npy")

    # Gera probabilidades
    print(f"Rodando predições em {len(X_teste)} imagens...")
    probs = modelo.predict(X_teste, batch_size=8, verbose=1).flatten()

    # Testa todos os limiares de 0.20 a 0.70
    limiares  = np.arange(0.20, 0.71, 0.01)
    resultados = []

    for limiar in limiares:
        preds = (probs >= limiar).astype(int)
        f1    = f1_score(y_teste, preds, zero_division=0)
        rec   = recall_score(y_teste, preds, zero_division=0)
        prec  = precision_score(y_teste, preds, zero_division=0)
        resultados.append({
            "limiar":   round(limiar, 2),
            "f1":       f1,
            "recall":   rec,
            "precisao": prec,
        })

    # Encontra o melhor limiar por F1
    melhor_f1   = max(resultados, key=lambda x: x["f1"])

    # Encontra limiar com recall >= 0.80 (prioriza detectar fakes)
    recall_alto = [r for r in resultados if r["recall"] >= 0.80]
    melhor_recall = max(recall_alto, key=lambda x: x["f1"]) if recall_alto else melhor_f1

    print(f"\n{'─'*50}")
    print(f"  Limiar atual (0.50):")
    atual = next(r for r in resultados if r["limiar"] == 0.50)
    print(f"    F1={atual['f1']:.4f} | Recall={atual['recall']:.4f} | Precisão={atual['precisao']:.4f}")

    print(f"\n  Melhor limiar por F1: {melhor_f1['limiar']}")
    print(f"    F1={melhor_f1['f1']:.4f} | Recall={melhor_f1['recall']:.4f} | Precisão={melhor_f1['precisao']:.4f}")

    print(f"\n  Melhor limiar (recall >= 0.80): {melhor_recall['limiar']}")
    print(f"    F1={melhor_recall['f1']:.4f} | Recall={melhor_recall['recall']:.4f} | Precisão={melhor_recall['precisao']:.4f}")
    print(f"{'─'*50}")

    # Plota curvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    limiares_vals = [r["limiar"] for r in resultados]
    f1_vals       = [r["f1"]     for r in resultados]
    rec_vals      = [r["recall"] for r in resultados]
    prec_vals     = [r["precisao"] for r in resultados]

    ax1.plot(limiares_vals, f1_vals,   color="#9B59B6", linewidth=2, label="F1-score")
    ax1.plot(limiares_vals, rec_vals,  color="#1D9E75", linewidth=2, label="Recall",   linestyle="--")
    ax1.plot(limiares_vals, prec_vals, color="#E27B4A", linewidth=2, label="Precisão", linestyle=":")
    ax1.axvline(0.50,                      color="gray",    linestyle="--", alpha=0.6, label="Atual (0.50)")
    ax1.axvline(melhor_f1["limiar"],       color="#9B59B6", linestyle="-",  alpha=0.8, label=f"Melhor F1 ({melhor_f1['limiar']})")
    ax1.axvline(melhor_recall["limiar"],   color="#1D9E75", linestyle="-",  alpha=0.8, label=f"Recall≥0.80 ({melhor_recall['limiar']})")
    ax1.set_xlabel("Limiar"); ax1.set_ylabel("Score")
    ax1.set_title("Métricas por Limiar"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Matriz de confusão com melhor limiar
    preds_melhor = (probs >= melhor_recall["limiar"]).astype(int)
    cm = confusion_matrix(y_teste, preds_melhor)
    import seaborn as sns
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    anot = np.array([[f"{cm[i][j]}\n({cm_norm[i][j]*100:.1f}%)"
                      for j in range(2)] for i in range(2)])
    sns.heatmap(cm_norm, annot=anot, fmt="", cmap="Blues", ax=ax2,
                xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"],
                linewidths=0.5, annot_kws={"size": 12})
    ax2.set_xlabel("Predição"); ax2.set_ylabel("Real")
    ax2.set_title(f"Matriz com limiar={melhor_recall['limiar']}")

    plt.suptitle("Otimização de Limiar", fontsize=13)
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/otimizacao_limiar.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n✅ Gráfico salvo em logs/otimizacao_limiar.png")

    # Salva o melhor limiar em arquivo para o main.py usar
    with open("models/melhor_limiar.txt", "w") as f:
        f.write(str(melhor_recall["limiar"]))
    print(f"✅ Limiar salvo em models/melhor_limiar.txt: {melhor_recall['limiar']}")
    print(f"\nAtualize o main.py:")
    print(f'  CONFIG["limiar"] = {melhor_recall["limiar"]}')


if __name__ == "__main__":
    otimizar_limiar()