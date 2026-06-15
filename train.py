# train.py
# Treinamento otimizado para FaceForensics++
# Focal Loss + MobileNetV2 fine-tuning + callbacks robustos
# Execute: python train.py

import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# CONFIGURAÇÃO DE AMBIENTE
# ─────────────────────────────────────────

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["OMP_NUM_THREADS"]        = "2"

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from utils.preprocessor import carregar_dataset
from utils.augmentation  import AugmentadorFacial
from models.modelo       import construir_modelo, exibir_resumo

# ─────────────────────────────────────────
# CONFIGURAÇÕES — FF++ / 8GB RAM
# ─────────────────────────────────────────

CONFIG = {
    "pasta_real":          "dataset/real",
    "pasta_fake":          "dataset/fake",

    # Usa tudo disponível até o limite
    # Será ajustado automaticamente pelo dataset real
    "limite_por_classe":   None,

    "proporcao_teste":     0.15,
    "proporcao_val":       0.15,

    # Treino
    "epocas":              50,
    "batch_size":          8,
    "taxa_aprendizado":    1e-4,
    "dropout_rate":        0.5,
    "descongelar_camadas": 40,
    "usar_augmentation":   True,

    # Focal Loss — prioriza detectar fakes
    "focal_gamma":         2.0,
    "focal_alpha":         0.75,

    "pasta_modelos":       "models",
    "pasta_logs":          "logs",
}


# ─────────────────────────────────────────
# FOCAL LOSS
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
# GERADOR LEVE
# ─────────────────────────────────────────

def criar_gerador(X, y, batch_size, augmentar=False):
    """
    Gerador de batches com gerenciamento agressivo de memória.
    Augmentation desligado por padrão para economizar RAM.
    """
    aug = AugmentadorFacial() if augmentar else None
    n   = len(X)

    while True:
        indices = np.random.permutation(n)
        for inicio in range(0, n, batch_size):
            idx = indices[inicio:inicio + batch_size]

            X_batch = []
            for i in idx:
                img = X[i].copy()
                if augmentar and aug is not None:
                    img = aug.augmentar(img)
                X_batch.append(img)

            batch_arr = np.array(X_batch, dtype=np.float32)
            X_batch.clear()
            gc.collect()
            yield batch_arr, y[idx]


# ─────────────────────────────────────────
# CALLBACKS ROBUSTOS
# ─────────────────────────────────────────

def obter_callbacks_ff(pasta_modelos, paciencia=8):
    """
    Callbacks otimizados para FF++.
    Monitora AUC em vez de loss — mais relevante para detecção de fraudes.
    Paciência maior (8) porque o FF++ tem padrões mais complexos.
    """
    os.makedirs(pasta_modelos, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    return [
        # Salva o melhor modelo por AUC
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(pasta_modelos, "melhor_modelo.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        # Para se não melhorar por 8 épocas
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=paciencia,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduz LR se travar por 4 épocas
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),

        # Salva histórico em CSV para o TCC
        tf.keras.callbacks.CSVLogger(
            filename="logs/historico_treino_ff.csv",
            append=False
        ),
    ]


# ─────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────

def plotar_historico(historico):
    os.makedirs("logs", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ep = range(1, len(historico.history["loss"]) + 1)

    # Loss
    axs[0].plot(ep, historico.history["loss"],
                color="#4A90E2", linewidth=2, label="Treino")
    axs[0].plot(ep, historico.history["val_loss"],
                color="#E27B4A", linewidth=2, linestyle="--", label="Validação")
    axs[0].set_title("Loss por Época")
    axs[0].set_xlabel("Época")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Acurácia
    axs[1].plot(ep, historico.history["accuracy"],
                color="#4A90E2", linewidth=2, label="Treino")
    axs[1].plot(ep, historico.history["val_accuracy"],
                color="#E27B4A", linewidth=2, linestyle="--", label="Validação")
    axs[1].set_title("Acurácia por Época")
    axs[1].set_xlabel("Época")
    axs[1].set_ylim([0, 1])
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # AUC
    axs[2].plot(ep, historico.history["auc"],
                color="#4A90E2", linewidth=2, label="Treino")
    axs[2].plot(ep, historico.history["val_auc"],
                color="#E27B4A", linewidth=2, linestyle="--", label="Validação")
    axs[2].set_title("AUC por Época")
    axs[2].set_xlabel("Época")
    axs[2].set_ylim([0, 1])
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.suptitle(
        "Treinamento FF++ — Focal Loss | MobileNetV2",
        fontsize=14
    )
    plt.tight_layout()
    caminho = "logs/historico_treinamento_ff.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Gráfico salvo em: {caminho}")


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def treinar():
    print("=" * 60)
    print("  TREINAMENTO FF++ — Detecção de Fraudes Faciais")
    print("  Focal Loss | MobileNetV2 | 8GB RAM")
    print("=" * 60)

    os.makedirs(CONFIG["pasta_modelos"], exist_ok=True)
    os.makedirs(CONFIG["pasta_logs"],    exist_ok=True)

    # ── 1. Carrega dataset ────────────────────────────────────────────────
    print("\n[1/5] Carregando dataset FF++...")
    X, y = carregar_dataset(
        pasta_real=CONFIG["pasta_real"],
        pasta_fake=CONFIG["pasta_fake"],
        limite_por_classe=CONFIG["limite_por_classe"]
    )

    if len(X) == 0:
        print("\nERRO: Dataset vazio.")
        print("Execute python ff_pipeline.py primeiro.")
        sys.exit(1)

    # Detecta desbalanceamento e ajusta
    n_real = int((y == 0).sum())
    n_fake = int((y == 1).sum())
    print(f"\n  REAL: {n_real} | FAKE: {n_fake}")

    if abs(n_real - n_fake) > 100:
        print(f"\n  ⚠ Classes desbalanceadas — ajustando para {min(n_real, n_fake)} por classe...")
        minimo = min(n_real, n_fake)
        idx_real = np.where(y == 0)[0][:minimo]
        idx_fake = np.where(y == 1)[0][:minimo]
        idx_bal  = np.concatenate([idx_real, idx_fake])
        np.random.shuffle(idx_bal)
        X = X[idx_bal]
        y = y[idx_bal]
        print(f"  ✓ Balanceado: {len(X)} imagens ({minimo} por classe)")

    # ── 2. Divide dados ───────────────────────────────────────────────────
    print("\n[2/5] Dividindo dados...")
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y,
        test_size=CONFIG["proporcao_teste"],
        stratify=y,
        random_state=42
    )
    val_rel = CONFIG["proporcao_val"] / (1 - CONFIG["proporcao_teste"])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_rel,
        stratify=y_tv,
        random_state=42
    )

    print(f"  Treino:    {len(X_tr)}")
    print(f"  Validação: {len(X_val)}")
    print(f"  Teste:     {len(X_te)}")

    np.save(os.path.join(CONFIG["pasta_modelos"], "X_teste.npy"), X_te)
    np.save(os.path.join(CONFIG["pasta_modelos"], "y_teste.npy"), y_te)
    print("  ✓ Conjunto de teste salvo.")

    del X, y, X_tv, y_tv
    gc.collect()

    # ── 3. Constrói modelo ────────────────────────────────────────────────
    print("\n[3/5] Construindo modelo...")
    modelo = construir_modelo(
        taxa_aprendizado=CONFIG["taxa_aprendizado"],
        dropout_rate=CONFIG["dropout_rate"],
        descongelar_camadas=CONFIG["descongelar_camadas"]
    )

    # Recompila com Focal Loss
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=CONFIG["taxa_aprendizado"]
        ),
        loss=focal_loss(
            gamma=CONFIG["focal_gamma"],
            alpha=CONFIG["focal_alpha"]
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precisao"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    exibir_resumo(modelo)
    print(f"\n  Loss: Focal Loss (gamma={CONFIG['focal_gamma']}, "
          f"alpha={CONFIG['focal_alpha']})")

    # ── 4. Treina ─────────────────────────────────────────────────────────
    print("\n[4/5] Iniciando treinamento...")
    batch      = CONFIG["batch_size"]
    steps_tr   = max(1, len(X_tr)  // batch)
    steps_val  = max(1, len(X_val) // batch)

    print(f"  Batch: {batch} | Steps/época: {steps_tr} | "
          f"Épocas: {CONFIG['epocas']}\n")

    historico = modelo.fit(
        criar_gerador(X_tr, y_tr, batch, augmentar=CONFIG["usar_augmentation"]),
        steps_per_epoch=steps_tr,
        epochs=CONFIG["epocas"],
        validation_data=criar_gerador(X_val, y_val, batch),
        validation_steps=steps_val,
        callbacks=obter_callbacks_ff(CONFIG["pasta_modelos"]),
        verbose=1
    )

    # ── 5. Salva e visualiza ──────────────────────────────────────────────
    print("\n[5/5] Salvando resultados...")
    modelo.save(os.path.join(CONFIG["pasta_modelos"], "modelo_ff_final.keras"))
    print("  ✓ Modelo final salvo: models/modelo_ff_final.keras")

    plotar_historico(historico)

    melhor_auc = max(historico.history["val_auc"])
    melhor_acc = max(historico.history["val_accuracy"])
    print(f"\n{'='*60}")
    print(f"  ✅ TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"  Melhor AUC:      {melhor_auc:.4f}")
    print(f"  Melhor Acurácia: {melhor_acc:.4f}")
    print(f"\n  Próximos passos:")
    print(f"  1. python utils/evaluator.py")
    print(f"  2. python utils/otimizar_limiar.py")
    print(f"  3. python main.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    treinar()