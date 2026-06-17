# utils/evaluator.py
# Avalia o modelo treinado com métricas completas para o TCC
# Execute: python utils/evaluator.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────

PASTA_MODELOS = "models"
PASTA_LOGS    = "logs"
LIMIAR        = 0.5


# ─────────────────────────────────────────
# FOCAL LOSS — necessária para carregar o modelo
# ─────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.70):
    """
    Focal Loss registrada para compatibilidade ao carregar o modelo.
    Deve ser idêntica à usada no train.py.
    """
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
# CLASSE PRINCIPAL DE AVALIAÇÃO
# ─────────────────────────────────────────

class Avaliador:
    """
    Carrega o modelo salvo, roda predições no conjunto de teste
    e gera relatório completo de métricas e gráficos.
    """

    def __init__(self, caminho_modelo=None, limiar=LIMIAR):
        self.limiar = limiar

        if caminho_modelo is None:
            caminho_modelo = os.path.join(PASTA_MODELOS, "melhor_modelo.keras")

        if not os.path.exists(caminho_modelo):
            print(f"ERRO: Modelo não encontrado em '{caminho_modelo}'")
            print("Execute python train.py primeiro para treinar o modelo.")
            sys.exit(1)

        print(f"[Avaliador] Carregando modelo: {caminho_modelo}")
        self.modelo = tf.keras.models.load_model(
            caminho_modelo,
            custom_objects={"loss": focal_loss()}
        )
        print("[Avaliador] Modelo carregado com sucesso.\n")

    # ─────────────────────────────────────
    # MÉTODO: Carregar conjunto de teste
    # ─────────────────────────────────────

    def carregar_teste(self, X_teste=None, y_teste=None):
        if X_teste is not None and y_teste is not None:
            self.X_teste = X_teste
            self.y_teste = y_teste
            return

        caminho_X = os.path.join(PASTA_MODELOS, "X_teste.npy")
        caminho_y = os.path.join(PASTA_MODELOS, "y_teste.npy")

        if not os.path.exists(caminho_X) or not os.path.exists(caminho_y):
            print("ERRO: Conjunto de teste não encontrado.")
            print("Execute python train.py para gerar os dados de teste.")
            sys.exit(1)

        self.X_teste = np.load(caminho_X)
        self.y_teste = np.load(caminho_y)

        print(f"Conjunto de teste carregado:")
        print(f"  Total:  {len(self.X_teste)} imagens")
        print(f"  REAL:   {(self.y_teste == 0).sum()}")
        print(f"  FAKE:   {(self.y_teste == 1).sum()}\n")

    # ─────────────────────────────────────
    # MÉTODO: Gerar predições
    # ─────────────────────────────────────

    def predizer(self, batch_size=8):
        print("Gerando predições no conjunto de teste...")
        self.probs = self.modelo.predict(
            self.X_teste,
            batch_size=batch_size,
            verbose=1
        ).flatten()

        self.preds = (self.probs >= self.limiar).astype(int)

        reais = (self.preds == 0).sum()
        fakes = (self.preds == 1).sum()
        print(f"\nResultado das predições:")
        print(f"  Predito como REAL: {reais}")
        print(f"  Predito como FAKE: {fakes}\n")

    # ─────────────────────────────────────
    # MÉTODO: Calcular e exibir métricas
    # ─────────────────────────────────────

    def calcular_metricas(self):
        acuracia = accuracy_score(self.y_teste, self.preds)
        precisao = precision_score(self.y_teste, self.preds, zero_division=0)
        recall   = recall_score(self.y_teste, self.preds, zero_division=0)
        f1       = f1_score(self.y_teste, self.preds, zero_division=0)

        self.metricas = {
            "Acurácia": acuracia,
            "Precisão": precisao,
            "Recall":   recall,
            "F1-score": f1,
        }

        print("=" * 50)
        print("  MÉTRICAS DE AVALIAÇÃO")
        print("=" * 50)
        for nome, valor in self.metricas.items():
            barra = "█" * int(valor * 20)
            print(f"  {nome:<12} {valor:.4f}  {barra}")
        print("=" * 50)

        print("\nRelatório detalhado por classe:")
        print(classification_report(
            self.y_teste, self.preds,
            target_names=["REAL", "FAKE"],
            digits=4
        ))

        return self.metricas

    # ─────────────────────────────────────
    # MÉTODO: Matriz de confusão
    # ─────────────────────────────────────

    def plotar_matriz_confusao(self, ax=None, salvar=False):
        cm      = confusion_matrix(self.y_teste, self.preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        criar_fig = ax is None
        if criar_fig:
            fig, ax = plt.subplots(figsize=(7, 6))

        anotacoes = np.array([
            [f"{cm[i][j]}\n({cm_norm[i][j]*100:.1f}%)"
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        sns.heatmap(
            cm_norm, annot=anotacoes, fmt="",
            cmap="Blues", ax=ax,
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"],
            linewidths=0.5, linecolor="gray",
            cbar_kws={"label": "Proporção"},
            annot_kws={"size": 13}
        )

        ax.set_xlabel("Predição do Modelo", fontsize=12)
        ax.set_ylabel("Rótulo Real",        fontsize=12)
        ax.set_title("Matriz de Confusão",  fontsize=13, fontweight="bold")

        ax.text(-0.35, 0.5, "VN", color="#2ecc71", fontsize=11,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.82,  0.5, "FP", color="#2ecc71", fontsize=11,
                fontweight="bold", transform=ax.transAxes)
        ax.text(-0.35, 0.0, "FN", color="#e74c3c", fontsize=11,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.82,  0.0, "VP", color="#e74c3c", fontsize=11,
                fontweight="bold", transform=ax.transAxes)

        if criar_fig:
            plt.tight_layout()
            if salvar:
                caminho = os.path.join(PASTA_LOGS, "matriz_confusao.png")
                plt.savefig(caminho, dpi=150, bbox_inches="tight")
                print(f"  ✓ Matriz salva em: {caminho}")
            plt.show()

    # ─────────────────────────────────────
    # MÉTODO: Curva ROC
    # ─────────────────────────────────────

    def plotar_curva_roc(self, ax=None, salvar=False):
        fpr, tpr, _ = roc_curve(self.y_teste, self.probs)
        auc_score   = auc(fpr, tpr)

        criar_fig = ax is None
        if criar_fig:
            fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(fpr, tpr, color="#4A90E2", linewidth=2.5,
                label=f"Modelo (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linewidth=1,
                linestyle="--", label="Aleatório (AUC = 0.50)")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#4A90E2")

        ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
        ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
        ax.set_title("Curva ROC", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        if criar_fig:
            plt.tight_layout()
            if salvar:
                caminho = os.path.join(PASTA_LOGS, "curva_roc.png")
                plt.savefig(caminho, dpi=150, bbox_inches="tight")
                print(f"  ✓ Curva ROC salva em: {caminho}")
            plt.show()

    # ─────────────────────────────────────
    # MÉTODO: Distribuição de probabilidades
    # ─────────────────────────────────────

    def plotar_distribuicao(self, ax=None, salvar=False):
        criar_fig = ax is None
        if criar_fig:
            fig, ax = plt.subplots(figsize=(8, 5))

        probs_real = self.probs[self.y_teste == 0]
        probs_fake = self.probs[self.y_teste == 1]

        ax.hist(probs_real, bins=40, alpha=0.6, color="#2ecc71",
                label=f"REAL (n={len(probs_real)})", density=True)
        ax.hist(probs_fake, bins=40, alpha=0.6, color="#e74c3c",
                label=f"FAKE (n={len(probs_fake)})", density=True)
        ax.axvline(x=self.limiar, color="black", linewidth=1.5,
                   linestyle="--", label=f"Limiar = {self.limiar}")

        ax.set_xlabel("Probabilidade predita (0=REAL → 1=FAKE)", fontsize=12)
        ax.set_ylabel("Densidade", fontsize=12)
        ax.set_title("Distribuição de Probabilidades por Classe",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        if criar_fig:
            plt.tight_layout()
            if salvar:
                caminho = os.path.join(PASTA_LOGS, "distribuicao_probs.png")
                plt.savefig(caminho, dpi=150, bbox_inches="tight")
                print(f"  ✓ Distribuição salva em: {caminho}")
            plt.show()

    # ─────────────────────────────────────
    # MÉTODO: Gráfico de barras das métricas
    # ─────────────────────────────────────

    def plotar_metricas(self, ax=None, salvar=False):
        criar_fig = ax is None
        if criar_fig:
            fig, ax = plt.subplots(figsize=(7, 5))

        nomes   = list(self.metricas.keys())
        valores = list(self.metricas.values())
        cores   = ["#4A90E2", "#E27B4A", "#2ecc71", "#9B59B6"]

        barras = ax.bar(nomes, valores, color=cores, width=0.5,
                        edgecolor="white", linewidth=0.5)

        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.01,
                f"{valor:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold"
            )

        ax.set_ylim([0, 1.12])
        ax.set_ylabel("Valor", fontsize=12)
        ax.set_title("Métricas de Avaliação", fontsize=13, fontweight="bold")
        ax.axhline(y=0.9, color="gray", linestyle="--",
                   alpha=0.5, label="Meta: 0.90")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        if criar_fig:
            plt.tight_layout()
            if salvar:
                caminho = os.path.join(PASTA_LOGS, "metricas.png")
                plt.savefig(caminho, dpi=150, bbox_inches="tight")
                print(f"  ✓ Métricas salvas em: {caminho}")
            plt.show()

    # ─────────────────────────────────────
    # MÉTODO: Relatório visual completo
    # ─────────────────────────────────────

    def relatorio_completo(self):
        os.makedirs(PASTA_LOGS, exist_ok=True)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Relatório de Avaliação — Detecção de Fraudes Faciais",
            fontsize=15, fontweight="bold", y=1.01
        )

        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        self.plotar_matriz_confusao(ax=ax1)
        self.plotar_curva_roc(ax=ax2)
        self.plotar_distribuicao(ax=ax3)
        self.plotar_metricas(ax=ax4)

        caminho = os.path.join(PASTA_LOGS, "relatorio_completo_ff.png")
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"\n✅ Relatório completo salvo em: {caminho}")
        print("   Use essa imagem no seu TCC!")


# ─────────────────────────────────────────
# PIPELINE DE AVALIAÇÃO
# ─────────────────────────────────────────

def avaliar():
    print("=" * 55)
    print("  AVALIAÇÃO — Detecção de Fraudes Faciais")
    print("=" * 55)

    os.makedirs(PASTA_LOGS, exist_ok=True)

    avaliador = Avaliador()
    avaliador.carregar_teste()
    avaliador.predizer()
    avaliador.calcular_metricas()
    avaliador.relatorio_completo()


if __name__ == "__main__":
    avaliar()