# models/modelo.py
# Define a arquitetura do modelo de classificação REAL vs FAKE
# Usa MobileNetV2 como base (transfer learning) + camadas densas no topo

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import os

# ─────────────────────────────────────────
# CONFIGURAÇÕES DO MODELO
# ─────────────────────────────────────────

IMG_SIZE    = (224, 224)       # tamanho de entrada esperado pelo MobileNetV2
CANAIS      = 3                # RGB
INPUT_SHAPE = (*IMG_SIZE, CANAIS)  # (224, 224, 3)
NUM_CLASSES = 1                # saída binária: 1 neurônio com sigmoid


# ─────────────────────────────────────────
# FUNÇÃO: Construir o modelo
# ─────────────────────────────────────────

def construir_modelo(
    taxa_aprendizado=1e-4,
    dropout_rate=0.4,
    descongelar_camadas=30
):
    """
    Constrói o modelo de detecção de fraudes faciais.

    Arquitetura:
        [MobileNetV2 pré-treinada] → GlobalAveragePooling → Dense → Dropout → Saída

    Parâmetros:
        taxa_aprendizado:   velocidade de atualização dos pesos (padrão: 0.0001)
        dropout_rate:       fração de neurônios desativados no treino (evita overfitting)
        descongelar_camadas: quantas camadas finais da base treinar junto

    Retorna:
        modelo Keras compilado, pronto para treinar
    """

    # ── 1. BASE PRÉ-TREINADA (MobileNetV2) ──────────────────────────────────
    #
    # MobileNetV2 foi treinada no ImageNet (1.4 milhão de imagens, 1000 classes).
    # Ela já aprendeu a detectar bordas, texturas, formas e padrões visuais.
    # Aproveitamos esse conhecimento para o nosso problema (transfer learning).
    #
    # include_top=False → remove as camadas de classificação originais
    # weights='imagenet' → carrega os pesos pré-treinados (baixa ~14MB na 1ª vez)

    base = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )

    # Congela TODA a base inicialmente
    # (não queremos destruir o conhecimento já aprendido no ImageNet)
    base.trainable = False

    # Descongela as últimas N camadas para fine-tuning
    # Essas camadas aprendem padrões mais específicos (ex: textura de pele)
    if descongelar_camadas > 0:
        for camada in base.layers[-descongelar_camadas:]:
            # BatchNormalization deve ficar congelada mesmo no fine-tuning
            if not isinstance(camada, layers.BatchNormalization):
                camada.trainable = True

    total_treinaveis = sum(1 for c in base.layers if c.trainable)
    print(f"[Modelo] Camadas treináveis na base: {total_treinaveis} / {len(base.layers)}")

    # ── 2. CABEÇA DE CLASSIFICAÇÃO ───────────────────────────────────────────
    #
    # Adicionamos nossas próprias camadas no topo da base.
    # Essas camadas aprendem a classificar REAL vs FAKE.

    entradas = tf.keras.Input(shape=INPUT_SHAPE, name="entrada_imagem")

    # Passa pela base (sem atualizar a maioria dos pesos)
    x = base(entradas, training=False)

    # GlobalAveragePooling: transforma o mapa de features (7x7x1280) em vetor (1280,)
    # Mais leve e menos propenso a overfitting que o Flatten
    x = layers.GlobalAveragePooling2D(name="pooling_global")(x)

    # Camada densa para aprender combinações de features
    x = layers.Dense(256, activation="relu", name="densa_256")(x)

    # Dropout: desativa aleatoriamente 40% dos neurônios durante o treino
    # Força o modelo a não depender de neurônios específicos → mais robusto
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Camada densa menor para refinar a decisão
    x = layers.Dense(64, activation="relu", name="densa_64")(x)

    # Saída: 1 neurônio com sigmoid
    # Saída próxima de 0 → REAL | Saída próxima de 1 → FAKE
    saida = layers.Dense(1, activation="sigmoid", name="saida")(x)

    # ── 3. MONTAGEM E COMPILAÇÃO ─────────────────────────────────────────────

    modelo = Model(inputs=entradas, outputs=saida, name="FraudeFacialDetector")

    modelo.compile(
        optimizer=Adam(learning_rate=taxa_aprendizado),

        # Binary crossentropy: função de perda padrão para classificação binária
        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precisao"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    return modelo


# ─────────────────────────────────────────
# FUNÇÃO: Callbacks de treinamento
# ─────────────────────────────────────────

def obter_callbacks(pasta_modelos="models", paciencia=5):
    """
    Retorna lista de callbacks para monitorar e controlar o treinamento.

    Callbacks são funções chamadas automaticamente ao fim de cada época.

    Parâmetros:
        pasta_modelos: onde salvar o melhor modelo
        paciencia:     épocas sem melhora antes de parar ou reduzir LR
    """
    os.makedirs(pasta_modelos, exist_ok=True)

    callbacks = [

        # ── Salva o MELHOR modelo automaticamente ────────────────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(pasta_modelos, "melhor_modelo.keras"),
            monitor="val_loss",        # monitora perda na validação
            save_best_only=True,       # só salva se melhorou
            verbose=1
        ),

        # ── Para o treino se não melhorar por N épocas ───────────────────────
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=paciencia,        # aguarda N épocas sem melhora
            restore_best_weights=True, # restaura os melhores pesos ao parar
            verbose=1
        ),

        # ── Reduz a taxa de aprendizado se travar ────────────────────────────
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,       # multiplica LR por 0.3 quando travar
            patience=3,       # aguarda 3 épocas antes de reduzir
            min_lr=1e-7,      # limite mínimo da taxa de aprendizado
            verbose=1
        ),

        # ── Salva logs para visualização no TensorBoard ──────────────────────
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join("logs", "historico_treino.csv"),
            append=False
        ),
    ]

    return callbacks


# ─────────────────────────────────────────
# FUNÇÃO: Resumo visual do modelo
# ─────────────────────────────────────────

def exibir_resumo(modelo):
    """Exibe resumo da arquitetura e conta parâmetros."""
    modelo.summary(line_length=80)

    total     = modelo.count_params()
    treinaveis = sum(
        tf.size(w).numpy() for w in modelo.trainable_weights
    )
    nao_treinaveis = total - treinaveis

    print(f"\n{'─'*50}")
    print(f"  Total de parâmetros:          {total:,}")
    print(f"  Parâmetros treináveis:        {treinaveis:,}")
    print(f"  Parâmetros congelados:        {nao_treinaveis:,}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────
# TESTE RÁPIDO (rodar direto: python models/modelo.py)
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Construindo modelo...\n")
    modelo = construir_modelo()
    exibir_resumo(modelo)

    # Teste com imagem falsa para verificar que a arquitetura funciona
    import numpy as np
    imagem_teste = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predicao = modelo.predict(imagem_teste, verbose=0)
    print(f"Teste de predição com imagem aleatória:")
    print(f"  Saída bruta:  {predicao[0][0]:.4f}")
    print(f"  Interpretação: {'FAKE' if predicao[0][0] > 0.5 else 'REAL'}")
    print("\n✅ Modelo construído com sucesso!")