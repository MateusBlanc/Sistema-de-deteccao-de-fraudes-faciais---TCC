# Sistema de Deteccao de Fraudes Faciais (Deepfakes)

Trabalho de Conclusao de Curso — sistema em Python que classifica rostos
como **REAL** ou **FAKE**, detectando deepfakes baseados em manipulacao
facial. Treinado com o dataset FaceForensics++ usando MobileNetV2 e
transfer learning.

## Resultados

Avaliacao no conjunto de teste (FaceForensics++):

| Metrica   | Valor  |
|-----------|--------|
| AUC-ROC   | 0.965  |
| Acuracia  | 89.0%  |
| Recall    | 91.6%  |
| F1-score  | 89.2%  |

## Tecnicas de manipulacao detectadas

O modelo foi treinado para reconhecer tres tecnicas do FaceForensics++:
Deepfakes, FaceSwap e NeuralTextures. Para rostos sinteticos gerados
puramente por GAN (ex.: StyleGAN), o desempenho e limitado — discutido
como trabalho futuro.

## Tecnologias

Python 3.10, TensorFlow 2.15, OpenCV 4.9, MediaPipe 0.10, scikit-learn.
MobileNetV2 pre-treinada (ImageNet) com fine-tuning e Focal Loss.

## Estrutura

```
.
├── main.py                  Sistema de deteccao em tempo real (webcam)
├── train.py                 Treinamento do modelo
├── ff_pipeline.py           Extracao de frames/rostos do FF++
├── testar_ff.py             Teste com imagens
├── testar_video.py          Teste com videos
├── models/
│   └── modelo.py            Arquitetura MobileNetV2
├── utils/                   Modulos de apoio
│   ├── preprocessor.py      Deteccao e recorte de rostos
│   ├── augmentation.py      Data augmentation
│   ├── evaluator.py         Metricas e relatorios
│   ├── otimizar_limiar.py   Calibracao do limiar
│   ├── classificador_ensemble.py
│   ├── analisador_frequencia.py
│   └── seguranca.py
└── logs/                    Graficos e CSVs dos resultados
```

## Como executar

```bash
# 1. Ambiente
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Preparar dataset (requer acesso ao FaceForensics++)
#    Apos baixar os videos, extrair os rostos:
python ff_pipeline.py

# 3. Treinar
python train.py

# 4. Avaliar e calibrar
python utils/evaluator.py
python utils/otimizar_limiar.py

# 5. Rodar ao vivo
python main.py
```

## O modelo treinado

O arquivo `models/melhor_modelo.keras` e versionado via **Git LFS**.
Para obte-lo apos clonar:

```bash
git lfs install
git lfs pull
```

## Dataset

FaceForensics++ (Rossler et al., ICCV 2019). O acesso requer
preenchimento do formulario oficial dos autores. O dataset nao esta
incluido neste repositorio.

```
@inproceedings{roessler2019faceforensicspp,
  author    = {Andreas Rossler and Davide Cozzolino and Luisa Verdoliva
               and Christian Riess and Justus Thies and Matthias Niessner},
  title     = {FaceForensics++: Learning to Detect Manipulated Facial Images},
  booktitle = {ICCV},
  year      = {2019}
}
```

## Limitacoes

- Detecta as tecnicas do FF++; rostos sinteticos por GAN tem desempenho reduzido.
- Domain shift entre frames de treino e captura por webcam.
- Desempenho cai sob recaptura de tela e forte compressao.
- Nao realiza autenticacao de identidade.