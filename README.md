# Project ML Portfolio — CNN, GANs, RAGs, LSTM, Transformers
## Este repositorio contiene el proyecto final del curso **IA Avanzada: Deep Learning **.
## Integra implementaciones de **CNN**, **GAN**, **RAG**, **LSTM** y **Transformers**, incluyendo entrenamiento, evaluación y visualización de resultados.


## 📊 Resultados destacados
- **CNN (CIFAR-10)** – accuracy ≈ 0.70  
- **Transformer (IMDB)** – accuracy ≈ 0.86  
- **LSTM** – MSE = 0.004  
- **GAN** – FID ≈ 345  
- **RAG** – flujo completo de recuperación + generación  


## Estructura
```
project-ml-portfolio/
├─ notebooks/
│  ├─ 0_overview.ipynb
│  ├─ 1_cnn_classification.ipynb
│  ├─ 2_dcgan_generation.ipynb
│  ├─ 3_rag_qa.ipynb
│  ├─ 4_lstm_time_series.ipynb
│  └─ 5_transformer_finetune.ipynb
├─ src/
│  ├─ data.py
│  ├─ models.py
│  ├─ train.py
│  └─ utils.py
├─ requirements.txt
├─ docs/
│  └─ report.md
├─ results/
├─ presentation/
└─ assets/
```

## Reproducibilidad
1. Crear entorno:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2. Abrir `notebooks/0_overview.ipynb` y seguir el orden sugerido.

