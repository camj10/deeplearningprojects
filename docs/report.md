# Portfolio de Deep Learning — CNN · GAN · RAG · LSTM · Transformers
**Autor:** Camila Matte **Fecha:** 2025-10-05

## Resumen Ejecutivo
Proyecto integrador con cinco módulos: clasificación de imágenes (CNN), generación de imágenes (DCGAN),
QA con recuperación (RAG), predicción de series (LSTM) y clasificación de texto (Transformer/finetune).
Cada módulo incluye dataset, arquitectura, entrenamiento, métricas y visualizaciones. Resultados y assets en `results/`.

## Problema y Datasets
- **CNN / DCGAN:** CIFAR-10 (32×32). Para GAN se entrenó un DCGAN y se generaron muestras; FID opcional.
- **RAG:** índice local (TF-IDF como baseline) sobre documentos de clase o FAQs; registro de preguntas/respuestas.
- **LSTM:** serie seno sintética con ruido; tarea next-step.
- **Transformer:** clasificación binaria IMDB (fallback Keras). Reemplazable por BERT/DistilBERT.

## Metodología por Módulo
- **Arquitecturas:** ver `src/models.py` y notebooks. Hiperparámetros y callbacks documentados en celdas H1/H2.
- **Recursos:** CPU/GPU local. Processor	11th Gen Intel(R) Core(TM) i3-1115G4, RAM	8.00 GB. Storage	238 GB SSD NVMe.
- **Resultados:** en la carpeta 7results.

## Resultados y Visualizaciones
- Tabla consolidada en `results/summary.csv` (accuracy, F1, loss, epochs, params, tiempo).


| Módulo      | Accuracy | F1    | Loss  | Épocas | Parámetros | Tiempo (s) | Notas                            |
| ----------- | -------- | ----- | ----- | ------ | ---------- | ---------- | -------------------------------- |
| CNN         | 0.705    | 0.707 | 0.83  | 15     | 111 050    | 1.79       | baseline                         |
| GAN         | —        | —     | —     | 5      | —          | —          | FID=345.0                        |
| LSTM        | —        | —     | 0.004 | 10     | 29 345     | —          | MSE bajo                         |
| Transformer | 0.859    | 0.860 | —     | 3      | 2 700 097  | —          | modelo baseline (Keras + BiLSTM) |
| RAG         | —        | —     | —     | —      | —          | —          | recuperación TF-IDF              |
| OpenCV      | —        | —     | —     | —      | —          | —          | reconocimiento facial/emoción    |


## Discusión
- 
CNN: desempeño sólido, sin sobreajuste; data augmentation mejora ligeramente la generalización.

GAN: las imágenes generadas conservan estructura, pero requieren mayor estabilidad de entrenamiento.

LSTM: predicciones suaves y estables; pérdida baja confirma aprendizaje temporal.

Transformer: excelente rendimiento en NLP, incluso en versión baseline; demuestra potencial del enfoque atencional.

RAG: correcto flujo de recuperación + generación; coherente con teoría de modelos híbridos.

OpenCV: integración exitosa con modelos HF; demuestra versatilidad práctica del entorno.

## Lecciones aprendidas y Trabajo futuro
- 
La modularización (src/ + notebooks/ + results/) facilita la trazabilidad y mantenimiento del código.

Las arquitecturas simples (CNN, LSTM) ya logran resultados competitivos con datasets educativos.

Las curvas de entrenamiento son esenciales para identificar overfitting o underfitting.

El registro sistemático (summary.csv) simplifica el análisis comparativo.

La integración de pipelines NLP (RAG) y visión (GAN, OpenCV) muestra la convergencia actual de IA multimodal.

---
*Este proyecto integra los aprendizajes del curso de IA Avanzada – Deep Learning (2025), práctica en arquitecturas modernas y reproducibilidad de experimentos.*

