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
- **Arquitecturas:**
- 1. CNN – Convolutional Neural Network (CIFAR-10)
Arquitectura: 2 capas Conv2D (32 y 64 filtros), MaxPooling, Flatten, Dense(128), Dropout(0.5), salida Softmax(10).
Optimizador: Adam (lr=0.001).
Entrenamiento: 15 épocas, batch 64.
Data Augmentation: rotación ±10°, traslación 0.1, flip horizontal.
Resultados: Accuracy=0.705 / F1=0.707 / Loss=0.83.
Observación: modelo estable, sin sobreajuste; mejora leve con augmentation.
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/d27b0c32-d29a-4b29-9e0c-754281d666f9" />
<img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/76d59354-975f-4378-b045-d4c78958c103" />

2. GAN – Deep Convolutional GAN
Generador: capas Conv2DTranspose + BatchNorm + ReLU.
Discriminador: Conv2D + LeakyReLU + Dropout + Sigmoid.
Entrenamiento: 5 épocas sobre imágenes CIFAR-10 (32×32 px).
Métrica: FID.
Visualización: grilla de imágenes generadas por época.
Observación: imágenes coherentes en color y forma; FID alto indica espacio de mejora (más épocas, normalización mejorada).
<img width="1200" height="1200" alt="image" src="https://github.com/user-attachments/assets/e07654b1-7b61-4792-8ad2-231eb28d4828" />

4. LSTM – Modelado de Secuencias (Serie Temporal)
Arquitectura: 2 capas LSTM(64) + Dense(lineal).
Loss: MSE (Error Cuadrático Medio).
Entrenamiento: 10 épocas.
Resultados: Loss=0.004 (val).
Visualización: comparación serie real vs predicha.
Observación: el modelo capta correctamente la periodicidad; sin sobreajuste.
<img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/6cd715e0-9c85-477c-bf7f-14cdba7d1e35" />

6. Transformer – Clasificación de Texto (IMDB)
Modelo utilizado: baseline Keras (Embedding + Bidirectional LSTM + Dense(sigmoid)), correspondiente a la versión usada en clase.
Entrenamiento: 3 épocas, batch 64.
Resultados: Accuracy=0.859 / F1=0.860.
Observación: el modelo logra buena generalización en tareas de NLP; la estructura admite sustitución futura por un modelo Transformer real (Hugging Face).
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/0b2d4c91-d229-4444-95fc-32937533dac2" />
<img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/7bdcfc3f-6723-4cf7-89a2-6407bc436e68" />

5. RAG – Retrieval-Augmented Generation (QA con documentos locales)
- Método: recuperación de contexto usando **TF-IDF** y **similitud del coseno** con `scikit-learn`.
- Generación: concatenación de los fragmentos más relevantes y construcción de una respuesta heurística basada en el contexto recuperado.
- Dataset: documentos `.txt` locales en español (FAQ o textos de clase).
- Implementación: `TfidfVectorizer(max_features=5000`
- Evaluación* relevancia cualitativa; las respuestas son coherentes con el texto fuente.
- Visualización: ejemplos de pregunta, contexto y respuesta guardados en `results/rag/examples.jsonl`.
- Observación: el módulo demuestra el flujo completo de un RAG básico.  


6. OpenCV + Hugging Face – Reconocimiento de emociones y edad
Modelos: dima806/facial_emotions_image_detection y Robys01/facial_age_estimator.
Framework: OpenCV + Transformers.
Aplicación: detección en tiempo real vía cámara o imagen.
Observación: demuestra integración de visión por computadora con modelos preentrenados.

más detalles en `src/models.py` y `/notebooks`.

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
En la CNN, el desempeño fue sólido y no se observó sobreajuste, accuracy de 0.705, con un F1-score de 0.707 y una pérdida (loss) de 0.83 sobre el conjunto de prueba. La implementación de data augmentation ayudó a mejorar un poco la capacidad de generalización del modelo, mostrando la importancia de esta técnica incluso en conjuntos de datos pequeños como CIFAR-10.
El modelo GAN pudo generar imágenes con estructuras reconocibles, aunque todavía prsentaba cierta inestabilidad en el entrenamiento, el FID aprox 345, lo que indica que aún existe margen de mejora con más épocas o un ajuste de hiperparámetros. Se puede notar también que ajustes pequeños pueden influir mucho en la calidad visual de las muestras generadas.
Los resultados con el LSTM fueron estables y tuvo baja pérdida en la validación, la pérdida de validación (MSE) alcanzó un valor de 0.004. 
El módulo Transformer tuvo buen rendimiento en la tarea de clasificación de texto, accuracy de 0.859 y un F1-score de 0.860 incluso utilizando la versión baseline basada en Keras, se percibe que esta arquitecura tiene mucha eficacia en problemas de NLP.
En el RAG, el flujo de recuperación y generación funcionó correctamente, y las respuestas dadas fueron coherentes con los documentos originales, me pareció un ejemplo claro de cómo se puede combinar la búsqueda de información con generación de texto dentro de un mismo sistema.

El módulo de OpenCV permitió aplicar modelos preentrenados para el reconocimiento de emociones y estimación de edad en imágenes, con resultados coherentes con las expresiones faciales detectadas, así integrándose modelos de visión por computadora en sistemas prácticos de IA.

## Lecciones aprendidas y Trabajo futuro
-
En el desarrollo de este proyecto sobre las distintas arquitecturas de Deep Learning, pude organizar el trabajo de forma modular (con carpetas como src/, notebooks/ y results/), esto me ayudó a entender mejor cómo se estructura un proyecto real y a mantener el código ordenado.

Las curvas de entrenamiento son muy útiles para observar si el modelo aprende o comenzaba a sobreajustarse, siendo un indicador que me permitió ajustar parámetros y entender mejor cómo estaba aprendiendo el modelo.

También aprendí el valor de registrar los resultados en archivos como summary.csv, porque permite comparar de manera rápida la performance de varios modelos dentro de un mismo proyecto, es una herramienta práctica que podría aplicarse fácilmente a proyectos más grandes o de investigación.

Como trabajo futuro, me gustaría profundizar el desarrollo de modelos probando otras técnicas para mejorar la calidad de las respuestas que da.


---
*Este proyecto integra los aprendizajes del curso de IA Avanzada – Deep Learning (2025), práctica en arquitecturas modernas y reproducibilidad de experimentos.*

