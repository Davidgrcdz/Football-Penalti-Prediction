# Football Penalty Prediction (Gait Embeddings)

Predicción de la dirección de un penalti (**3 clases: izquierda / centro / derecha**) a partir de **secuencias temporales de gait embeddings** ya extraídos y almacenados en **CSVs**. El repositorio implementa un pipeline experimental común para **4 arquitecturas** (MLP, LSTM, TCN y Transformer) y una batería de fases **Optuna + V1…V5**, incluyendo experimentos con **embeddings combinados** y **selección de steps (Top-K / Low-K)**.

> Estado: WIP. El repo está orientado a experimentación y análisis; la documentación se irá consolidando junto con la memoria del TFG.

---

## Idea general

Cada muestra corresponde a un **vídeo** (`video_ID`) representado como una secuencia de frames/pasos de tamaño variable `(T × D)`.  
Los CSVs siguen un esquema típico:

- `video_ID`: identificador del vídeo
- `shoot_zone`: etiqueta (derecha / centro / izquierda)
- `feat_*`: columnas numéricas del embedding por frame (p. ej. `feat_0 ... feat_n`)

El objetivo es entrenar modelos que reciben una secuencia `(T × D)` y predicen la clase final del disparo, evaluando con métricas robustas a desbalance (especialmente **F1-macro**).

---

## ¿Qué hace este proyecto?

A partir de los CSVs de embeddings, el proyecto se encarga de:

- **Cargar y validar** embeddings y etiquetas.
- **Explorar** el dataset: distribución de clases, análisis por step, estadísticas y checks de calidad.
- **Preparar entradas para modelos**:
  - secuencias completas por vídeo (variable length),
  - agregaciones/fusiones (p. ej. mean / concat / reduced),
  - selección de subconjuntos temporales (**Top-K / Low-K steps**) cuando aplica.
- **Entrenar y comparar modelos** de clasificación con un protocolo común:
  - MLP (vectorial), LSTM (secuencial), TCN (convolucional temporal), Transformer (autoatención).
- **Evaluar** resultados y generar comparativas (tablas, figuras, matrices de confusión y curvas PR/ROC cuando corresponde).
- **Analizar el impacto**: qué variantes de embedding o qué steps aportan más señal y en qué casos fallan más los modelos.

---

## Experimentos (visión general)

La experimentación se estructura en fases:

- **Optuna**: búsqueda de hiperparámetros (cuando aplica) con validación estratificada por `video_ID`.
- **V1**: entrenamiento con mejores parámetros por embedding (base de comparación).
- **V2**: entrenamientos con **presets manuales** (configuraciones diseñadas/ajustadas).
- **V3**: re-entrenamiento de **mejores configuraciones** (desde históricos) con receta estabilizada.
- **V4**: experimentos con **embeddings combinados** (p. ej. `concat/mean/reduced`).
- **V5**: selección temporal por **Top-K** (steps más informativos) y pruebas con **Low-K** (steps menos informativos).

---

## Estructura del repositorio

- `src/`  
  Código modular en `.py` (modelos, entrenamiento, preparación de datos y utilidades).  
  Aquí viven las implementaciones de las 4 arquitecturas y helpers comunes (IDs, presets, guardado de checkpoints, etc.).

- `results/`  
  CSVs de resultados por arquitectura y fase (métricas por run/config/embedding).

- `saved_models/`  
  Checkpoints guardados cuando un run supera un umbral de F1. Los nombres incluyen `id/fase/embedding` para trazabilidad.

- `presets_configs/`  
  Presets (configuraciones) generados desde históricos: selección top-k por CSV, deduplicación por hiperparámetros, etc.

- `embedding_audit_report/`  
  Reportes auxiliares para validar/inspeccionar datasets de embeddings.

- Notebooks principales:
  - `embeddings_preparation.ipynb`: Preparación de CSVs de embeddings, validación y guardado.
  - `experiments_pipeline.ipynb`: Ejecución de las 4 arquitecturas y las fases V1…V5 (MLP/LSTM/TCN/Transformer).
  - `evaluation_pipeline.ipynb`: Evaluación repetida con múltiples seeds (pipeline reproducible).
  - `result_tables_plots.ipynb`: Agrega CSVs finales y construye tablas/figuras.
---

## Datos de entrada (CSVs de embeddings)

Ejemplos de archivos (pueden variar según el extractor / preentrenamiento):

- `baseline_*.csv`
- `gaitset_*.csv`
- `gaitpart_*.csv`
- `gaitgl_*.csv`
- `gln_*.csv`
