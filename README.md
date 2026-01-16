# Predicción de dirección de penalti a partir de embeddings (gait)

Este proyecto trabaja **a partir de CSVs de embeddings** (características ya extraídas) para **predecir la dirección de un penalti**: izquierda / centro / derecha.

La idea principal es tratar cada penalti como una **secuencia de embeddings** (normalmente organizada por *steps* o “franjas” del cuerpo / partes temporales) y entrenar distintos modelos para clasificar la dirección final del tiro.

> Estado: WIP. El repositorio se centra en experimentación y análisis; la documentación se irá ampliando cuando el proyecto esté más “cerrado”.

---

## ¿Qué hace este proyecto?

A partir de los CSV de embeddings, el proyecto se encarga de:

- **Cargar y validar** los embeddings y sus etiquetas (por ejemplo `shoot_zone` con 3 clases).
- **Explorar** el dataset: distribución de clases (desbalance), estadísticas por *step*, correlaciones, etc.
- **Preparar entradas para modelos**:
  - usar embeddings por *step* (secuencia),
  - probar agregaciones (promedios, concatenaciones u otras fusiones),
  - reducir dimensionalidad o seleccionar subconjuntos (si aplica).
- **Entrenar y comparar modelos** de clasificación para la predicción de dirección:
  - baseline simples (p. ej. regresión logística / árboles),
  - redes (p. ej. MLP) y modelos secuenciales (según el experimento).
- **Evaluar** con métricas adecuadas para datos desbalanceados (por ejemplo **F1-macro**, además de accuracy).
- **Analizar interpretabilidad/impacto**: qué *steps* o componentes del embedding aportan más señal y cuándo los modelos fallan más.

---

## Datos de entrada (CSVs de embeddings)

Los CSV contienen, para cada penalti, un conjunto de embeddings (normalmente por *step*) junto con metadatos y la etiqueta.

Ejemplos de archivos (pueden variar según el extractor o el preentrenamiento):
- `baseline_*.csv`
- `gaitset_*.csv`
- `gaitpart_*.csv`
- `gaitgl_*.csv`
- `gln_*.csv`

> Nota: según el experimento, puede haber variantes del mismo embedding (p. ej. preentrenos distintos) y/o versiones “fusionadas” o “reducidas”.

---

## Experimentos (visión general)

El trabajo se organiza como iteraciones de experimentación (por ejemplo):
- establecer un **baseline** razonable,
- buscar configuraciones/hyperparámetros de forma más sistemática,
- mejorar estabilidad del entrenamiento (semillas, regularización, class weights, etc.),
- **fusionar embeddings** y comparar si mejora frente a usar uno solo,
- estudiar qué *steps* o partes del embedding son más informativas.

---

## Salidas / resultados

Según el experimento, se generan:
- tablas de métricas por embedding/modelo,
- comparativas entre enfoques (baseline vs secuenciales),
- análisis por *step* (importancia / contribución),
- gráficos de rendimiento (y, si aplica, matrices de confusión).

---

## Notas

- Este repo **empieza en los embeddings**: asume que los CSV ya existen.
- Es posible que los datos/CSVs completos no se publiquen por licencia/privacidad; en ese caso, el repo se orienta a reproducir el flujo con datos propios.

---

Dependencias típicas (orientativo):
- `python`, `numpy`, `pandas`
- `scikit-learn`
- `torch` (si ejecutas modelos deep)
- `optuna` (si reproduces búsquedas)

## Referencias

La motivación y el contexto del enfoque (gait embeddings, modelos secuenciales y evaluación) están documentados en la memoria y en los notebooks del proyecto.
