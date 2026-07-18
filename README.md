# ML_RENDIMIENTO — Predicción de abandono estudiantil
### Trabajo realizado por Paula Comas y Ana Corrochano para el bootcamp de Data Science y Artificial Inteligence

Modelo de clasificación multiclase para identificar, **en el momento de la matrícula**, qué
estudiantes tienen mayor riesgo de abandonar los estudios, con el objetivo de activar medidas de
apoyo tempranas (becas, tutorías, planes de pago flexibles, seguimiento académico).

## Problema y enfoque

- **Tipo de problema:** clasificación multiclase — el modelo predice si un estudiante recién
  matriculado será `Dropout` (abandona), `Enrolled` (sigue matriculado, trayectoria aún no resuelta)
  o `Graduate` (se gradúa).
- **Métrica principal:** F1 macro (el target tiene un desbalance moderado: 49.9% Graduate, 32.1%
  Dropout, 17.9% Enrolled — accuracy premiaría ignorar la clase minoritaria).
- **Decisión clave — sin data leakage:** el dataset incluye 12 variables `Curricular units 1st/2nd
  sem (*)` que predicen el target casi perfectamente (F1 macro 0.713 vs 0.566 sin ellas), pero solo
  se conocen *después* de que el estudiante ya cursó uno o dos semestres. Como el objetivo es
  intervenir **lo antes posible**, estas variables se excluyen del modelo final a propósito. El F1
  macro del modelo (~0.57) es más bajo de lo que el dataset permitiría, pero es el número honesto
  para un modelo usable en el momento de matrícula.

## Estructura del proyecto

```
ML_rendimiento/
├── src/
│   ├── data_sample/
│   │   └── data.csv                              # Dataset de origen
|   ├── models/
│   │   └── modelo_dropout_final.joblib           # Modelo final, listo para producción
|   ├── img/                              
│   ├── notebooks/
│   │   ├── src/                     
│   │   │   └── artifacts/                        # Artefactos intermedios (.joblib) entre notebooks
│   │   │       └── predicciones_dropout.csv      # Salida de predict_batch.py
│   │   ├── 01_EDA.ipynb                          # Análisis exploratorio
│   │   ├── 02_feature-eng.ipynb                  # Selección de variables, decisión de leakage
│   │   ├── 03_preprocessing.ipynb                # Split train/test + pipeline de preprocesado
│   │   ├── 04_modeling.ipynb                     # Comparativa de modelos + tuning
│   │   └── 05_evaluation.ipynb                   # Evaluación final + guardado del modelo
│   └── utils/
│       └── predict_batch.py                      # Script para generar predicciones en batch
├── venv/
├── ML_RENDIMIENTO_Informe.pdf
├── main.ipynb
├── requirements.txt
└── README.md
```

> ⚠️ **Nota sobre rutas:** Si ejecutas un notebook, asegúrate de que
> el directorio de trabajo de Jupyter sea esa carpeta. El script `predict_batch.py`, en cambio, está
> pensado para ejecutarse desde la **raíz del proyecto** (ver sección de uso más abajo).

## Pipeline de datos (cadena de artefactos)

Cada notebook carga el `.joblib` del anterior y genera el suyo, para poder ejecutarse de forma
independiente sin repetir pasos ya hechos:

| Notebook | Entrada | Salida | Qué hace |
|---|---|---|---|
| `01_EDA` | `data.csv` | `01_eda.joblib` | Limpieza inicial, ANOVA/Chi² por variable vs target |
| `02_feature-eng` | `01_eda.joblib` | `02_feature_eng.joblib` | Tipificación manual de variables, exclusión de leakage y variables no significativas → 20 variables finales |
| `03_preprocessing` | `02_feature_eng.joblib` | `03_preprocessing.joblib` | Split train/test (80/20, estratificado), `ColumnTransformer` (StandardScaler + OneHotEncoder + passthrough) |
| `04_modeling` | `03_preprocessing.joblib` | `04_modeling.joblib` | Baseline, comparativa de 4 algoritmos, `class_weight` vs SMOTE, `RandomizedSearchCV` |
| `05_evaluation` | `04_modeling.joblib` | `modelo_dropout_final.joblib` | Evaluación en test, interpretabilidad, guardado del modelo final |

### Variables descartadas y por qué

- **Leakage (12 vars):** todas las `Curricular units 1st/2nd sem (*)` — se conocen después de la
  matrícula.
- **No significativas (4 vars):** `Nacionality` (Chi²=0.242, además 97.5% concentrado en una sola
  categoría), `Educational special needs` (p=0.725), `International` (p=0.527), `Inflation rate`
  (ANOVA p=0.175).
- **Variables finales (20):** datos sociodemográficos, académicos previos y económicos disponibles
  en el momento de matrícula (edad, notas previas, curso, situación económica, etc.).

## Resultados

| Etapa | F1 macro (CV) |
|---|---|
| Baseline (clase mayoritaria) | 0.222 |
| Comparativa — mejor (SVM RBF) | 0.581 |
| **Logistic Regression (elegido)** | 0.570 → **0.575** tras tuning |
| Con `Curricular units` (leakage, solo referencia) | 0.713 |

**Modelo elegido:** Logistic Regression, no el SVM con mejor score, porque el rendimiento es
prácticamente idéntico pero es interpretable y más rápido de entrenar — necesario para que el
equipo de orientación académica entienda *por qué* un estudiante está en riesgo, no solo la
predicción.

**Variables más influyentes para `Dropout`:** `Tuition fees up to date`, `Scholarship holder`,
`Debtor`, junto con la edad de matriculación y las notas previas.

**Limitación conocida:** la clase `Enrolled` es la más difícil de predecir — representa una
trayectoria aún no resuelta en el momento de matrícula, no un fallo del modelo.

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo obtener las predicciones (quién va a abandonar)

El proyecto incluye `src/utils/predict_batch.py`, que carga el modelo final, predice sobre un
conjunto de estudiantes y exporta un CSV **ordenado de mayor a menor riesgo de abandono**.

Ejecutar **desde la raíz del proyecto**:

```bash
python src/utils/predict_batch.py
```

Esto genera `src/notebooks/src/artifacts/predicciones_dropout.csv`, con una fila por estudiante y
columnas:

| columna | significado |
|---|---|
| `prediccion` | clase predicha: `Dropout`, `Enrolled` o `Graduate` |
| `prob_Dropout` | probabilidad (%) de que abandone |
| `prob_Enrolled` | probabilidad (%) de que siga matriculado |
| `prob_Graduate` | probabilidad (%) de que se gradúe |
| `riesgo_dropout_pct` | igual que `prob_Dropout`, pensada para ordenar/filtrar en Excel |

Por defecto usa el conjunto de test del proyecto como ejemplo. Para predecir sobre estudiantes
nuevos reales, hay que sustituir la función `cargar_estudiantes_ejemplo()` del script por la carga
de tu propio CSV (debe tener las mismas 20 columnas usadas en entrenamiento).

### Uso programático (sin el script)

```python
import joblib

modelo = joblib.load('src/models/modelo_dropout_final.joblib')

predicciones = modelo.predict(X_nuevo)        # clase predicha
probabilidades = modelo.predict_proba(X_nuevo)  # probabilidad de cada clase
```

El objeto guardado es el **pipeline completo** (preprocesado + modelo), así que no hace falta
repetir el escalado ni el one-hot encoding manualmente.

### Trabajo realizado por Paula Comas y Ana Corrochano
