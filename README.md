# PayrollLens — Detector de anomalías en liquidaciones de nómina

Trabajo Práctico — **Python para Inteligencia Artificial** (UAI 2026)

Clases integradas:

1. **Análisis de datos con Pandas** (carga, limpieza, outliers, agregaciones, filtros, visualización)
2. **Redes neuronales** (modelo no supervisado y modelo supervisado, con ajuste de hiperparámetros)

## Qué hace

PayrollLens simula un problema real de un consultor de nómina (SAP HCM / payroll):
detectar liquidaciones de sueldo con errores o inconsistencias — embargos que
exceden el tope legal, horas extra fuera de rango, cadenas de wage types
rotas (neto mayor que bruto), etc. — antes de que lleguen al empleado.

El flujo end-to-end es:

1. **Cargar datos**: el usuario genera un dataset sintético de liquidaciones
   (con anomalías inyectadas a propósito) o sube su propio CSV.
2. **Análisis exploratorio con Pandas**: tipos de dato, nulos, estadística
   descriptiva, detección de outliers (método IQR), agregaciones por
   categoría, matriz de correlación y filtros interactivos.
3. **Detección no supervisada**: se entrena un **autoencoder**
   (`MLPRegressor` entrada→entrada con cuello de botella) *sin usar ninguna
   etiqueta*. Los registros con mayor error de reconstrucción se marcan como
   posibles anomalías.
4. **Clasificación supervisada**: se entrena un **`MLPClassifier`** usando la
   etiqueta de anomalía, con hiperparámetros ajustables desde la UI (capas
   ocultas, learning rate, iteraciones) y métricas de evaluación
   (accuracy, precision, recall, F1, matriz de confusión).
5. **Evaluar un registro puntual**: el usuario carga a mano los datos de una
   liquidación y consulta el score de ambos modelos entrenados.

La integración entre las dos clases es real: el mismo `Preprocessor` (Pandas
+ scikit-learn) alimenta a ambas redes, y el análisis exploratorio del paso 2
es el que informa qué variables conviene mirar en los pasos 3 y 4 (por
ejemplo, los outliers de `monto_embargo` detectados con Pandas son,
justamente, los que el autoencoder termina marcando como anómalos).

## Stack

- [Streamlit](https://streamlit.io/) — interfaz web
- [Pandas](https://pandas.pydata.org/) / NumPy — análisis de datos
- [scikit-learn](https://scikit-learn.org/) — `MLPRegressor` (autoencoder) y `MLPClassifier`
- [Matplotlib](https://matplotlib.org/) — visualizaciones

## Estructura del repo

```
payroll-anomaly-detector/
├── app.py                    # UI de Streamlit (orquesta todo el flujo)
├── payroll_ai/
│   ├── __init__.py
│   ├── data.py                # PayrollDataGenerator + DataValidator (regex, reglas de negocio)
│   ├── eda.py                  # PayrollEDA (Pandas: nulos, outliers, agregaciones, filtros)
│   └── models.py               # Preprocessor, PayrollAnomalyDetector (no sup.), PayrollRiskClassifier (sup.)
├── sample_data/
│   └── payroll_sample.csv      # Dataset de ejemplo ya generado, por si querés probar sin generar uno nuevo
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

## Correrlo localmente

Requisitos: Python 3.10+

```bash
git clone <URL_DE_ESTE_REPO>
cd payroll-anomaly-detector

python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`. Desde la barra lateral,
tocá **"🎲 Generar datos"** para crear un dataset de ejemplo y arrancar a
explorar las pestañas.

## Deploy (Streamlit Community Cloud — gratis)

1. Subí este repo a GitHub (debe ser público).
2. Entrá a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. **New app** → elegí el repo, la rama (`main`) y el archivo principal (`app.py`).
4. Deploy. En un par de minutos queda la URL pública
   (`https://<nombre-de-tu-app>.streamlit.app`).

> Alternativas equivalentes: Render, Railway o Hugging Face Spaces (con SDK
> "Streamlit"). El `requirements.txt` ya está listo para cualquiera de las tres.

**URL del deploy:** _completar acá una vez desplegado_

## Detalle técnico de las redes neuronales

### No supervisado — Autoencoder (`PayrollAnomalyDetector`)

- `MLPRegressor(hidden_layer_sizes=(12, 4, 12))`: la capa del medio (4
  neuronas) es más chica que la entrada, así que fuerza a la red a comprimir
  cada liquidación "normal" a una representación de baja dimensión y después
  reconstruirla.
- Se entrena con `X` como entrada **y como salida** (no hay etiqueta).
- El umbral de anomalía es un percentil (configurable desde la UI) del error
  de reconstrucción (MSE) sobre el propio set de entrenamiento.
- Ventaja: funciona incluso si nunca viste un caso de anomalía etiquetado —
  el escenario más común en la vida real de un ticket de nómina.

### Supervisado — Clasificador (`PayrollRiskClassifier`)

- `MLPClassifier` entrenado con la columna `es_anomalo` (train/test split
  80/25, estratificado).
- Hiperparámetros ajustables desde la UI: neuronas por capa oculta (una o
  dos capas), `learning_rate_init`, `max_iter`.
- Métricas reportadas: accuracy, precision, recall, F1 y matriz de confusión
  sobre el conjunto de test.

### Recursos de Python aplicados

- **Clases y dataclasses**: `PayrollDataGenerator`, `DataValidator`,
  `PayrollEDA`, `Preprocessor`, `PayrollAnomalyDetector`, `PayrollRiskClassifier`.
- **Expresiones regulares**: validación de `wage_type` (formato `/118`,
  `/325`, etc.) y `legajo` en `DataValidator`.
- **Funciones lambda**: reglas de outliers en `PayrollEDA.outliers_iqr` y
  mapeo de categorías desconocidas en `Preprocessor.transform`.
- **Manejo de excepciones**: `PayrollDataValidationError`,
  `ModelNotFittedError`, validaciones defensivas al leer un CSV subido por el usuario.
- **Generators**: `PayrollDataGenerator.iter_batches` recorre el dataset en
  lotes sin cargar todo en memoria de una.
- **Comprehensions**: listas y diccionarios en `PayrollEDA` (`alertas_rapidas`,
  `reporte_outliers`, `resumen_general`).
- **Estructuras de datos**: tuplas para columnas fijas (`NUMERIC_FEATURES`),
  diccionarios de encoders por columna, sets para validar categorías conocidas.

## Nota sobre los datos

Todo el dataset es **sintético**, generado por `PayrollDataGenerator`. No
contiene información real de ningún cliente ni empleado.
