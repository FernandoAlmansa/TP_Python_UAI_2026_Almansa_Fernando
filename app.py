"""
app.py
------
PayrollLens - Detector de anomalías en liquidaciones de nómina.

TP Python para Inteligencia Artificial (UAI 2026).
Clases integradas: (1) Análisis de datos con Pandas + (2) Redes neuronales
(un autoencoder no supervisado y un MLPClassifier supervisado, con ajuste de
hiperparámetros).

Cómo correrlo local:
    streamlit run app.py
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from payroll_ai import (
    DataValidator,
    PayrollAnomalyDetector,
    PayrollDataGenerator,
    PayrollEDA,
    PayrollRiskClassifier,
)

st.set_page_config(page_title="PayrollLens", page_icon="🧾", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos (sidebar)
# ---------------------------------------------------------------------------

def cargar_dataset() -> pd.DataFrame | None:
    st.sidebar.header("1. Datos")
    fuente = st.sidebar.radio(
        "Origen de los datos",
        ["Generar dataset de ejemplo", "Subir mi propio CSV"],
    )

    if fuente == "Generar dataset de ejemplo":
        n_rows = st.sidebar.slider("Cantidad de liquidaciones", 100, 3000, 500, step=100)
        anomaly_rate = st.sidebar.slider("Tasa de anomalías inyectadas", 0.02, 0.20, 0.06, step=0.01)
        seed = st.sidebar.number_input("Semilla aleatoria", value=42, step=1)
        if st.sidebar.button("🎲 Generar datos", width='stretch'):
            generador = PayrollDataGenerator(seed=int(seed))
            st.session_state["df"] = generador.generate(n_rows=n_rows, anomaly_rate=anomaly_rate)
            st.session_state["tiene_ground_truth"] = True
            for k in ("autoencoder", "clasificador"):
                st.session_state.pop(k, None)

    else:
        archivo = st.sidebar.file_uploader("CSV de liquidaciones", type=["csv"])
        st.sidebar.caption(
            "Columnas esperadas: legajo, categoria, wage_type, tipo_liquidacion, "
            "fecha_liquidacion, antiguedad_anios, horas_extra, sueldo_basico, "
            "monto_embargo, monto_bruto, monto_neto"
        )
        if archivo is not None:
            try:
                df_subido = pd.read_csv(archivo)
            except Exception as exc:  # noqa: BLE001 - queremos mostrarle cualquier error al usuario
                st.sidebar.error(f"No se pudo leer el CSV: {exc}")
                return st.session_state.get("df")

            problemas = DataValidator().validate(df_subido)
            if problemas:
                st.sidebar.warning("Se encontraron observaciones de calidad de datos:")
                for p in problemas:
                    st.sidebar.write(f"- {p}")

            st.session_state["df"] = df_subido
            st.session_state["tiene_ground_truth"] = "es_anomalo" in df_subido.columns
            for k in ("autoencoder", "clasificador"):
                st.session_state.pop(k, None)

    return st.session_state.get("df")


def descargar_dataset(df: pd.DataFrame) -> None:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    st.sidebar.download_button(
        "⬇️ Descargar dataset actual (CSV)",
        buffer.getvalue(),
        file_name="nomina_dataset.csv",
        mime="text/csv",
        width='stretch',
    )


# ---------------------------------------------------------------------------
# Tab 1: Análisis exploratorio (Pandas)
# ---------------------------------------------------------------------------

def tab_eda(df: pd.DataFrame) -> None:
    eda = PayrollEDA(df)
    resumen = eda.resumen_general()

    col1, col2, col3 = st.columns(3)
    col1.metric("Liquidaciones", resumen["n_filas"])
    col2.metric("Columnas", resumen["n_columnas"])
    col3.metric("Legajos únicos", df["legajo"].nunique())

    st.subheader("Tipos de dato y nulos")
    tipos_df = pd.DataFrame(
        {"tipo": resumen["tipos"]}
    )
    st.dataframe(tipos_df, width='stretch')
    if resumen["nulos_por_columna"]:
        st.warning(f"Columnas con nulos: {list(resumen['nulos_por_columna'].keys())}")
    else:
        st.success("No hay valores nulos en el dataset.")

    st.subheader("Estadística descriptiva")
    st.dataframe(eda.describe_numericas(), width='stretch')

    st.subheader("Outliers (método IQR)")
    alertas = eda.alertas_rapidas()
    if alertas:
        for a in alertas:
            st.write(f"⚠️ {a}")
    else:
        st.success("No se detectaron outliers relevantes.")

    colg1, colg2 = st.columns(2)
    with colg1:
        st.caption("Distribución de monto bruto")
        fig, ax = plt.subplots()
        ax.hist(df["monto_bruto"], bins=30, color="#4C72B0")
        ax.set_xlabel("Monto bruto")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    with colg2:
        st.caption("Monto bruto por categoría (boxplot)")
        fig, ax = plt.subplots()
        categorias = df["categoria"].unique()
        datos = [df.loc[df["categoria"] == c, "monto_bruto"] for c in categorias]
        ax.boxplot(datos, tick_labels=categorias)
        ax.set_ylabel("Monto bruto")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    st.subheader("Promedios por categoría")
    st.dataframe(eda.agregado_por_categoria(), width='stretch')

    st.subheader("Matriz de correlación")
    corr = eda.matriz_correlacion()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)

    st.subheader("Filtros interactivos")
    fc1, fc2, fc3, fc4 = st.columns(4)
    categoria = fc1.selectbox("Categoría", ["Todas"] + sorted(df["categoria"].unique().tolist()))
    wage_type = fc2.selectbox("Wage type", ["Todos"] + sorted(df["wage_type"].unique().tolist()))
    monto_min = fc3.number_input("Monto bruto mínimo", value=float(df["monto_bruto"].min()))
    monto_max = fc4.number_input("Monto bruto máximo", value=float(df["monto_bruto"].max()))

    filtrado = eda.filtrar(categoria=categoria, wage_type=wage_type, monto_min=monto_min, monto_max=monto_max)
    st.write(f"{len(filtrado)} registros coinciden con el filtro")
    st.dataframe(filtrado, width='stretch', height=250)


# ---------------------------------------------------------------------------
# Tab 2: Detección NO supervisada (autoencoder)
# ---------------------------------------------------------------------------

def tab_no_supervisado(df: pd.DataFrame) -> None:
    st.write(
        "Se entrena un **autoencoder** (`MLPRegressor` de entrada→entrada, con un cuello de "
        "botella interno) sobre todos los registros, **sin usar ninguna etiqueta**. Los "
        "registros que la red no logra reconstruir bien (alto error de reconstrucción) se "
        "marcan como posibles anomalías."
    )

    percentil = st.slider(
        "Percentil de error usado como umbral de anomalía",
        50, 99, 90,
        help="Cuanto más alto, más estricto: solo el % superior de errores se marca como anomalía",
    )

    if st.button("🧠 Entrenar autoencoder", type="primary"):
        columnas_modelo = df.drop(columns=["es_anomalo"], errors="ignore")
        detector = PayrollAnomalyDetector()
        detector.fit(columnas_modelo, percentile_umbral=percentil)
        st.session_state["autoencoder"] = detector
        st.session_state["autoencoder_scores"] = detector.scores(columnas_modelo)

    if "autoencoder_scores" in st.session_state:
        scored = st.session_state["autoencoder_scores"]
        detector: PayrollAnomalyDetector = st.session_state["autoencoder"]

        n_anom = int(scored["es_anomalo_pred"].sum())
        col1, col2 = st.columns(2)
        col1.metric("Anomalías detectadas", f"{n_anom} / {len(scored)}")
        col2.metric("Umbral de error (MSE)", f"{detector.threshold:.4f}")

        fig, ax = plt.subplots()
        ax.hist(scored["error_reconstruccion"], bins=40, color="#55A868")
        ax.axvline(detector.threshold, color="red", linestyle="--", label="Umbral")
        ax.set_xlabel("Error de reconstrucción")
        ax.set_ylabel("Cantidad de registros")
        ax.legend()
        st.pyplot(fig)

        if st.session_state.get("tiene_ground_truth") and "es_anomalo" in df.columns:
            st.caption(
                "Como este dataset es sintético, sabemos cuáles filas son realmente "
                "anómalas: así se ve el modelo no supervisado comparado contra esa verdad."
            )
            comparacion = pd.crosstab(
                df["es_anomalo"].map({0: "Normal (real)", 1: "Anómalo (real)"}),
                scored["es_anomalo_pred"].map({0: "Normal (predicho)", 1: "Anómalo (predicho)"}),
            )
            st.dataframe(comparacion, width='stretch')

        st.subheader("Registros marcados como anómalos")
        st.dataframe(
            scored[scored["es_anomalo_pred"] == 1].sort_values("error_reconstruccion", ascending=False),
            width='stretch',
            height=300,
        )
    else:
        st.info("Configurá el umbral y presioná 'Entrenar autoencoder' para ver resultados.")


# ---------------------------------------------------------------------------
# Tab 3: Clasificador supervisado
# ---------------------------------------------------------------------------

def tab_supervisado(df: pd.DataFrame) -> None:
    if "es_anomalo" not in df.columns:
        st.warning(
            "Este dataset no tiene una columna de etiqueta `es_anomalo`, así que no se puede "
            "entrenar el clasificador supervisado. Generá el dataset de ejemplo, o subí un CSV "
            "propio que incluya esa columna (1 = anómalo, 0 = normal)."
        )
        return

    st.write(
        "Un `MLPClassifier` (red neuronal supervisada) entrenado con la columna `es_anomalo` "
        "como etiqueta. Podés ajustar hiperparámetros y ver cómo cambia el desempeño."
    )

    c1, c2, c3 = st.columns(3)
    capa1 = c1.slider("Neuronas capa oculta 1", 4, 64, 16)
    capa2 = c2.slider("Neuronas capa oculta 2", 0, 32, 8, help="0 = una sola capa oculta")
    learning_rate = c3.select_slider(
        "Learning rate inicial", options=[0.0001, 0.001, 0.005, 0.01, 0.05], value=0.001
    )
    max_iter = st.slider("Máximo de iteraciones de entrenamiento", 100, 1500, 500, step=100)

    capas = (capa1,) if capa2 == 0 else (capa1, capa2)

    if st.button("🎯 Entrenar clasificador", type="primary"):
        clasificador = PayrollRiskClassifier(
            hidden_layer_sizes=capas, learning_rate_init=learning_rate, max_iter=max_iter
        )
        metricas = clasificador.fit(df, target_col="es_anomalo")
        st.session_state["clasificador"] = clasificador
        st.session_state["metricas_clasificador"] = metricas

    if "metricas_clasificador" in st.session_state:
        m = st.session_state["metricas_clasificador"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", m["accuracy"])
        col2.metric("Precision", m["precision"])
        col3.metric("Recall", m["recall"])
        col4.metric("F1-score", m["f1"])

        st.caption(f"Evaluado sobre {m['n_test']} registros de test (holdout)")

        st.subheader("Matriz de confusión")
        matriz = m["matriz_confusion"]
        fig, ax = plt.subplots()
        ax.imshow(matriz, cmap="Blues")
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                ax.text(j, i, str(matriz[i][j]), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Anómalo"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal", "Anómalo"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        st.pyplot(fig)
    else:
        st.info("Ajustá los hiperparámetros y presioná 'Entrenar clasificador'.")


# ---------------------------------------------------------------------------
# Tab 4: evaluar un registro manual
# ---------------------------------------------------------------------------

def tab_prediccion_manual(df: pd.DataFrame) -> None:
    st.write("Cargá los datos de una liquidación puntual y consultá ambos modelos entrenados.")

    if "autoencoder" not in st.session_state and "clasificador" not in st.session_state:
        st.info("Entrená al menos un modelo en las otras pestañas antes de evaluar un registro.")
        return

    with st.form("form_registro_manual"):
        c1, c2, c3 = st.columns(3)
        categoria = c1.selectbox("Categoría", sorted(df["categoria"].unique()))
        wage_type = c2.selectbox("Wage type", sorted(df["wage_type"].unique()))
        tipo_liq = c3.selectbox("Tipo de liquidación", sorted(df["tipo_liquidacion"].unique()))

        c4, c5, c6 = st.columns(3)
        antiguedad = c4.number_input("Antigüedad (años)", 0.0, 45.0, 5.0)
        horas_extra = c5.number_input("Horas extra", 0.0, 300.0, 6.0)
        sueldo_basico = c6.number_input("Sueldo básico", 0.0, 10_000_000.0, 1_000_000.0)

        c7, c8, c9 = st.columns(3)
        monto_embargo = c7.number_input("Monto embargo", 0.0, 10_000_000.0, 0.0)
        monto_bruto = c8.number_input("Monto bruto", 0.0, 20_000_000.0, 1_150_000.0)
        monto_neto = c9.number_input("Monto neto", 0.0, 20_000_000.0, 950_000.0)

        enviado = st.form_submit_button("🔍 Evaluar registro")

    if enviado:
        registro = pd.DataFrame(
            [
                {
                    "categoria": categoria,
                    "wage_type": wage_type,
                    "tipo_liquidacion": tipo_liq,
                    "antiguedad_anios": antiguedad,
                    "horas_extra": horas_extra,
                    "sueldo_basico": sueldo_basico,
                    "monto_embargo": monto_embargo,
                    "monto_bruto": monto_bruto,
                    "monto_neto": monto_neto,
                }
            ]
        )

        col1, col2 = st.columns(2)
        if "autoencoder" in st.session_state:
            detector: PayrollAnomalyDetector = st.session_state["autoencoder"]
            scored = detector.scores(registro)
            error = scored["error_reconstruccion"].iloc[0]
            es_anomalo = bool(scored["es_anomalo_pred"].iloc[0])
            with col1:
                st.metric("Error de reconstrucción (no supervisado)", f"{error:.4f}")
                st.write("🔴 Anómalo" if es_anomalo else "🟢 Normal")

        if "clasificador" in st.session_state:
            clasificador: PayrollRiskClassifier = st.session_state["clasificador"]
            clase, proba = clasificador.predict_one(registro)
            with col2:
                st.metric("Probabilidad de anomalía (supervisado)", f"{proba:.1%}")
                st.write("🔴 Anómalo" if clase == 1 else "🟢 Normal")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("PayrollLens")
    st.caption(
        "Detector de anomalías en liquidaciones de nómina — integra Análisis de datos con "
        "Pandas + Redes Neuronales (autoencoder no supervisado y clasificador supervisado). "
        "TP Python para IA, UAI 2026."
    )

    df = cargar_dataset()

    if df is None or df.empty:
        st.info("👈 Generá un dataset de ejemplo o subí tu propio CSV desde la barra lateral para empezar.")
        return

    descargar_dataset(df)

    tabs = st.tabs(
        [
            "📊 Análisis exploratorio",
            "🕵️ Detección no supervisada",
            "🎯 Clasificador supervisado",
            "🔍 Evaluar un registro",
        ]
    )
    with tabs[0]:
        tab_eda(df)
    with tabs[1]:
        tab_no_supervisado(df)
    with tabs[2]:
        tab_supervisado(df)
    with tabs[3]:
        tab_prediccion_manual(df)


if __name__ == "__main__":
    main()
