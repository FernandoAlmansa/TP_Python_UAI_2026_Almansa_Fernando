"""
eda.py
------
Análisis exploratorio de datos (EDA) con Pandas sobre el dataset de nómina.

Toda la clase PayrollEDA trabaja *solo* con pandas/numpy (sin librerías de
graficado): la app Streamlit se encarga de tomar estos resultados (DataFrames,
diccionarios, Series) y graficarlos. Esto separa claramente "análisis de datos"
de "presentación", como se pide en la consigna del TP.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = (
    "antiguedad_anios",
    "horas_extra",
    "sueldo_basico",
    "monto_embargo",
    "monto_bruto",
    "monto_neto",
)


class PayrollEDA:
    """Encapsula el análisis exploratorio de un DataFrame de nómina."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # -- resúmenes generales -------------------------------------------------

    def resumen_general(self) -> dict:
        """Cantidad de filas/columnas, tipos de dato y % de nulos por columna."""
        nulos = self.df.isna().sum()
        pct_nulos = (nulos / len(self.df) * 100).round(2)

        return {
            "n_filas": len(self.df),
            "n_columnas": len(self.df.columns),
            "tipos": self.df.dtypes.astype(str).to_dict(),
            "nulos_por_columna": {
                col: {"cantidad": int(nulos[col]), "porcentaje": float(pct_nulos[col])}
                for col in self.df.columns
                if nulos[col] > 0
            },
        }

    def describe_numericas(self) -> pd.DataFrame:
        """Estadística descriptiva estándar de pandas para columnas numéricas."""
        cols = [c for c in NUMERIC_COLUMNS if c in self.df.columns]
        return self.df[cols].describe().T

    # -- outliers --------------------------------------------------------

    def outliers_iqr(self, columna: str) -> pd.DataFrame:
        """Detecta outliers univariados con el método clásico de rango intercuartílico.

        Devuelve el subconjunto de filas cuyo valor en `columna` cae fuera de
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        """
        if columna not in self.df.columns:
            raise KeyError(f"Columna '{columna}' no existe en el dataset")

        q1 = self.df[columna].quantile(0.25)
        q3 = self.df[columna].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        es_outlier: Callable[[float], bool] = lambda v: v < limite_inf or v > limite_sup
        mascara = self.df[columna].map(es_outlier)
        return self.df.loc[mascara]

    def reporte_outliers(self) -> dict[str, int]:
        """Cuenta outliers IQR para cada columna numérica del dataset."""
        return {
            col: len(self.outliers_iqr(col))
            for col in NUMERIC_COLUMNS
            if col in self.df.columns
        }

    # -- agregaciones -----------------------------------------------------

    def agregado_por_categoria(self) -> pd.DataFrame:
        """Promedios de las variables clave, agrupados por categoría de puesto."""
        cols = [c for c in NUMERIC_COLUMNS if c in self.df.columns]
        return self.df.groupby("categoria")[cols].mean().round(2)

    def matriz_correlacion(self) -> pd.DataFrame:
        cols = [c for c in NUMERIC_COLUMNS if c in self.df.columns]
        return self.df[cols].corr().round(2)

    # -- filtros interactivos ----------------------------------------------

    def filtrar(
        self,
        categoria: str | None = None,
        wage_type: str | None = None,
        monto_min: float | None = None,
        monto_max: float | None = None,
    ) -> pd.DataFrame:
        """Filtra el dataset por combinación de criterios (indexado booleano de pandas)."""
        resultado = self.df

        if categoria and categoria != "Todas":
            resultado = resultado[resultado["categoria"] == categoria]
        if wage_type and wage_type != "Todos":
            resultado = resultado[resultado["wage_type"] == wage_type]
        if monto_min is not None:
            resultado = resultado[resultado["monto_bruto"] >= monto_min]
        if monto_max is not None:
            resultado = resultado[resultado["monto_bruto"] <= monto_max]

        return resultado

    # -- comprehension de ejemplo, usada en la UI para armar un resumen de texto --

    def alertas_rapidas(self) -> list[str]:
        """Genera alertas legibles en base a reglas simples (list comprehension)."""
        reporte = self.reporte_outliers()
        return [
            f"'{col}' tiene {cant} valores atípicos (outliers IQR)"
            for col, cant in reporte.items()
            if cant > 0
        ]
