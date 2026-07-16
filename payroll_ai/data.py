"""
data.py
--------
Generación y validación de datos de nómina (payroll) para el detector de anomalías.

Incluye:
- PayrollDataGenerator: crea datasets sintéticos de liquidaciones de sueldo,
  inspirados en conceptos de nómina argentina (legajo, categoría, horas extra,
  embargos, tipos de wage type al estilo SAP /1XX).
- DataValidator: valida el formato de un DataFrame de nómina usando expresiones
  regulares y reglas de negocio simples, antes de que entre al pipeline de EDA/ML.

Este módulo se usa tanto para generar un dataset de demo (si el usuario no
sube uno propio) como para validar cualquier CSV que el usuario cargue.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd


class PayrollDataValidationError(Exception):
    """Excepción propia para errores de validación de datos de nómina."""


# Columnas mínimas que cualquier dataset de nómina debe tener para poder
# aplicar el pipeline de EDA + modelos.
REQUIRED_COLUMNS = (
    "legajo",
    "categoria",
    "wage_type",
    "horas_extra",
    "antiguedad_anios",
    "sueldo_basico",
    "monto_embargo",
    "monto_bruto",
    "monto_neto",
)

# Un wage type "estilo SAP" (ej: /118, /325, /T30) se valida con una regex simple.
WAGE_TYPE_PATTERN = re.compile(r"^/[A-Z0-9]{3}$")
LEGAJO_PATTERN = re.compile(r"^\d{3,6}$")


@dataclass
class DataValidator:
    """Valida estructura y consistencia básica de un DataFrame de nómina.

    Se apoya en expresiones regulares para columnas con formato tipo SAP
    (wage_type, legajo) y en reglas de negocio (montos negativos, neto > bruto).
    """

    required_columns: tuple[str, ...] = field(default_factory=lambda: REQUIRED_COLUMNS)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Devuelve una lista de problemas encontrados (vacía si todo OK).

        No lanza excepción: la UI decide si son bloqueantes o solo advertencias.
        """
        problems: list[str] = []

        faltantes = [c for c in self.required_columns if c not in df.columns]
        if faltantes:
            problems.append(f"Faltan columnas requeridas: {', '.join(faltantes)}")
            # Si faltan columnas core no tiene sentido seguir validando el resto.
            return problems

        # Validación por regex: wage_type y legajo bien formados.
        wt_invalidos = df.loc[~df["wage_type"].astype(str).map(self._is_valid_wage_type)]
        if len(wt_invalidos) > 0:
            problems.append(
                f"{len(wt_invalidos)} filas con wage_type fuera de formato "
                f"(esperado tipo '/118', '/325', etc.)"
            )

        legajos_invalidos = df.loc[~df["legajo"].astype(str).map(self._is_valid_legajo)]
        if len(legajos_invalidos) > 0:
            problems.append(f"{len(legajos_invalidos)} filas con legajo inválido")

        # Reglas de negocio simples (lambda + boolean indexing de pandas).
        neto_mayor_bruto = df[df["monto_neto"] > df["monto_bruto"]]
        if len(neto_mayor_bruto) > 0:
            problems.append(
                f"{len(neto_mayor_bruto)} filas con monto_neto > monto_bruto (inconsistente)"
            )

        montos_negativos = df[
            (df["monto_bruto"] < 0) | (df["monto_neto"] < 0) | (df["sueldo_basico"] < 0)
        ]
        if len(montos_negativos) > 0:
            problems.append(f"{len(montos_negativos)} filas con montos negativos")

        return problems

    @staticmethod
    def _is_valid_wage_type(value: str) -> bool:
        return bool(WAGE_TYPE_PATTERN.match(value.strip()))

    @staticmethod
    def _is_valid_legajo(value: str) -> bool:
        return bool(LEGAJO_PATTERN.match(value.strip()))


class PayrollDataGenerator:
    """Genera un dataset sintético de liquidaciones de sueldo.

    La idea es simular, en miniatura, algunos de los problemas típicos que
    aparecen en la operación real de nómina (embargos mal topeados, horas
    extra fuera de rango, wage types de SAC que no se generan, etc.) para que
    el detector de anomalías tenga algo concreto que encontrar.
    """

    CATEGORIAS = ("Administrativo", "Tecnico", "Supervisor", "Gerente", "Jubilado")
    WAGE_TYPES = ("/118", "/325", "/355", "/365", "/T30", "/T31", "/T34", "/100")
    TIPOS_LIQUIDACION = ("mensual", "sac", "finiquito")

    def __init__(self, seed: int | None = 42) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate(self, n_rows: int = 500, anomaly_rate: float = 0.06) -> pd.DataFrame:
        """Genera `n_rows` liquidaciones, con una fracción `anomaly_rate` anómalas.

        Devuelve un DataFrame con una columna extra `es_anomalo` (0/1) que
        actúa como "ground truth" para poder evaluar el modelo supervisado y
        validar el no supervisado. En un caso real esta columna no existiría.
        """
        if not 0 < anomaly_rate < 0.5:
            raise ValueError("anomaly_rate debe estar entre 0 y 0.5")

        rows = [self._generate_normal_row(legajo_id) for legajo_id in range(1, n_rows + 1)]
        df = pd.DataFrame(rows)

        # Inyectamos anomalías sobre una muestra aleatoria de filas.
        n_anomalias = int(n_rows * anomaly_rate)
        idx_anomalos = self._np_rng.choice(df.index, size=n_anomalias, replace=False)
        for idx in idx_anomalos:
            df.loc[idx] = self._inject_anomaly(df.loc[idx].to_dict())

        df["es_anomalo"] = 0
        df.loc[idx_anomalos, "es_anomalo"] = 1

        return df.reset_index(drop=True)

    def iter_batches(self, df: pd.DataFrame, batch_size: int = 50) -> Iterator[pd.DataFrame]:
        """Generador: recorre el DataFrame en lotes (útil para datasets grandes).

        Demuestra el uso de un generator en vez de cargar/iterar todo en memoria
        de una sola vez, tal como se pediría al procesar archivos de nómina
        reales que pueden tener miles de líneas.
        """
        for start in range(0, len(df), batch_size):
            yield df.iloc[start : start + batch_size]

    # -- helpers internos -------------------------------------------------

    def _generate_normal_row(self, legajo_id: int) -> dict:
        categoria = self._rng.choice(self.CATEGORIAS)
        antiguedad = round(self._rng.uniform(0, 25), 1)
        sueldo_basico = self._basico_por_categoria(categoria) * (1 + antiguedad * 0.01)
        horas_extra = max(0, self._rng.gauss(6, 4))
        wage_type = self._rng.choice(self.WAGE_TYPES)
        tipo_liq = self._rng.choice(self.TIPOS_LIQUIDACION)

        monto_bruto = sueldo_basico + horas_extra * (sueldo_basico / 180)
        # Embargo topeado legalmente (simplificación) a ~20% del neto.
        tiene_embargo = self._rng.random() < 0.15
        monto_neto_previo = monto_bruto * 0.83  # descuentos ley simplificados
        monto_embargo = round(monto_neto_previo * 0.20, 2) if tiene_embargo else 0.0
        monto_neto = round(monto_neto_previo - monto_embargo, 2)

        fecha = datetime(2026, 1, 1) + timedelta(days=self._rng.randint(0, 180))

        return {
            "legajo": str(1000 + legajo_id),
            "categoria": categoria,
            "wage_type": wage_type,
            "tipo_liquidacion": tipo_liq,
            "fecha_liquidacion": fecha.strftime("%Y-%m-%d"),
            "antiguedad_anios": antiguedad,
            "horas_extra": round(horas_extra, 1),
            "sueldo_basico": round(sueldo_basico, 2),
            "monto_embargo": monto_embargo,
            "monto_bruto": round(monto_bruto, 2),
            "monto_neto": monto_neto,
        }

    def _basico_por_categoria(self, categoria: str) -> float:
        base = {
            "Administrativo": 900_000,
            "Tecnico": 1_100_000,
            "Supervisor": 1_450_000,
            "Gerente": 2_300_000,
            "Jubilado": 650_000,
        }
        return base[categoria]

    def _inject_anomaly(self, row: dict) -> dict:
        """Aplica una de varias distorsiones típicas de un error de liquidación."""
        tipo_anomalia = self._rng.choice(
            ["embargo_excedido", "horas_extra_extremas", "neto_mayor_bruto", "wage_type_roto"]
        )

        if tipo_anomalia == "embargo_excedido":
            # Embargo que excede ampliamente el tope legal del 20%.
            row["monto_embargo"] = round(row["monto_neto"] * self._rng.uniform(0.6, 1.5), 2)
            row["monto_neto"] = round(row["monto_neto"] - row["monto_embargo"], 2)

        elif tipo_anomalia == "horas_extra_extremas":
            row["horas_extra"] = round(self._rng.uniform(80, 200), 1)
            row["monto_bruto"] = round(
                row["monto_bruto"] + row["horas_extra"] * (row["sueldo_basico"] / 180), 2
            )

        elif tipo_anomalia == "neto_mayor_bruto":
            # Error clásico de cadena de wage types rota (ej. /325 no se generó).
            row["monto_neto"] = round(row["monto_bruto"] * self._rng.uniform(1.05, 1.3), 2)

        elif tipo_anomalia == "wage_type_roto":
            row["wage_type"] = "/ERR"
            row["monto_bruto"] = 0.0
            row["monto_neto"] = 0.0

        return row
