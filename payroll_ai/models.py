"""
models.py
---------
Redes neuronales aplicadas a la detección de anomalías en liquidaciones de nómina.

Dos enfoques, ambos vistos en la clase de Redes Neuronales:

1) PayrollAnomalyDetector (NO supervisado):
   Un autoencoder (red neuronal entrenada para reconstruir su propia entrada)
   implementado con sklearn.neural_network.MLPRegressor. Se entrena SOLO con
   registros "normales" y aprende a reconstruirlos con bajo error. Frente a un
   registro anómalo, el error de reconstrucción es alto -> se marca como
   anomalía. No necesita ninguna etiqueta.

2) PayrollRiskClassifier (supervisado):
   Un MLPClassifier clásico, entrenado con la columna `es_anomalo` (cuando
   existe, real o generada) para comparar contra el autoencoder y para mostrar
   ajuste de hiperparámetros (capas ocultas, learning rate, iteraciones).

Ambas clases comparten un mismo Preprocessor para no duplicar lógica de
encoding/escalado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

NUMERIC_FEATURES = (
    "antiguedad_anios",
    "horas_extra",
    "sueldo_basico",
    "monto_embargo",
    "monto_bruto",
    "monto_neto",
)
CATEGORICAL_FEATURES = ("categoria", "wage_type", "tipo_liquidacion")


class ModelNotFittedError(Exception):
    """Se lanza si se intenta usar (predecir/transformar) un modelo aún no entrenado."""


@dataclass
class Preprocessor:
    """Codifica variables categóricas y escala variables numéricas.

    Se guarda un LabelEncoder por columna categórica (diccionario), de forma
    que el mismo preprocesador pueda usarse tanto para entrenar como para
    transformar un único registro nuevo ingresado a mano en la UI.
    """

    numeric_features: tuple[str, ...] = NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES

    def __post_init__(self) -> None:
        self._encoders: dict[str, LabelEncoder] = {}
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in self.categorical_features:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            self._encoders[col] = encoder

        matriz_numerica = df[list(self.numeric_features)].to_numpy(dtype=float)
        matriz_categorica = df[list(self.categorical_features)].to_numpy(dtype=float)
        matriz_numerica = self._scaler.fit_transform(matriz_numerica)

        self._fitted = True
        return np.hstack([matriz_numerica, matriz_categorica])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ModelNotFittedError("Preprocessor no fue entrenado (fit_transform) todavía")

        df = df.copy()
        for col in self.categorical_features:
            encoder = self._encoders[col]
            # Si aparece una categoría nunca vista, la mapeamos a la más frecuente
            # conocida en vez de romper (manejo de excepciones defensivo).
            valores_conocidos = set(encoder.classes_)
            df[col] = df[col].astype(str).map(
                lambda v: v if v in valores_conocidos else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

        matriz_numerica = self._scaler.transform(df[list(self.numeric_features)].to_numpy(dtype=float))
        matriz_categorica = df[list(self.categorical_features)].to_numpy(dtype=float)
        return np.hstack([matriz_numerica, matriz_categorica])


class PayrollAnomalyDetector:
    """Autoencoder no supervisado (MLPRegressor input->input) para detectar anomalías.

    El "cuello de botella" (una capa oculta más chica que la entrada) obliga a
    la red a aprender una representación comprimida de lo que es un registro
    "normal". Los registros que no encajan en ese patrón generan mayor error
    de reconstrucción.
    """

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (12, 4, 12), random_state: int = 42):
        self.preprocessor = Preprocessor()
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=800,
            random_state=random_state,
            early_stopping=True,
        )
        self._threshold: float | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, percentile_umbral: float = 90.0) -> "PayrollAnomalyDetector":
        """Entrena el autoencoder y fija el umbral de anomalía en un percentil
        del error de reconstrucción sobre los propios datos de entrenamiento."""
        try:
            X = self.preprocessor.fit_transform(df)
            self._model.fit(X, X)
            errores = self._reconstruction_error(X)
            self._threshold = float(np.percentile(errores, percentile_umbral))
            self._fitted = True
        except ValueError as exc:
            raise ModelNotFittedError(f"No se pudo entrenar el autoencoder: {exc}") from exc
        return self

    def scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve el DataFrame original + columnas `error_reconstruccion` y `es_anomalo_pred`."""
        if not self._fitted:
            raise ModelNotFittedError("Autoencoder no entrenado: llamar a .fit() primero")

        X = self.preprocessor.transform(df)
        errores = self._reconstruction_error(X)

        resultado = df.copy()
        resultado["error_reconstruccion"] = errores
        resultado["es_anomalo_pred"] = (errores > self._threshold).astype(int)
        return resultado

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        reconstruccion = self._model.predict(X)
        # Error cuadrático medio por fila (a lo largo de todas las features).
        return np.mean((X - reconstruccion) ** 2, axis=1)

    @property
    def threshold(self) -> float | None:
        return self._threshold


class PayrollRiskClassifier:
    """Clasificador supervisado (MLPClassifier) entrenado con la etiqueta `es_anomalo`."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (16, 8),
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.preprocessor = Preprocessor()
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "es_anomalo", test_size: float = 0.25) -> dict:
        """Entrena con train/test split y devuelve un diccionario de métricas."""
        if target_col not in df.columns:
            raise ModelNotFittedError(f"El dataset no tiene la columna objetivo '{target_col}'")

        y = df[target_col].to_numpy()
        X = self.preprocessor.fit_transform(df.drop(columns=[target_col]))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.sum() > 1 else None
        )
        self._model.fit(X_train, y_train)
        self._fitted = True

        y_pred = self._model.predict(X_test)
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
            "n_test": len(y_test),
        }

    def predict_one(self, registro: pd.DataFrame) -> tuple[int, float]:
        """Predice para un único registro (DataFrame de una fila) y devuelve (clase, probabilidad)."""
        if not self._fitted:
            raise ModelNotFittedError("Clasificador no entrenado: llamar a .fit() primero")

        X = self.preprocessor.transform(registro)
        clase = int(self._model.predict(X)[0])
        proba = float(self._model.predict_proba(X)[0][1])
        return clase, proba
