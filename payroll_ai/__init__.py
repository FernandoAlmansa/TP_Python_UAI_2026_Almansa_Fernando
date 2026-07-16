"""
payroll_ai
----------
Paquete con la lógica de negocio de la app (independiente de Streamlit):

- data.py    -> generación y validación de datos de nómina
- eda.py     -> análisis exploratorio con Pandas
- models.py  -> redes neuronales (autoencoder no supervisado + clasificador supervisado)
"""

from .data import PayrollDataGenerator, DataValidator, PayrollDataValidationError
from .eda import PayrollEDA
from .models import PayrollAnomalyDetector, PayrollRiskClassifier, Preprocessor

__all__ = [
    "PayrollDataGenerator",
    "DataValidator",
    "PayrollDataValidationError",
    "PayrollEDA",
    "PayrollAnomalyDetector",
    "PayrollRiskClassifier",
    "Preprocessor",
]
