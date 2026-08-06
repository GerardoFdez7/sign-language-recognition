"""Catálogo de modelos y transformaciones seleccionados para la fase experimental."""

from __future__ import annotations


MODEL_PLAN = (
    {
        "name": "CNN base",
        "family": "convolutional",
        "input": "64x64x3",
        "purpose": "Línea base espacial de costo moderado",
        "search": "filtros 32-64-128; dropout 0.25/0.40; lr 1e-3/3e-4",
    },
    {
        "name": "CNN con normalización",
        "family": "convolutional",
        "input": "64x64x3",
        "purpose": "Más profundidad, BatchNorm y regularización",
        "search": "3/4 bloques; dropout 0.30/0.50; L2 0/1e-4",
    },
    {
        "name": "MLP",
        "family": "fully_connected",
        "input": "64x64x3 aplanado",
        "purpose": "Control sin sesgo inductivo espacial",
        "search": "256/512 unidades; 1/2 capas; dropout 0.30/0.50",
    },
    {
        "name": "SVM con HOG",
        "family": "classical",
        "input": "descriptor HOG",
        "purpose": "Comparación basada en contornos y margen máximo",
        "search": "kernel lineal/RBF; C 1/10; gamma scale/auto",
    },
)


def build_safe_augmentation():
    """Describe transformaciones moderadas para experimentos posteriores."""
    return (
        {"name": "rotation", "range": "-14 a 14 grados"},
        {"name": "translation", "range": "hasta 6 % por eje"},
        {"name": "zoom", "range": "-8 % a 8 %"},
        {"name": "contrast", "range": "-12 % a 12 %"},
    )


def model_plan_table():
    """Devuelve el catálogo como DataFrame para mostrarlo en el informe."""
    import pandas as pd

    return pd.DataFrame(MODEL_PLAN)
