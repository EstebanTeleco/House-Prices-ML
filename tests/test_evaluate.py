"""
Tests para src/evaluate.py
"""

import numpy as np
import pandas as pd

from src.evaluate import evaluate_models, get_best_model


class ModeloFalso:
    """Simula un modelo entrenado: predict() devuelve valores fijos."""
    def __init__(self, predicciones):
        self.predicciones = np.array(predicciones)

    def predict(self, X):
        return self.predicciones


def test_evaluate_models_devuelve_metricas_para_cada_modelo():
    y_val = pd.Series([100000, 200000, 300000])
    modelos = {
        'ModeloA': ModeloFalso([100000, 200000, 300000]),  # predicción perfecta
        'ModeloB': ModeloFalso([90000, 210000, 280000]),
    }

    resultados = evaluate_models(modelos, X_val=None, y_val=y_val)

    assert set(resultados.keys()) == {'ModeloA', 'ModeloB'}
    for nombre in resultados:
        for key in ('r2', 'rmsle', 'err_prom', 'err_pct', 'pred'):
            assert key in resultados[nombre]

    # El modelo con predicción perfecta debe tener r2 == 1 y rmsle == 0
    assert resultados['ModeloA']['r2'] == 1.0
    assert resultados['ModeloA']['rmsle'] == 0.0
    assert resultados['ModeloA']['err_prom'] == 0.0


def test_get_best_model_elige_menor_rmsle():
    modelos = {
        'Bueno': ModeloFalso([1, 2, 3]),
        'Malo': ModeloFalso([1, 2, 3]),
    }
    resultados = {
        'Bueno': {'rmsle': 0.05},
        'Malo': {'rmsle': 0.20},
    }

    nombre, modelo = get_best_model(modelos, resultados)

    assert nombre == 'Bueno'
    assert modelo is modelos['Bueno']
