"""
Evaluación y comparación de los modelos entrenados.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_models(modelos, X_val, y_val):
    """
    Calcula las métricas de cada modelo sobre el set de validación:
    R², RMSLE (la métrica que usa la competencia de Kaggle), error
    promedio en dólares y error porcentual.
    """
    resultados = {}

    for nombre, modelo in modelos.items():
        pred = modelo.predict(X_val)

        r2 = r2_score(y_val, pred)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(pred.clip(0))))
        err_prom = np.mean(np.abs(pred - y_val))
        err_pct = np.mean(np.abs(pred - y_val) / y_val) * 100

        resultados[nombre] = {
            'r2': r2,
            'rmsle': rmsle,
            'err_prom': err_prom,
            'err_pct': err_pct,
            'pred': pred,
        }

        print(f"{nombre}:")
        print(f"  R² (accuracy)    = {r2 * 100:.1f}%")
        print(f"  RMSLE            = {rmsle:.5f}")
        print(f"  Error promedio   = ${err_prom:,.0f} por casa")
        print(f"  Error porcentual = {err_pct:.1f}%\n")

    return resultados


def get_best_model(modelos, resultados):
    """El mejor modelo es el que tiene menor RMSLE (métrica oficial de la competencia)."""
    mejor_nombre = min(resultados, key=lambda k: resultados[k]['rmsle'])
    return mejor_nombre, modelos[mejor_nombre]
