"""
Definición y entrenamiento de los 3 modelos que se comparan en el proyecto.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def train_models(X_train, y_train):
    """
    Entrena un modelo de regresión lineal, un Random Forest y un XGBoost
    con los mismos datos de entrenamiento, y los devuelve en un diccionario.
    """
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)

    modelo_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    modelo_rf.fit(X_train, y_train)

    modelo_xgb = xgb.XGBRegressor(
        n_estimators=500,      # número de árboles
        learning_rate=0.05,    # qué tan rápido aprende
        max_depth=4,           # profundidad máxima de cada árbol
        colsample_bytree=0.8,  # usa 80% de las features por árbol
        subsample=0.8,         # usa 80% de las filas por árbol
        random_state=42,
        verbosity=0
    )
    modelo_xgb.fit(X_train, y_train)

    return {
        'Regresión Lineal': modelo_lineal,
        'Random Forest': modelo_rf,
        'XGBoost': modelo_xgb,
    }
