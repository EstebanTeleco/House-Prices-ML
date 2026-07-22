# ================================================
#  HOUSE PRICES - KAGGLE
#  3 modelos: Regresión Lineal, Random Forest y XGBoost
# ================================================

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import pandas as pd

from src.data_processing import (
    load_data, remove_outliers, merge_train_test, fill_missing_values
)
from src.feature_engineering import create_features, split_features_target, FEATURES
from src.models import train_models
from src.evaluate import evaluate_models, get_best_model
from src.visualize import plot_comparison


def main():
    # 1. Cargar datos
    train, test = load_data()
    print(f"Train: {train.shape}  |  Test: {test.shape}")

    # 2. Sacar outliers conocidos del dataset
    train = remove_outliers(train)
    print(f"Sin outliers: {train.shape}")

    # 3. Juntar train y test para procesarlos igual
    all_data, n_train = merge_train_test(train, test)

    # 4. Rellenar nulos
    all_data = fill_missing_values(all_data)

    # 5. Crear features nuevas
    all_data = create_features(all_data)

    # 6. Armar X, y
    X, X_test, y = split_features_target(all_data, train, n_train)
    print(f"\nFeatures totales usadas: {len(FEATURES)}")

    # 7. Separar entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} filas  |  Validación: {X_val.shape[0]} filas")

    # 8. Entrenar los 3 modelos
    modelos = train_models(X_train, y_train)
    print("\n✅ Los 3 modelos fueron entrenados")

    # 9. Evaluar y comparar
    print("\n── RESULTADOS ───────────────────────────────")
    resultados = evaluate_models(modelos, X_val, y_val)

    # 10. Elegir el mejor modelo y generar el submission
    mejor_nombre, mejor_modelo = get_best_model(modelos, resultados)
    print(f"🏆 Mejor modelo: {mejor_nombre}")

    predicciones = mejor_modelo.predict(X_test)
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': predicciones
    })
    submission.to_csv('outputs/submission.csv', index=False)
    print(f"✅ outputs/submission.csv creado usando {mejor_nombre}")

    # 11. Gráficos comparativos
    plot_comparison(resultados, mejor_nombre, mejor_modelo, FEATURES, y_val)


if __name__ == '__main__':
    main()
