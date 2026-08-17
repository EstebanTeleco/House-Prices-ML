"""
Tests para src/data_processing.py

Se corren con: pytest tests/
No dependen de train.csv/test.csv reales, arman DataFrames chicos a mano.
"""

import numpy as np
import pandas as pd

from src.data_processing import (
    remove_outliers, fill_missing_values, merge_train_test,
    COLUMNS_FILL_ZERO, QUALITY_COLUMNS,
)


def test_remove_outliers_saca_casas_grandes_y_baratas():
    """Las 2 casas con GrLivArea > 4000 y SalePrice < 300000 deben desaparecer."""
    train = pd.DataFrame({
        'GrLivArea': [5000, 4500, 1500, 3000],
        'SalePrice': [100000, 200000, 250000, 350000],
    })
    resultado = remove_outliers(train)

    assert len(resultado) == 2
    assert 5000 not in resultado['GrLivArea'].values
    assert 4500 not in resultado['GrLivArea'].values


def test_remove_outliers_no_saca_casas_grandes_pero_caras():
    """Una casa grande con precio alto no es un outlier según la regla, no se saca."""
    train = pd.DataFrame({
        'GrLivArea': [4500],
        'SalePrice': [400000],  # no es < 300000, así que se queda
    })
    resultado = remove_outliers(train)

    assert len(resultado) == 1


def test_remove_outliers_resetea_el_index():
    train = pd.DataFrame({
        'GrLivArea': [5000, 1500, 1800],
        'SalePrice': [100000, 250000, 260000],
    })
    resultado = remove_outliers(train)

    assert list(resultado.index) == list(range(len(resultado)))


def _df_base(n=4):
    """DataFrame mínimo con todas las columnas que fill_missing_values necesita."""
    data = {col: [100.0] * n for col in COLUMNS_FILL_ZERO}
    data['Neighborhood'] = ['NAmes', 'NAmes', 'CollgCr', 'CollgCr']
    data['LotFrontage'] = [60.0, np.nan, 80.0, np.nan]
    for col in QUALITY_COLUMNS:
        data[col] = ['Gd'] * n
    return pd.DataFrame(data)


def test_fill_missing_values_columnas_de_cero_no_quedan_nulas():
    df = _df_base()
    for col in COLUMNS_FILL_ZERO:
        df.loc[0, col] = np.nan

    resultado = fill_missing_values(df)

    for col in COLUMNS_FILL_ZERO:
        assert resultado[col].isna().sum() == 0
    assert resultado.loc[0, COLUMNS_FILL_ZERO[0]] == 0


def test_fill_missing_values_lot_frontage_usa_mediana_del_barrio():
    """LotFrontage nulo debe rellenarse con la mediana de SU barrio, no la global."""
    df = _df_base()
    resultado = fill_missing_values(df)

    # NAmes tiene [60, nan] -> la mediana del barrio es 60
    assert resultado.loc[1, 'LotFrontage'] == 60.0
    # CollgCr tiene [80, nan] -> la mediana del barrio es 80
    assert resultado.loc[3, 'LotFrontage'] == 80.0


def test_fill_missing_values_calidad_se_mapea_a_numero():
    df = _df_base()
    df.loc[0, 'ExterQual'] = 'Ex'
    df.loc[1, 'ExterQual'] = 'Po'
    df.loc[2, 'ExterQual'] = np.nan  # sin dato -> se trata como 'None' -> 0

    resultado = fill_missing_values(df)

    assert resultado.loc[0, 'ExterQual'] == 5
    assert resultado.loc[1, 'ExterQual'] == 1
    assert resultado.loc[2, 'ExterQual'] == 0


def test_merge_train_test_cuenta_filas_de_train_correctamente():
    train = pd.DataFrame({'Id': [1, 2], 'SalePrice': [100, 200], 'X': [1, 2]})
    test = pd.DataFrame({'Id': [3, 4, 5], 'X': [3, 4, 5]})

    all_data, n_train = merge_train_test(train, test)

    assert n_train == 2
    assert len(all_data) == 5
    assert 'SalePrice' not in all_data.columns
    assert 'Id' not in all_data.columns
