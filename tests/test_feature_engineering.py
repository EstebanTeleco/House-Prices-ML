"""
Tests para src/feature_engineering.py
"""

import pandas as pd

from src.feature_engineering import create_features, split_features_target, FEATURES


def _df_base():
    return pd.DataFrame({
        'TotalBsmtSF': [500.0],
        '1stFlrSF': [800.0],
        '2ndFlrSF': [400.0],
        'OverallQual': [7],
        'GrLivArea': [1200.0],
        'FullBath': [2],
        'HalfBath': [1],
        'BsmtFullBath': [1],
        'BsmtHalfBath': [0],
        'YrSold': [2010],
        'YearBuilt': [1990],
        'YearRemodAdd': [2005],
        'OpenPorchSF': [50],
        'WoodDeckSF': [100],
        'EnclosedPorch': [20],
        'GarageArea': [0],
        'Fireplaces': [0],
    })


def test_total_sf_es_la_suma_de_sotano_y_pisos():
    df = create_features(_df_base())
    assert df.loc[0, 'TotalSF'] == 500.0 + 800.0 + 400.0


def test_qual_sf_multiplica_calidad_por_superficie():
    df = create_features(_df_base())
    assert df.loc[0, 'QualSF'] == 7 * 1200.0


def test_total_baths_cuenta_medios_banos_como_0_5():
    df = create_features(_df_base())
    # 2 full + 0.5*1 half + 1 bsmt full + 0.5*0 bsmt half = 3.5
    assert df.loc[0, 'TotalBaths'] == 3.5


def test_house_age_y_remod_age():
    df = create_features(_df_base())
    assert df.loc[0, 'HouseAge'] == 2010 - 1990
    assert df.loc[0, 'RemodAge'] == 2010 - 2005


def test_total_porch_suma_las_tres_areas():
    df = create_features(_df_base())
    assert df.loc[0, 'TotalPorch'] == 50 + 100 + 20


def test_has_garage_y_has_fireplace_son_binarias():
    df = _df_base()
    df.loc[0, 'GarageArea'] = 300
    df.loc[0, 'Fireplaces'] = 2
    resultado = create_features(df)

    assert resultado.loc[0, 'HasGarage'] == 1
    assert resultado.loc[0, 'HasFireplace'] == 1

    sin_garage = create_features(_df_base())  # GarageArea=0, Fireplaces=0
    assert sin_garage.loc[0, 'HasGarage'] == 0
    assert sin_garage.loc[0, 'HasFireplace'] == 0


def test_split_features_target_separa_train_y_test_bien():
    all_data = pd.DataFrame({col: [1, 2, 3] for col in FEATURES})
    train = pd.DataFrame({'SalePrice': [100, 200]})
    n_train = 2

    X, X_test, y = split_features_target(all_data, train, n_train)

    assert len(X) == 2
    assert len(X_test) == 1
    assert list(y) == [100, 200]
    assert list(X.columns) == FEATURES
