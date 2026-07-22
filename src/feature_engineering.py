"""
Creación de nuevas variables a partir de las que ya vienen en el dataset,
y la lista final de features que se le pasa a los modelos.
"""

FEATURES = [
    # Features originales del dataset
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath',
    'YearBuilt', 'YearRemodAdd', 'LotArea', 'Fireplaces', '1stFlrSF', '2ndFlrSF',
    'GarageArea', 'OpenPorchSF', 'WoodDeckSF', 'LotFrontage', 'MasVnrArea',
    'OverallCond', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt',
    # Calidades convertidas a número en data_processing.py
    'ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'GarageQual', 'HeatingQC',
    # Features nuevas, creadas acá abajo
    'TotalSF', 'QualSF', 'QualTotSF', 'TotalBaths',
    'HouseAge', 'RemodAge', 'TotalPorch',
    'HasGarage', 'HasFireplace',
]


def create_features(all_data):
    """
    Combina columnas existentes para darle al modelo información que no
    está de forma directa en el dataset original (superficie total,
    antigüedad de la casa, etc).
    """
    all_data = all_data.copy()

    # Superficie total: sótano + primer piso + segundo piso
    all_data['TotalSF'] = (
        all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    )

    # Calidad x tamaño: una casa grande y de buena calidad vale bastante
    # más que la simple suma de esas dos cosas por separado
    all_data['QualSF'] = all_data['OverallQual'] * all_data['GrLivArea']
    all_data['QualTotSF'] = all_data['OverallQual'] * all_data['TotalSF']

    # Baños totales (medio baño cuenta como 0.5)
    all_data['TotalBaths'] = (
        all_data['FullBath'] +
        0.5 * all_data['HalfBath'] +
        all_data['BsmtFullBath'] +
        0.5 * all_data['BsmtHalfBath']
    )

    # Antigüedad de la casa y años desde la última remodelación
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']

    # Porche total
    all_data['TotalPorch'] = (
        all_data['OpenPorchSF'] + all_data['WoodDeckSF'] + all_data['EnclosedPorch']
    )

    # Variables binarias: ¿tiene o no tiene?
    all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

    return all_data


def split_features_target(all_data, train, n_train):
    """Separa de nuevo el X/y de train y el X de test, ya con todo procesado."""
    X = all_data[FEATURES].iloc[:n_train].fillna(0)
    X_test = all_data[FEATURES].iloc[n_train:].fillna(0)
    y = train['SalePrice']
    return X, X_test, y
