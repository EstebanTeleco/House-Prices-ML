"""
Carga y limpieza de los datos de train/test.
"""

import pandas as pd

# Columnas donde un valor nulo en realidad significa "la casa no tiene esto",
# así que en vez de imputar con la media o mediana, ponemos 0.
COLUMNS_FILL_ZERO = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
]

# Las columnas de calidad vienen como texto pero tienen un orden claro,
# así que las pasamos a números para que los modelos las puedan usar.
QUALITY_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

QUALITY_COLUMNS = [
    'ExterQual', 'KitchenQual', 'BsmtQual',
    'FireplaceQu', 'GarageQual', 'ExterCond', 'HeatingQC',
]


def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Lee los CSV de train y test."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def remove_outliers(train):
    """
    Saca del train un par de casas con área enorme pero precio muy bajo.
    Son errores conocidos del dataset (se comentan en varias soluciones de
    la competencia) y si se dejan, arrastran a los modelos a hacer un mal ajuste.
    """
    train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]
    return train.reset_index(drop=True)


def merge_train_test(train, test):
    """
    Junta train y test en un solo DataFrame para aplicarles el mismo
    preprocesamiento (así no hay riesgo de tratar una columna distinto
    en cada uno). Devuelve también cuántas filas eran del train original,
    para poder separarlos después.
    """
    n_train = len(train)
    all_data = pd.concat(
        [train.drop(['SalePrice', 'Id'], axis=1),
         test.drop('Id', axis=1)],
        ignore_index=True
    )
    return all_data, n_train


def fill_missing_values(all_data):
    """Imputa los valores nulos según el significado de cada columna."""
    all_data = all_data.copy()

    for col in COLUMNS_FILL_ZERO:
        all_data[col] = all_data[col].fillna(0)

    # LotFrontage: usamos la mediana del barrio en lugar de la mediana
    # global, porque el tamaño de frente de una casa depende mucho de
    # en qué zona está.
    all_data['LotFrontage'] = (
        all_data.groupby('Neighborhood')['LotFrontage']
        .transform(lambda x: x.fillna(x.median()))
    )

    for col in QUALITY_COLUMNS:
        all_data[col] = all_data[col].fillna('None').map(QUALITY_MAP).fillna(0)

    return all_data
