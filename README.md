# House Prices - Predicción de precios de casas

Proyecto de machine learning para la competencia de Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). La idea es predecir el precio de venta de una casa a partir de sus características (tamaño, calidad, año de construcción, barrio, etc), usando el dataset de Ames, Iowa.

Lo hice para practicar el flujo completo de un problema de regresión: limpieza de datos, creación de features, entrenamiento de varios modelos y comparación entre ellos.

## Qué hace

El script entrena y compara tres modelos distintos sobre los mismos datos:

- **Regresión Lineal** — el modelo más simple, sirve de referencia.
- **Random Forest** — un ensamble de árboles de decisión.
- **XGBoost** — boosting, en general el que mejor generaliza en este tipo de problemas.

Al final se queda con el que tenga menor RMSLE (la métrica que usa la competencia) y con ese arma el `submission.csv` para subir a Kaggle.

## Estructura

```
house-prices-ml/
├── data/               # acá van train.csv y test.csv (no están en el repo, ver data/README.md)
├── outputs/            # submission.csv y el gráfico de comparación se generan acá
├── src/
│   ├── data_processing.py     # carga de datos, outliers, imputación de nulos
│   ├── feature_engineering.py # features nuevas y lista final de columnas
│   ├── models.py               # definición y entrenamiento de los 3 modelos
│   ├── evaluate.py             # métricas y selección del mejor modelo
│   └── visualize.py            # gráficos de comparación
├── main.py              # corre todo el pipeline de punta a punta
└── requirements.txt
```

## Cómo correrlo

1. Cloná el repo e instalá las dependencias:

```bash
git clone https://github.com/tu-usuario/house-prices-ml.git
cd house-prices-ml
pip install -r requirements.txt
```

2. Descargá `train.csv` y `test.csv` de la [página de la competencia en Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) y ponelos en `data/`.

3. Corré:

```bash
python main.py
```

Esto va a imprimir las métricas de cada modelo en la consola y va a dejar en `outputs/` el `submission.csv` (listo para subir a Kaggle) y un `comparacion_modelos.png` con los gráficos.

## Sobre el procesamiento de datos

Algunas decisiones que tomé al limpiar los datos:

- Hay dos casas en el train con un área gigante pero un precio muy bajo — son errores conocidos del dataset, así que las saco antes de entrenar.
- Para varias columnas (garage, sótano, chimenea) un valor nulo en realidad significa "la casa no tiene eso", no un dato faltante de verdad, así que las relleno con 0 en vez de con la media.
- `LotFrontage` lo relleno con la mediana del barrio en lugar de la mediana general, porque el tamaño de frente varía bastante según la zona.
- Las columnas de calidad (`ExterQual`, `KitchenQual`, etc.) vienen como texto (`Po`, `Fa`, `TA`, `Gd`, `Ex`) pero tienen un orden lógico, así que las paso a una escala numérica del 1 al 5.

También armé algunas features nuevas combinando las que ya venían: superficie total de la casa, calidad multiplicada por tamaño, baños totales, antigüedad de la casa y años desde la última remodelación, entre otras. Ninguna de estas te la da el dataset directamente, pero se pueden calcular y le dan más información al modelo.

## Resultados

Al correr `main.py`, la consola muestra las métricas de cada modelo sobre el set de validación (80/20 sobre el train), con este formato:

```
XGBoost:
  R² (accuracy)    = ...
  RMSLE            = ...
  Error promedio   = $... por casa
  Error porcentual = ...%
```

El script elige automáticamente el modelo con menor RMSLE (la métrica que usa la competencia) para armar el `submission.csv`, y deja los números completos de los tres modelos, más el gráfico comparativo, en `outputs/`.

*(No dejo un número fijo acá en el README porque cambia un poco según la versión de los datos y el split; lo vas a ver directo en la consola al correrlo.)*

## Ideas para seguir mejorando

- Probar con validación cruzada en vez de un solo split train/val.
- Hacer tuning de hiperparámetros más en serio (grid search u Optuna) en vez de los valores que puse a mano.
- Meter un stacking o un promedio ponderado de los 3 modelos.
- Trabajar más las variables categóricas que quedaron afuera (ahora mismo el foco está en las numéricas y en las de calidad).

## Tecnologías

Python, pandas, scikit-learn, XGBoost, matplotlib.
