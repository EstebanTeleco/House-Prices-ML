"""
Gráficos para comparar visualmente los 3 modelos.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison(resultados, mejor_nombre, mejor_modelo, features, y_val,
                     output_path='outputs/comparacion_modelos.png'):
    """
    Arma una figura con 4 paneles:
      1. R² de cada modelo
      2. RMSLE de cada modelo
      3. Predicho vs Real del mejor modelo
      4. Las features más importantes del mejor modelo (si aplica)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de 3 Modelos — House Prices', fontsize=15, fontweight='bold')

    nombres = list(resultados.keys())
    colores = ['#4F8EF7', '#3DD68C', '#F5C842']

    # Panel 1: R²
    ax = axes[0, 0]
    r2s = [resultados[n]['r2'] * 100 for n in nombres]
    bars = ax.bar(nombres, r2s, color=colores, width=0.5)
    ax.set_title('Accuracy (R²) por Modelo')
    ax.set_ylabel('R² (%)')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Panel 2: RMSLE
    ax = axes[0, 1]
    rmsles = [resultados[n]['rmsle'] for n in nombres]
    bars = ax.bar(nombres, rmsles, color=colores, width=0.5)
    ax.set_title('RMSLE por Modelo (menor = mejor)')
    ax.set_ylabel('RMSLE')
    for bar, val in zip(bars, rmsles):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Panel 3: Predicho vs Real (mejor modelo)
    ax = axes[1, 0]
    pred_mejor = resultados[mejor_nombre]['pred']
    ax.scatter(y_val, pred_mejor, alpha=0.4, color='#4F8EF7', s=20)
    mn, mx = y_val.min(), y_val.max()
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Predicción perfecta')
    ax.set_xlabel('Precio real')
    ax.set_ylabel('Precio predicho')
    ax.set_title(f'Predicho vs Real — {mejor_nombre}')
    ax.legend()

    # Panel 4: Top features (solo si el modelo las expone)
    ax = axes[1, 1]
    if hasattr(mejor_modelo, 'feature_importances_'):
        importancias = pd.Series(
            mejor_modelo.feature_importances_, index=features
        ).nlargest(12).sort_values()
        bc = ['#F5C842' if v == importancias.max() else '#4F8EF7' for v in importancias]
        ax.barh(importancias.index, importancias.values, color=bc)
        ax.set_title(f'Top 12 Features — {mejor_nombre}')
        ax.set_xlabel('Importancia')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Regresión Lineal no tiene\nfeature importance',
                ha='center', va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f"✅ {output_path} guardado")
