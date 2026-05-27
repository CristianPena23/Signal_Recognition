"""
Genera gráficas comparativas entre el modelo de 158 features (votos >= 2)
y el modelo de 100 features (top F-score).

Salida: results/graficas_top100/
  1_metricas_globales.png
  2_metricas_por_vocal_100.png
  3_f1_por_vocal_comparativo.png
  4_matriz_confusion_100.png
  5_radar_precision.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Configuración ──────────────────────────────────────────────────────────────

RUTA_158    = "DataSet/processed/features_vocales_seleccionadas.csv"
RUTA_100    = "DataSet/processed/features_vocales_top100.csv"
MODELO_158  = "models/modelo_randomforest_seleccionadas.pkl"
MODELO_100  = "models/modelo_randomforest_top100.pkl"
LE_158      = "models/label_encoder_seleccionadas.pkl"
LE_100      = "models/label_encoder_top100.pkl"

DIR_SALIDA  = "results/graficas_top100"
META        = {"clase", "persona", "dataset"}
VOCALES     = ["A", "E", "I", "O", "U"]

COLOR_158   = "#3498db"   # azul
COLOR_100   = "#e67e22"   # naranja

os.makedirs(DIR_SALIDA, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Utilidades ─────────────────────────────────────────────────────────────────

def guardar(fig, nombre):
    ruta = os.path.join(DIR_SALIDA, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta}")


def cargar_y_evaluar(ruta_csv, ruta_modelo, ruta_le):
    """Reproduce el mismo split que durante el entrenamiento y devuelve métricas."""
    df       = pd.read_csv(ruta_csv)
    feat_cols = [c for c in df.columns if c not in META]
    X        = df[feat_cols].values
    y        = df["clase"].values

    le    = joblib.load(ruta_le)
    y_enc = le.transform(y)

    _, X_test, _, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    modelo = joblib.load(ruta_modelo)
    y_pred = modelo.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    mat = confusion_matrix(y_test, y_pred)

    return acc, report, mat, le.classes_


# ── Evaluar modelos ────────────────────────────────────────────────────────────

print("Evaluando modelo de 158 features...")
acc_158, rep_158, mat_158, clases_158 = cargar_y_evaluar(RUTA_158, MODELO_158, LE_158)

print("Evaluando modelo de 100 features...")
acc_100, rep_100, mat_100, clases_100 = cargar_y_evaluar(RUTA_100, MODELO_100, LE_100)

print(f"\n  Accuracy 158 features : {acc_158:.4f}")
print(f"  Accuracy 100 features : {acc_100:.4f}")


# ── 1. Barras dobles — métricas globales ───────────────────────────────────────

print("\nGráfica 1: métricas globales comparativas...")

etiquetas = ["Accuracy", "Precisión\n(macro)", "Recall\n(macro)", "F1-score\n(macro)"]
vals_158  = [
    acc_158,
    rep_158["macro avg"]["precision"],
    rep_158["macro avg"]["recall"],
    rep_158["macro avg"]["f1-score"],
]
vals_100  = [
    acc_100,
    rep_100["macro avg"]["precision"],
    rep_100["macro avg"]["recall"],
    rep_100["macro avg"]["f1-score"],
]

x     = np.arange(len(etiquetas))
ancho = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_158 = ax.bar(x - ancho / 2, vals_158, ancho,
                  color=COLOR_158, label="158 features (votos ≥ 2)", edgecolor="white", linewidth=0.8)
bars_100 = ax.bar(x + ancho / 2, vals_100, ancho,
                  color=COLOR_100, label="100 features (top F-score)",  edgecolor="white", linewidth=0.8)

for bar in list(bars_158) + list(bars_100):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.006,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(etiquetas, fontsize=11)
ax.set_ylabel("Valor", fontsize=11)
ax.set_ylim(0, 1.13)
ax.set_title(
    "Comparación de métricas globales\nModelo 158 features vs Modelo 100 features",
    fontsize=13,
)
ax.legend(fontsize=10)
guardar(fig, "1_metricas_globales.png")


# ── 2. Barras agrupadas — precisión, recall y F1 por vocal (modelo 100) ────────

print("Gráfica 2: métricas por vocal — modelo 100 features...")

metricas_nombres = ["precision", "recall", "f1-score"]
etiq_met         = ["Precisión", "Recall", "F1-score"]
colores_met      = ["#2ecc71", "#9b59b6", "#e74c3c"]

x     = np.arange(len(VOCALES))
ancho = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (met, etiq, col) in enumerate(zip(metricas_nombres, etiq_met, colores_met)):
    vals   = [rep_100[v][met] for v in VOCALES]
    offset = (i - 1) * ancho
    bars   = ax.bar(x + offset, vals, ancho,
                    color=col, label=etiq, edgecolor="white", linewidth=0.8)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8.5,
        )

ax.set_xticks(x)
ax.set_xticklabels(VOCALES, fontsize=12)
ax.set_ylabel("Valor", fontsize=11)
ax.set_ylim(0, 1.13)
ax.set_title(
    "Precisión, Recall y F1-score por vocal\nModelo de 100 features (top F-score)",
    fontsize=13,
)
ax.legend(fontsize=11)
guardar(fig, "2_metricas_por_vocal_100.png")


# ── 3. Barras agrupadas — F1-score por vocal, ambos modelos ───────────────────

print("Gráfica 3: F1-score por vocal — comparativo...")

f1_158 = [rep_158[v]["f1-score"] for v in VOCALES]
f1_100 = [rep_100[v]["f1-score"] for v in VOCALES]

x     = np.arange(len(VOCALES))
ancho = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_158 = ax.bar(x - ancho / 2, f1_158, ancho,
                  color=COLOR_158, label="158 features (votos ≥ 2)", edgecolor="white", linewidth=0.8)
bars_100 = ax.bar(x + ancho / 2, f1_100, ancho,
                  color=COLOR_100, label="100 features (top F-score)",  edgecolor="white", linewidth=0.8)

for bar in list(bars_158) + list(bars_100):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(VOCALES, fontsize=12)
ax.set_ylabel("F1-score", fontsize=11)
ax.set_ylim(0, 1.13)
ax.set_title(
    "Comparación de F1-score por vocal\nModelo 158 features vs Modelo 100 features",
    fontsize=13,
)
ax.legend(fontsize=10)
guardar(fig, "3_f1_por_vocal_comparativo.png")


# ── 4. Heatmap matriz de confusión (modelo 100) ────────────────────────────────

print("Gráfica 4: heatmap matriz de confusión — modelo 100 features...")

totales_fila = mat_100.sum(axis=1, keepdims=True)
mat_norm     = mat_100.astype(float) / totales_fila

anotaciones = np.array([
    [f"{mat_100[i, j]}\n({mat_norm[i, j] * 100:.1f}%)"
     for j in range(len(VOCALES))]
    for i in range(len(VOCALES))
])

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    mat_norm,
    annot=anotaciones,
    fmt="",
    cmap="Blues",
    xticklabels=clases_100,
    yticklabels=clases_100,
    linewidths=0.6,
    linecolor="white",
    vmin=0, vmax=1,
    cbar_kws={"label": "Proporción por fila"},
    ax=ax,
    annot_kws={"size": 11},
)
ax.set_xlabel("Predicción", fontsize=12, labelpad=8)
ax.set_ylabel("Etiqueta real", fontsize=12, labelpad=8)
ax.set_title(
    "Matriz de confusión — Modelo 100 features (top F-score)\n"
    "Valores absolutos y porcentaje por fila",
    fontsize=12,
)
plt.tight_layout()
guardar(fig, "4_matriz_confusion_100.png")


# ── 5. Radar (spider chart) — precisión por vocal, ambos modelos ───────────────

print("Gráfica 5: radar de precisión por vocal...")

prec_158 = [rep_158[v]["precision"] for v in VOCALES]
prec_100 = [rep_100[v]["precision"] for v in VOCALES]

N       = len(VOCALES)
angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# cerrar el polígono repitiendo el primer punto
angulos_c  = angulos + angulos[:1]
prec_158_c = prec_158 + prec_158[:1]
prec_100_c = prec_100 + prec_100[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.plot(angulos_c, prec_158_c, "o-", color=COLOR_158, linewidth=2.2,
        label="158 features (votos ≥ 2)")
ax.fill(angulos_c, prec_158_c, color=COLOR_158, alpha=0.18)

ax.plot(angulos_c, prec_100_c, "o-", color=COLOR_100, linewidth=2.2,
        label="100 features (top F-score)")
ax.fill(angulos_c, prec_100_c, color=COLOR_100, alpha=0.18)

# Añadir etiquetas con el valor exacto de cada vértice
for ang, v158, v100, vocal in zip(angulos, prec_158, prec_100, VOCALES):
    ax.text(ang, v158 + 0.04, f"{v158:.2f}", ha="center", va="center",
            fontsize=9, color=COLOR_158, fontweight="bold")
    ax.text(ang, v100 - 0.07, f"{v100:.2f}", ha="center", va="center",
            fontsize=9, color=COLOR_100, fontweight="bold")

ax.set_xticks(angulos)
ax.set_xticklabels(VOCALES, fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
ax.set_title(
    "Precisión por vocal — Comparación\nModelo 158 features vs Modelo 100 features",
    fontsize=13, pad=22,
)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=10)
guardar(fig, "5_radar_precision.png")


# ── Resumen final ──────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("  Resumen de resultados")
print("=" * 55)
print(f"  {'Métrica':<28} {'158 feats':>9}  {'100 feats':>9}")
print(f"  {'-'*48}")
print(f"  {'Accuracy':<28} {acc_158:>9.4f}  {acc_100:>9.4f}")
for met, etiq in [("precision", "Precisión macro"), ("recall", "Recall macro"), ("f1-score", "F1-score macro")]:
    v158 = rep_158["macro avg"][met]
    v100 = rep_100["macro avg"][met]
    print(f"  {etiq:<28} {v158:>9.4f}  {v100:>9.4f}")
print(f"\n  Gráficas guardadas en: {DIR_SALIDA}/")
