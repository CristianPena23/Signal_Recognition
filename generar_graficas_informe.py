"""
Genera las gráficas del informe a partir de features_vocales_seleccionadas.csv.

Salidas en results/graficas_informe/:
  1_boxplots_top10.png          — Box plots de las 10 features más discriminativas
  2_violinplots_top10.png       — Violin plots de las mismas 10 features
  3_pca_scatter_elipses.png     — Scatter PCA 2D con elipses de confianza por vocal
  4_pca_varianza_acumulada.png  — Varianza explicada por componente PCA
  5_heatmap_correlacion.png     — Heatmap de correlación entre features seleccionadas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

# ── Configuración ─────────────────────────────────────────────────────────────

RUTA_CSV = "DataSet/processed/features_vocales_seleccionadas.csv"
DIR_OUT  = "results/graficas_informe"
DPI      = 150
N_TOP    = 10   # features más discriminativas a graficar

VOCALES = ["A", "E", "I", "O", "U"]
PALETA  = {
    "A": "#e74c3c",
    "E": "#3498db",
    "I": "#2ecc71",
    "O": "#f39c12",
    "U": "#9b59b6",
}
META = {"clase", "persona", "dataset"}

os.makedirs(DIR_OUT, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": DPI})


# ── Utilidades ────────────────────────────────────────────────────────────────

def guardar(fig, nombre):
    ruta = os.path.join(DIR_OUT, nombre)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta}")


def _elipse_confianza(x, y, ax, n_std=2.0, color="blue", alpha_relleno=0.12):
    """
    Dibuja una elipse de confianza de n_std desviaciones estándar
    usando la transformación afín sobre la covarianza 2D.
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    rx = np.sqrt(max(1 + pearson, 0))
    ry = np.sqrt(max(1 - pearson, 0))

    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    mx, my = np.mean(x), np.mean(y)
    t = transforms.Affine2D().rotate_deg(45).scale(sx, sy).translate(mx, my)

    relleno = Ellipse((0, 0), width=rx * 2, height=ry * 2,
                      facecolor=color, edgecolor="none", alpha=alpha_relleno)
    borde   = Ellipse((0, 0), width=rx * 2, height=ry * 2,
                      facecolor="none", edgecolor=color, linewidth=2)
    for patch in (relleno, borde):
        patch.set_transform(t + ax.transData)
        ax.add_patch(patch)


def _nombre_legible(feat):
    """Convierte nombres internos a texto más limpio para los ejes."""
    return feat.replace("_", " ")


# ── Carga de datos ────────────────────────────────────────────────────────────

print("Cargando datos...")
df = pd.read_csv(RUTA_CSV)
df = df[df["clase"].isin(VOCALES)].reset_index(drop=True)

feat_cols = [c for c in df.columns if c not in META]
X = df[feat_cols].values.astype(np.float64)
y = df["clase"].values

le    = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"  Features : {len(feat_cols)}")
print(f"  Muestras : {len(X)}")
print(f"  Clases   : {dict(pd.Series(y).value_counts().sort_index())}")

# Top-10 features por F-score (ANOVA)
f_scores, _ = f_classif(X, y_enc)
f_scores    = np.nan_to_num(f_scores)
top10_idx   = np.argsort(f_scores)[::-1][:N_TOP]
top10_feats = [feat_cols[i] for i in top10_idx]

print(f"\n  Top {N_TOP} features (F-score):")
for i, feat in enumerate(top10_feats, 1):
    print(f"    {i:>2}. {feat:<35}  F={f_scores[feat_cols.index(feat)]:.2f}")


# ── 1. Box plots ──────────────────────────────────────────────────────────────

print("\n[1/5] Box plots...")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for idx, feat in enumerate(top10_feats):
    ax = axes[idx]
    datos = [df.loc[df["clase"] == v, feat].values for v in VOCALES]

    bp = ax.boxplot(
        datos,
        patch_artist=True,
        labels=VOCALES,
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2, color="#555555"),
        capprops=dict(linewidth=1.2, color="#555555"),
        flierprops=dict(marker="o", markersize=3, alpha=0.4,
                        markeredgecolor="none"),
        widths=0.55,
    )
    for patch, vocal in zip(bp["boxes"], VOCALES):
        patch.set_facecolor(PALETA[vocal])
        patch.set_alpha(0.80)

    f_val = f_scores[feat_cols.index(feat)]
    ax.set_title(f"{_nombre_legible(feat)}\nF = {f_val:.1f}",
                 fontsize=8.5, fontweight="bold", pad=4)
    ax.set_ylabel("Valor normalizado", fontsize=8)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.suptitle(
    f"Box plots — Top {N_TOP} features más discriminativas por vocal",
    fontsize=14, fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
guardar(fig, "1_boxplots_top10.png")


# ── 2. Violin plots ───────────────────────────────────────────────────────────

print("[2/5] Violin plots...")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for idx, feat in enumerate(top10_feats):
    ax = axes[idx]
    datos_long = pd.DataFrame({"Vocal": y, "Valor": df[feat].values})
    datos_long["Vocal"] = pd.Categorical(datos_long["Vocal"], categories=VOCALES)

    sns.violinplot(
        data=datos_long,
        x="Vocal", y="Valor",
        order=VOCALES,
        palette=PALETA,
        inner="box",
        linewidth=0.9,
        ax=ax,
    )
    f_val = f_scores[feat_cols.index(feat)]
    ax.set_title(f"{_nombre_legible(feat)}\nF = {f_val:.1f}",
                 fontsize=8.5, fontweight="bold", pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("Valor normalizado", fontsize=8)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle(
    f"Violin plots — Top {N_TOP} features más discriminativas por vocal",
    fontsize=14, fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
guardar(fig, "2_violinplots_top10.png")


# ── 3. Scatter PCA 2D con elipses de confianza ───────────────────────────────

print("[3/5] Scatter PCA 2D con elipses...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca2     = PCA(n_components=2, random_state=42)
X_pca    = pca2.fit_transform(X_scaled)
var2     = pca2.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))

for vocal in VOCALES:
    mask = y == vocal
    xv, yv = X_pca[mask, 0], X_pca[mask, 1]

    # Puntos
    ax.scatter(xv, yv, color=PALETA[vocal], label=vocal,
               alpha=0.40, s=20, edgecolors="none", zorder=3)
    # Centroide
    ax.scatter(np.mean(xv), np.mean(yv), color=PALETA[vocal],
               s=120, marker="*", edgecolors="white", linewidths=0.8,
               zorder=5)
    # Elipse de confianza (2σ)
    _elipse_confianza(xv, yv, ax, n_std=2.0, color=PALETA[vocal])

ax.set_xlabel(f"Componente principal 1  ({var2[0]*100:.1f}% varianza)",
              fontsize=11)
ax.set_ylabel(f"Componente principal 2  ({var2[1]*100:.1f}% varianza)",
              fontsize=11)
ax.set_title("PCA 2D — Distribución de vocales con elipses de confianza (2σ)",
             fontsize=13, fontweight="bold")

handles = [
    plt.scatter([], [], color=PALETA[v], s=60, label=f"Vocal {v}",
                edgecolors="none")
    for v in VOCALES
]
ax.legend(handles=handles, title="Vocal", fontsize=10, title_fontsize=10,
          framealpha=0.85, loc="best")
ax.grid(linestyle="--", alpha=0.35)

guardar(fig, "3_pca_scatter_elipses.png")


# ── 4. Varianza explicada PCA ─────────────────────────────────────────────────

print("[4/5] Varianza explicada PCA...")

n_comp   = min(30, len(feat_cols), X.shape[0] - 1)
pca_full = PCA(n_components=n_comp, random_state=42)
pca_full.fit(X_scaled)

var_ind  = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_ind)
ejes_x   = np.arange(1, n_comp + 1)

n_90 = int(np.searchsorted(var_acum, 0.90)) + 1
n_95 = int(np.searchsorted(var_acum, 0.95)) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Varianza individual en barras
axes[0].bar(ejes_x, var_ind * 100, color="#3498db", alpha=0.85,
            edgecolor="white")
axes[0].set_xlabel("Componente principal", fontsize=11)
axes[0].set_ylabel("Varianza explicada (%)", fontsize=11)
axes[0].set_title("Varianza individual por componente", fontsize=12,
                  fontweight="bold")
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

# Varianza acumulada con líneas de referencia
axes[1].plot(ejes_x, var_acum * 100, "o-", color="#e74c3c",
             linewidth=2, markersize=5, zorder=3)
axes[1].fill_between(ejes_x, var_acum * 100, alpha=0.12, color="#e74c3c")
axes[1].axhline(90, ls="--", color="#f39c12", lw=1.5, label=f"90%  (PC {n_90})")
axes[1].axhline(95, ls="--", color="#9b59b6", lw=1.5, label=f"95%  (PC {n_95})")
axes[1].axvline(n_90, ls=":", color="#f39c12", lw=1.2)
axes[1].axvline(n_95, ls=":", color="#9b59b6", lw=1.2)
axes[1].set_xlabel("Número de componentes principales", fontsize=11)
axes[1].set_ylabel("Varianza acumulada (%)", fontsize=11)
axes[1].set_title("Varianza acumulada (curva codo)", fontsize=12,
                  fontweight="bold")
axes[1].legend(fontsize=10, framealpha=0.85)
axes[1].set_ylim(0, 102)
axes[1].grid(linestyle="--", alpha=0.4)

fig.suptitle("Análisis de Componentes Principales (PCA) — Features seleccionadas",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
guardar(fig, "4_pca_varianza_acumulada.png")


# ── 5. Heatmap de correlación ─────────────────────────────────────────────────

print("[5/5] Heatmap de correlación...")

corr = pd.DataFrame(X, columns=feat_cols).corr()
n_f  = len(feat_cols)

# Ajustar tamaño: más features → celda más pequeña, sin etiquetas
lado    = max(10, min(22, n_f * 0.14))
mostrar = n_f <= 60       # mostrar etiquetas solo si hay pocas features

fig, ax = plt.subplots(figsize=(lado, lado * 0.88))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    xticklabels=mostrar,
    yticklabels=mostrar,
    linewidths=0 if n_f > 60 else 0.3,
    cbar_kws={"shrink": 0.7, "label": "Correlación de Pearson"},
    ax=ax,
)
if mostrar:
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

ax.set_title(
    f"Heatmap de correlación — {n_f} features seleccionadas\n"
    "(colores: azul = correlación negativa, rojo = positiva)",
    fontsize=12, fontweight="bold", pad=12,
)
guardar(fig, "5_heatmap_correlacion.png")


# ── Resumen ───────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Gráficas generadas en: {DIR_OUT}/")
print(f"  1_boxplots_top10.png")
print(f"  2_violinplots_top10.png")
print(f"  3_pca_scatter_elipses.png")
print(f"  4_pca_varianza_acumulada.png")
print(f"  5_heatmap_correlacion.png")
print("=" * 55)
