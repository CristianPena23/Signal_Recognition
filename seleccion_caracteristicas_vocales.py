"""
Pipeline de selección de características para las vocales A, E, I, O, U.

Etapas:
  1. Correlación de Pearson        — eliminar features redundantes (|r| > 0.9)
  2. ANOVA + FDR B-H + Bonferroni  — features estadísticamente significativas
  3. PCA                           — varianza explicada + scatter 2D
  4. SelectKBest (F-score y MI)    — ranking automático top-K
  5. Sistema de votos              — features en ≥ 2 métodos → conjunto final
  6. Box plots y violin plots      — conjunto final por vocal
  7. Guardado de gráficas y CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif

# ── Configuración ─────────────────────────────────────────────────────────────

RUTA_FEATURES   = "DataSet/processed/features_vocales.csv"
DIR_RESULTADOS  = "results/seleccion_vocales"
RUTA_SALIDA_CSV = "DataSet/processed/features_vocales_seleccionadas.csv"

UMBRAL_CORR  = 0.90   # eliminar si |r| > umbral con alguna feature ya retenida
ALPHA        = 0.05   # nivel de significancia ANOVA
K_BEST       = 50     # top-K para SelectKBest (se ajusta automáticamente)
N_VIZUALIZAR = 15     # máximo de features en box/violin plots

VOCALES = ["A", "E", "I", "O", "U"]
PALETA  = {"A": "#e74c3c", "E": "#3498db", "I": "#2ecc71",
           "O": "#f39c12", "U": "#9b59b6"}

os.makedirs(DIR_RESULTADOS, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.0)

META = {"clase", "persona", "dataset"}


# ── Utilidades ────────────────────────────────────────────────────────────────

def guardar(fig, nombre):
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta}")


def banner(titulo):
    print(f"\n{'='*62}")
    print(f"  {titulo}")
    print("=" * 62)


# ── 0. Carga de datos ─────────────────────────────────────────────────────────

banner("0. Carga de datos")

df = pd.read_csv(RUTA_FEATURES)
df = df[df["clase"].isin(VOCALES)].reset_index(drop=True)

feat_cols = [c for c in df.columns if c not in META]
X_orig    = df[feat_cols].values.astype(np.float64)
y         = df["clase"].values

le    = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"  Features cargadas : {len(feat_cols)}")
print(f"  Muestras          : {len(X_orig)}")
print(f"  Distribución      : {dict(pd.Series(y).value_counts().sort_index())}")


# ── 1. Correlación de Pearson ─────────────────────────────────────────────────

banner("1. Correlación de Pearson  (umbral |r| > 0.9)")

df_feat = pd.DataFrame(X_orig, columns=feat_cols)
corr_abs = df_feat.corr().abs()

# Greedy: recorrer features en orden; descartar si correlaciona > umbral
# con alguna ya retenida. Las features con mayor varianza son preferidas.
varianzas = df_feat.var().sort_values(ascending=False)
orden_por_var = varianzas.index.tolist()

retenidas_corr: list[str] = []
for col in orden_por_var:
    if retenidas_corr:
        if corr_abs.loc[col, retenidas_corr].max() > UMBRAL_CORR:
            continue
    retenidas_corr.append(col)

n_elim = len(feat_cols) - len(retenidas_corr)
print(f"  Eliminadas por correlación : {n_elim}")
print(f"  Features retenidas         : {len(retenidas_corr)}")

X_corr = df[retenidas_corr].values.astype(np.float64)

# Heatmap de correlación (hasta 60 features para legibilidad)
n_hm = min(60, len(retenidas_corr))
muestra_cols = retenidas_corr[:n_hm]
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(
    pd.DataFrame(X_corr[:, :n_hm], columns=muestra_cols).corr(),
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    xticklabels=False, yticklabels=False, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title(
    f"Correlación de Pearson — primeras {n_hm} features tras filtrado\n"
    f"(total retenidas: {len(retenidas_corr)})", fontsize=12
)
guardar(fig, "1_correlacion_pearson.png")


# ── 2. ANOVA + FDR B-H + Bonferroni ──────────────────────────────────────────

banner("2. ANOVA + correcciones FDR (B-H y Bonferroni)")

grupos_por_vocal = [X_corr[y == v] for v in VOCALES]

f_vals = np.zeros(len(retenidas_corr))
p_vals = np.ones(len(retenidas_corr))
for i in range(len(retenidas_corr)):
    cols = [g[:, i] for g in grupos_por_vocal]
    f, p = stats.f_oneway(*cols)
    f_vals[i] = f if np.isfinite(f) else 0.0
    p_vals[i] = p if np.isfinite(p) else 1.0

_, p_bh, _, _   = multipletests(p_vals, alpha=ALPHA, method="fdr_bh")
_, p_bonf, _, _ = multipletests(p_vals, alpha=ALPHA, method="bonferroni")

set_bh   = {retenidas_corr[i] for i, sig in enumerate(p_bh   < ALPHA) if sig}
set_bonf = {retenidas_corr[i] for i, sig in enumerate(p_bonf < ALPHA) if sig}

print(f"  Significativas B-H        : {len(set_bh)}")
print(f"  Significativas Bonferroni : {len(set_bonf)}")
print(f"  Ambas correcciones        : {len(set_bh & set_bonf)}")

# Distribución de p-valores
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(p_vals, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
axes[0].axvline(ALPHA, color="red", ls="--", lw=1.5, label=f"α = {ALPHA}")
axes[0].set_xlabel("p-valor crudo (ANOVA)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Distribución de p-valores crudos")
axes[0].legend()

axes[1].hist(p_bh,   bins=50, color="#2ecc71", edgecolor="white", alpha=0.7, label="B-H")
axes[1].hist(p_bonf, bins=50, color="#e74c3c", edgecolor="white", alpha=0.5, label="Bonferroni")
axes[1].axvline(ALPHA, color="black", ls="--", lw=1.5, label=f"α = {ALPHA}")
axes[1].set_xlabel("p-valor corregido")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("p-valores corregidos (B-H vs Bonferroni)")
axes[1].legend()

plt.suptitle("ANOVA — Distribución de p-valores", fontsize=13, y=1.01)
plt.tight_layout()
guardar(fig, "2_pvalores_anova.png")

# Top 20 features por F-estadístico
top20_idx = np.argsort(f_vals)[::-1][:20]
fig, ax = plt.subplots(figsize=(11, 6))
labels_top20 = [retenidas_corr[i] for i in top20_idx[::-1]]
values_top20 = [f_vals[i] for i in top20_idx[::-1]]
colores_top20 = [
    "#e74c3c" if retenidas_corr[i] in set_bh else "#bdc3c7"
    for i in top20_idx[::-1]
]
ax.barh(labels_top20, values_top20, color=colores_top20)
leyenda_anova = [
    mpatches.Patch(color="#e74c3c", label="Significativa (B-H)"),
    mpatches.Patch(color="#bdc3c7", label="No significativa"),
]
ax.legend(handles=leyenda_anova)
ax.set_xlabel("F-estadístico")
ax.set_title("Top 20 features por F-estadístico (ANOVA)")
guardar(fig, "2_top20_f_estadistico.png")


# ── 3. PCA ────────────────────────────────────────────────────────────────────

banner("3. PCA")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_corr)

n_comp = min(20, X_corr.shape[1], X_corr.shape[0] - 1)
pca_full = PCA(n_components=n_comp, random_state=42)
pca_full.fit(X_scaled)

var_ind  = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_ind)
n_95     = int(np.searchsorted(var_acum, 0.95)) + 1

print(f"  Componentes para ≥95% varianza : {n_95}")
print(f"  Varianza PC1={var_ind[0]*100:.2f}%  PC2={var_ind[1]*100:.2f}%")

# Gráficas de varianza explicada
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ejes_x = range(1, len(var_ind) + 1)
axes[0].bar(ejes_x, var_ind * 100, color="#3498db", alpha=0.85)
axes[0].set_xlabel("Componente principal")
axes[0].set_ylabel("Varianza explicada (%)")
axes[0].set_title("Varianza por componente")

axes[1].plot(ejes_x, var_acum * 100, "o-", color="#e74c3c", ms=5)
axes[1].axhline(95, ls="--", color="gray", lw=1.5, label="95%")
axes[1].axvline(n_95, ls="--", color="gray", lw=1.5, label=f"PC {n_95}")
axes[1].set_xlabel("Número de componentes")
axes[1].set_ylabel("Varianza acumulada (%)")
axes[1].set_title("Varianza acumulada")
axes[1].legend()

plt.suptitle("PCA — Varianza explicada", fontsize=13, y=1.01)
plt.tight_layout()
guardar(fig, "3_pca_varianza.png")

# Scatter 2D (PC1 vs PC2) coloreado por vocal
pca2     = PCA(n_components=2, random_state=42)
X_pca2   = pca2.fit_transform(X_scaled)
var2     = pca2.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))
for vocal in VOCALES:
    mask = y == vocal
    ax.scatter(
        X_pca2[mask, 0], X_pca2[mask, 1],
        color=PALETA[vocal], label=vocal,
        alpha=0.45, s=22, edgecolors="none"
    )
ax.set_xlabel(f"PC1  ({var2[0]*100:.1f}% varianza)", fontsize=11)
ax.set_ylabel(f"PC2  ({var2[1]*100:.1f}% varianza)", fontsize=11)
ax.set_title("PCA — Scatter 2D coloreado por vocal", fontsize=13)
ax.legend(title="Vocal", markerscale=2.5, fontsize=11)
guardar(fig, "3_pca_scatter2d.png")


# ── 4. SelectKBest — F-score y Mutual Information ────────────────────────────

banner("4. SelectKBest  (F-score y Mutual Information)")

K_actual = min(K_BEST, max(2, len(retenidas_corr) // 2))
print(f"  K efectivo : {K_actual}  (de {len(retenidas_corr)} features post-correlación)")

f_scores_arr, _  = f_classif(X_corr, y_enc)
mi_scores_arr    = mutual_info_classif(X_corr, y_enc, random_state=42)

f_scores_arr  = np.nan_to_num(f_scores_arr)
mi_scores_arr = np.nan_to_num(mi_scores_arr)

idx_top_f  = set(np.argsort(f_scores_arr)[::-1][:K_actual])
idx_top_mi = set(np.argsort(mi_scores_arr)[::-1][:K_actual])

set_fscore = {retenidas_corr[i] for i in idx_top_f}
set_mi     = {retenidas_corr[i] for i in idx_top_mi}

print(f"  Features SelectKBest F-score : {len(set_fscore)}")
print(f"  Features SelectKBest MI      : {len(set_mi)}")
print(f"  Intersección F ∩ MI          : {len(set_fscore & set_mi)}")

# Scatter F-score vs MI con colores por pertenencia
f_norm  = f_scores_arr  / (f_scores_arr.max()  + 1e-12)
mi_norm = mi_scores_arr / (mi_scores_arr.max() + 1e-12)

color_pts = []
for i in range(len(retenidas_corr)):
    en_f  = i in idx_top_f
    en_mi = i in idx_top_mi
    if en_f and en_mi:
        color_pts.append("#e74c3c")   # ambos
    elif en_f:
        color_pts.append("#3498db")   # solo F
    elif en_mi:
        color_pts.append("#2ecc71")   # solo MI
    else:
        color_pts.append("#bdc3c7")   # ninguno

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(f_norm, mi_norm, c=color_pts, alpha=0.6, s=18, edgecolors="none")
ley = [
    mpatches.Patch(color="#e74c3c", label=f"Ambos top-{K_actual}"),
    mpatches.Patch(color="#3498db", label=f"Solo F-score"),
    mpatches.Patch(color="#2ecc71", label=f"Solo MI"),
    mpatches.Patch(color="#bdc3c7", label="Resto"),
]
ax.legend(handles=ley, loc="lower right", fontsize=10)
ax.set_xlabel("F-score normalizado", fontsize=11)
ax.set_ylabel("Mutual Information normalizado", fontsize=11)
ax.set_title(f"F-score vs Mutual Information  (top-{K_actual} marcados)", fontsize=13)
guardar(fig, "4_fscore_vs_mi.png")


# ── 5. Sistema de votos ───────────────────────────────────────────────────────

banner("5. Sistema de votos  (umbral ≥ 2 de 4 métodos)")

metodos = {
    "ANOVA B-H"        : set_bh,
    "ANOVA Bonferroni" : set_bonf,
    "SelectKBest F"    : set_fscore,
    "SelectKBest MI"   : set_mi,
}

candidatas = set().union(*metodos.values())
votos_dict = {
    feat: sum(1 for s in metodos.values() if feat in s)
    for feat in candidatas
}

conjunto_final = sorted(f for f, v in votos_dict.items() if v >= 2)

print(f"\n  Votos por método:")
for nombre, s in metodos.items():
    print(f"    {nombre:<22} : {len(s):>4} features")
print(f"\n  Features candidatas       : {len(candidatas)}")
print(f"  Features con ≥ 2 votos    : {len(conjunto_final)}")

# Histograma de votos
conteo_v = pd.Series(votos_dict.values()).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
colores_v = ["#bdc3c7" if v < 2 else "#9b59b6" for v in conteo_v.index]
ax.bar(conteo_v.index.astype(str), conteo_v.values, color=colores_v, edgecolor="white")
ax.axvline(1.5, color="red", ls="--", lw=1.5, label="Umbral ≥ 2")
ax.set_xlabel("Número de votos recibidos")
ax.set_ylabel("Número de features")
ax.set_title("Distribución de votos por feature")
ax.legend()
guardar(fig, "5_distribucion_votos.png")

# Tabla resumen de votos
df_votos = pd.DataFrame([
    {
        "feature"          : feat,
        "votos"            : votos_dict[feat],
        "ANOVA_BH"         : feat in set_bh,
        "ANOVA_Bonferroni" : feat in set_bonf,
        "SelectKBest_F"    : feat in set_fscore,
        "SelectKBest_MI"   : feat in set_mi,
        "F_score"          : (f_scores_arr[retenidas_corr.index(feat)]
                              if feat in retenidas_corr else np.nan),
        "MI_score"         : (mi_scores_arr[retenidas_corr.index(feat)]
                              if feat in retenidas_corr else np.nan),
    }
    for feat in sorted(candidatas, key=lambda f: -votos_dict[f])
])
df_votos.to_csv(os.path.join(DIR_RESULTADOS, "tabla_votos.csv"), index=False)
print(f"  Tabla de votos guardada")


# ── 6. Box plots y violin plots del conjunto final ────────────────────────────

banner("6. Box plots y violin plots del conjunto final")

# Ordenar el conjunto final por F-score descendente
def f_de(feat):
    if feat in retenidas_corr:
        return f_scores_arr[retenidas_corr.index(feat)]
    return 0.0

conjunto_ordenado = sorted(conjunto_final, key=f_de, reverse=True)
top_viz = conjunto_ordenado[:N_VIZUALIZAR]
n_viz   = len(top_viz)
print(f"  Features a graficar : {n_viz}")

ncols = 3
nrows = int(np.ceil(n_viz / ncols))

# Box plots
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

for idx, feat in enumerate(top_viz):
    ax = axes_flat[idx]
    datos_por_vocal = [df.loc[df["clase"] == v, feat].values for v in VOCALES]
    bp = ax.boxplot(datos_por_vocal, patch_artist=True, labels=VOCALES,
                    medianprops=dict(color="black", lw=1.5))
    for patch, vocal in zip(bp["boxes"], VOCALES):
        patch.set_facecolor(PALETA[vocal])
        patch.set_alpha(0.75)
    ax.set_title(feat, fontsize=7.5)
    ax.set_ylabel("Valor", fontsize=8)
    ax.tick_params(axis="x", labelsize=9)

for idx in range(n_viz, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(f"Box plots — Top {n_viz} features del conjunto final  (por F-score)",
             fontsize=13, y=1.01)
plt.tight_layout()
guardar(fig, "6_boxplots.png")

# Violin plots
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

for idx, feat in enumerate(top_viz):
    ax = axes_flat[idx]
    datos_long = pd.DataFrame({
        "vocal": y,
        "valor": df[feat].values,
    })
    datos_long["vocal"] = pd.Categorical(datos_long["vocal"], categories=VOCALES)
    sns.violinplot(
        data=datos_long, x="vocal", y="valor",
        order=VOCALES, palette=PALETA,
        inner="box", ax=ax, linewidth=0.8,
    )
    ax.set_title(feat, fontsize=7.5)
    ax.set_xlabel("")
    ax.set_ylabel("Valor", fontsize=8)
    ax.tick_params(axis="x", labelsize=9)

for idx in range(n_viz, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(f"Violin plots — Top {n_viz} features del conjunto final  (por F-score)",
             fontsize=13, y=1.01)
plt.tight_layout()
guardar(fig, "6_violinplots.png")


# ── 7. Guardado del conjunto final ────────────────────────────────────────────

banner("7. Guardado del conjunto final")

cols_extra = [c for c in ["clase", "persona", "dataset"] if c in df.columns]
df_final   = df[conjunto_final + cols_extra].copy()

os.makedirs(os.path.dirname(RUTA_SALIDA_CSV), exist_ok=True)
df_final.to_csv(RUTA_SALIDA_CSV, index=False)

print(f"  CSV guardado en  : {RUTA_SALIDA_CSV}")
print(f"  Shape final      : {df_final.shape}  "
      f"({len(conjunto_final)} features × {len(df_final)} muestras)")
print(f"  Gráficas en      : {DIR_RESULTADOS}/")

banner("Pipeline completado")
print(f"  Features originales             : {len(feat_cols)}")
print(f"  Tras correlación Pearson        : {len(retenidas_corr)}")
print(f"  Tras ANOVA B-H                  : {len(set_bh)}")
print(f"  Tras ANOVA Bonferroni           : {len(set_bonf)}")
print(f"  SelectKBest F-score (top-{K_actual:<3})  : {len(set_fscore)}")
print(f"  SelectKBest MI      (top-{K_actual:<3})  : {len(set_mi)}")
print(f"  Conjunto final (≥ 2 votos)      : {len(conjunto_final)}")
