import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def sacar_columnas_numericas(df):
    columnas_quitar = ["clase", "persona", "dataset"]
    columnas_features = [col for col in df.columns if col not in columnas_quitar]
    return columnas_features


def grafica_cantidad_por_clase(df, carpeta_salida):
    conteo_clases = df["clase"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.bar(conteo_clases.index.astype(str), conteo_clases.values)
    plt.title("Cantidad de muestras por clase")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "cantidad_por_clase.png"))
    plt.show()


def grafica_cantidad_por_persona(df, carpeta_salida):
    conteo_personas = df["persona"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.bar(conteo_personas.index.astype(str), conteo_personas.values)
    plt.title("Cantidad de muestras por persona")
    plt.xlabel("Persona")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "cantidad_por_persona.png"))
    plt.show()


def grafica_correlacion(df, columnas_features, carpeta_salida):
    matriz_corr = df[columnas_features].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(matriz_corr, aspect="auto")
    plt.colorbar()
    plt.title("Matriz de correlación de features")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "matriz_correlacion.png"))
    plt.show()


def grafica_pca_2d(df, columnas_features, carpeta_salida):
    x = df[columnas_features].values
    y = df["clase"].values

    escalador = StandardScaler()
    x_escalado = escalador.fit_transform(x)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_escalado)

    plt.figure(figsize=(10, 6))

    clases_unicas = sorted(df["clase"].unique())

    for clase in clases_unicas:
        mascara = y == clase
        plt.scatter(
            x_pca[mascara, 0],
            x_pca[mascara, 1],
            label=f"Clase {clase}",
            alpha=0.7
        )

    plt.title("Visualización PCA en 2D")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "pca_2d.png"))
    plt.show()

    print("Varianza explicada por PCA:")
    print("PC1:", pca.explained_variance_ratio_[0])
    print("PC2:", pca.explained_variance_ratio_[1])
    print("Total:", np.sum(pca.explained_variance_ratio_))


def boxplots_features_importantes(df, carpeta_salida):
    features_mostrar = [
        "dist_muneca_pulgar",
        "dist_muneca_indice",
        "dist_pulgar_indice",
        "angulo_pulgar",
        "angulo_indice",
        "apertura_mano"
    ]

    features_existentes = [f for f in features_mostrar if f in df.columns]

    for feature in features_existentes:
        plt.figure(figsize=(10, 5))
        df.boxplot(column=feature, by="clase")
        plt.title(f"Boxplot de {feature} por clase")
        plt.suptitle("")
        plt.xlabel("Clase")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_salida, f"boxplot_{feature}.png"))
        plt.show()


def resumen_numerico(df, columnas_features):
    print("\nResumen estadístico general:")
    print(df[columnas_features].describe())

    print("\nCantidad de muestras por clase:")
    print(df["clase"].value_counts().sort_index())

    print("\nCantidad de muestras por persona:")
    print(df["persona"].value_counts().sort_index())

    if "dataset" in df.columns:
        print("\nCantidad de muestras por dataset:")
        print(df["dataset"].value_counts().sort_index())


if __name__ == "__main__":
    ruta_features = "DataSet/processed/features.csv"
    carpeta_salida = "results/graficas"

    os.makedirs(carpeta_salida, exist_ok=True)

    df = pd.read_csv(ruta_features)
    columnas_features = sacar_columnas_numericas(df)

    resumen_numerico(df, columnas_features)
    grafica_cantidad_por_clase(df, carpeta_salida)
    grafica_cantidad_por_persona(df, carpeta_salida)
    grafica_correlacion(df, columnas_features, carpeta_salida)
    grafica_pca_2d(df, columnas_features, carpeta_salida)
    boxplots_features_importantes(df, carpeta_salida)

    print("\nListo. Las gráficas quedaron guardadas en:", carpeta_salida)