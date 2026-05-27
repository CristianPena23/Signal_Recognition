import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def sacar_columnas_features(df):
    columnas_quitar = ["clase", "persona", "dataset"]
    columnas_features = [col for col in df.columns if col not in columnas_quitar]
    return columnas_features


def cargar_datos(ruta_features):
    df = pd.read_csv(ruta_features)
    columnas_features = sacar_columnas_features(df)

    x = df[columnas_features].values
    y = df["clase"].values

    return df, x, y, columnas_features


def preparar_etiquetas(y):
    label_encoder = LabelEncoder()
    y_codificado = label_encoder.fit_transform(y)
    return y_codificado, label_encoder


def partir_datos(x, y, test_size=0.2, random_state=42):
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def escalar_datos(x_train, x_test):
    scaler = StandardScaler()
    x_train_escalado = scaler.fit_transform(x_train)
    x_test_escalado = scaler.transform(x_test)
    return x_train_escalado, x_test_escalado, scaler


def entrenar_knn(x_train, y_train):
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(x_train, y_train)
    return modelo


def entrenar_svm(x_train, y_train):
    modelo = SVC(kernel="rbf", C=10, gamma="scale")
    modelo.fit(x_train, y_train)
    return modelo


def entrenar_random_forest(x_train, y_train):
    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    modelo.fit(x_train, y_train)
    return modelo


def evaluar_modelo(nombre_modelo, modelo, x_test, y_test, label_encoder, carpeta_salida):
    y_pred = modelo.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    reporte = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    matriz = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Resultados para: {nombre_modelo}")
    print(f"{'='*50}")
    print("Accuracy:", acc)
    print("\nReporte de clasificación:")
    print(reporte)
    print("Matriz de confusión:")
    print(matriz)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title(f"Matriz de confusión - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f"matriz_confusion_{nombre_modelo}.png"))
    plt.show()

    return acc, reporte, matriz


def guardar_modelo_y_objetos(modelo, scaler, label_encoder, carpeta_modelos, nombre_base, sufijo=""):
    os.makedirs(carpeta_modelos, exist_ok=True)

    ruta_modelo = os.path.join(carpeta_modelos, f"{nombre_base}.pkl")
    ruta_scaler = os.path.join(carpeta_modelos, f"scaler{sufijo}.pkl")
    ruta_label_encoder = os.path.join(carpeta_modelos, f"label_encoder{sufijo}.pkl")

    joblib.dump(modelo, ruta_modelo)
    if scaler is not None:
        joblib.dump(scaler, ruta_scaler)
        print("Scaler guardado en:", ruta_scaler)
    joblib.dump(label_encoder, ruta_label_encoder)

    print("Modelo guardado en:", ruta_modelo)
    print("LabelEncoder guardado en:", ruta_label_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelos de clasificación de señas.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--vocales",
        action="store_true",
        help="Usa features_vocales.csv y guarda modelos con sufijo _vocales.",
    )
    grupo.add_argument(
        "--seleccionadas",
        action="store_true",
        help="Usa features_vocales_seleccionadas.csv y guarda modelos con sufijo _seleccionadas.",
    )
    grupo.add_argument(
        "--top100",
        action="store_true",
        help="Usa features_vocales_top100.csv y entrena solo Random Forest con métricas completas.",
    )
    args = parser.parse_args()

    if args.top100:
        ruta_features      = "DataSet/processed/features_vocales_top100.csv"
        carpeta_resultados = "results/modelos_top100"
        sufijo_modelos     = "_top100"
    elif args.seleccionadas:
        ruta_features      = "DataSet/processed/features_vocales_seleccionadas.csv"
        carpeta_resultados = "results/modelos_seleccionadas"
        sufijo_modelos     = "_seleccionadas"
    elif args.vocales:
        ruta_features      = "DataSet/processed/features_vocales.csv"
        carpeta_resultados = "results/modelos_vocales"
        sufijo_modelos     = "_vocales"
    else:
        ruta_features      = "DataSet/processed/features.csv"
        carpeta_resultados = "results/modelos"
        sufijo_modelos     = ""

    carpeta_modelos = "models"

    os.makedirs(carpeta_resultados, exist_ok=True)
    os.makedirs(carpeta_modelos, exist_ok=True)

    print(f"Usando: {ruta_features}")
    df, x, y, columnas_features = cargar_datos(ruta_features)
    print(f"Clases: {sorted(set(y))}  |  Muestras: {len(y)}")

    y_codificado, label_encoder = preparar_etiquetas(y)

    x_train, x_test, y_train, y_test = partir_datos(x, y_codificado)

    x_train_escalado, x_test_escalado, scaler = escalar_datos(x_train, x_test)

    if args.top100:
        print("\nEntrenando Random Forest (modo --top100)...")
        modelo_rf = entrenar_random_forest(x_train, y_train)
        acc_rf, reporte_rf, matriz_rf = evaluar_modelo(
            "RandomForest_top100", modelo_rf, x_test, y_test, label_encoder, carpeta_resultados
        )
        ruta_metricas = os.path.join(carpeta_resultados, "metricas_randomforest_top100.txt")
        with open(ruta_metricas, "w", encoding="utf-8") as fh:
            fh.write(f"Accuracy global: {acc_rf:.4f}\n\n")
            fh.write("Reporte de clasificación (precisión, recall, F1 por vocal y global):\n")
            fh.write(reporte_rf)
            fh.write("\nMatriz de confusión:\n")
            fh.write(str(matriz_rf))
        print(f"Métricas guardadas en: {ruta_metricas}")
        guardar_modelo_y_objetos(
            modelo_rf,
            None,
            label_encoder,
            carpeta_modelos,
            nombre_base="modelo_randomforest_top100",
            sufijo="_top100",
        )
    elif args.seleccionadas:
        # Solo Random Forest; no necesita escalado
        print("\nEntrenando Random Forest (modo --seleccionadas)...")
        modelo_rf = entrenar_random_forest(x_train, y_train)
        evaluar_modelo(
            "RandomForest", modelo_rf, x_test, y_test, label_encoder, carpeta_resultados
        )
        guardar_modelo_y_objetos(
            modelo_rf,
            None,
            label_encoder,
            carpeta_modelos,
            nombre_base="modelo_randomforest_seleccionadas",
            sufijo="_seleccionadas",
        )
    else:
        resultados = {}

        # KNN
        modelo_knn = entrenar_knn(x_train_escalado, y_train)
        acc_knn, _, _ = evaluar_modelo(
            "KNN", modelo_knn, x_test_escalado, y_test, label_encoder, carpeta_resultados
        )
        resultados["KNN"] = {"modelo": modelo_knn, "accuracy": acc_knn, "escalado": True}

        # SVM
        modelo_svm = entrenar_svm(x_train_escalado, y_train)
        acc_svm, _, _ = evaluar_modelo(
            "SVM", modelo_svm, x_test_escalado, y_test, label_encoder, carpeta_resultados
        )
        resultados["SVM"] = {"modelo": modelo_svm, "accuracy": acc_svm, "escalado": True}

        # Random Forest
        modelo_rf = entrenar_random_forest(x_train, y_train)
        acc_rf, _, _ = evaluar_modelo(
            "RandomForest", modelo_rf, x_test, y_test, label_encoder, carpeta_resultados
        )
        resultados["RandomForest"] = {"modelo": modelo_rf, "accuracy": acc_rf, "escalado": False}

        print("\nResumen de accuracies:")
        for nombre, info in resultados.items():
            print(f"  {nombre}: {info['accuracy']:.4f}")

        mejor_nombre = max(resultados, key=lambda k: resultados[k]["accuracy"])
        mejor_modelo = resultados[mejor_nombre]["modelo"]
        usa_scaler   = resultados[mejor_nombre]["escalado"]

        print(f"\nEl mejor modelo fue: {mejor_nombre}")

        guardar_modelo_y_objetos(
            mejor_modelo,
            scaler if usa_scaler else None,
            label_encoder,
            carpeta_modelos,
            nombre_base=f"modelo_{mejor_nombre.lower()}{sufijo_modelos}",
            sufijo=sufijo_modelos,
        )