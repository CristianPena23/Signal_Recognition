import os
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


def guardar_modelo_y_objetos(modelo, scaler, label_encoder, carpeta_modelos, nombre_base):
    os.makedirs(carpeta_modelos, exist_ok=True)

    ruta_modelo = os.path.join(carpeta_modelos, f"{nombre_base}.pkl")
    ruta_scaler = os.path.join(carpeta_modelos, "scaler.pkl")
    ruta_label_encoder = os.path.join(carpeta_modelos, "label_encoder.pkl")

    joblib.dump(modelo, ruta_modelo)
    joblib.dump(scaler, ruta_scaler)
    joblib.dump(label_encoder, ruta_label_encoder)

    print("\nModelo guardado en:", ruta_modelo)
    print("Scaler guardado en:", ruta_scaler)
    print("LabelEncoder guardado en:", ruta_label_encoder)


if __name__ == "__main__":
    ruta_features = "DataSet/processed/features.csv"
    carpeta_resultados = "results/modelos"
    carpeta_modelos = "models"

    os.makedirs(carpeta_resultados, exist_ok=True)
    os.makedirs(carpeta_modelos, exist_ok=True)

    df, x, y, columnas_features = cargar_datos(ruta_features)
    y_codificado, label_encoder = preparar_etiquetas(y)

    x_train, x_test, y_train, y_test = partir_datos(x, y_codificado)

    x_train_escalado, x_test_escalado, scaler = escalar_datos(x_train, x_test)

    resultados = {}

    # KNN
    modelo_knn = entrenar_knn(x_train_escalado, y_train)
    acc_knn, rep_knn, mat_knn = evaluar_modelo(
        "KNN", modelo_knn, x_test_escalado, y_test, label_encoder, carpeta_resultados
    )
    resultados["KNN"] = {
        "modelo": modelo_knn,
        "accuracy": acc_knn
    }

    # SVM
    modelo_svm = entrenar_svm(x_train_escalado, y_train)
    acc_svm, rep_svm, mat_svm = evaluar_modelo(
        "SVM", modelo_svm, x_test_escalado, y_test, label_encoder, carpeta_resultados
    )
    resultados["SVM"] = {
        "modelo": modelo_svm,
        "accuracy": acc_svm
    }

    # Random Forest
    modelo_rf = entrenar_random_forest(x_train, y_train)
    acc_rf, rep_rf, mat_rf = evaluar_modelo(
        "RandomForest", modelo_rf, x_test, y_test, label_encoder, carpeta_resultados
    )
    resultados["RandomForest"] = {
        "modelo": modelo_rf,
        "accuracy": acc_rf
    }

    print("\nResumen de accuracies:")
    for nombre, info in resultados.items():
        print(f"{nombre}: {info['accuracy']:.4f}")

    mejor_nombre = max(resultados, key=lambda k: resultados[k]["accuracy"])
    mejor_modelo = resultados[mejor_nombre]["modelo"]

    print(f"\nEl mejor modelo fue: {mejor_nombre}")

    # guardado según el mejor modelo
    if mejor_nombre in ["KNN", "SVM"]:
        guardar_modelo_y_objetos(
            mejor_modelo,
            scaler,
            label_encoder,
            carpeta_modelos,
            f"modelo_{mejor_nombre.lower()}"
        )
    else:
        guardar_modelo_y_objetos(
            mejor_modelo,
            None,
            label_encoder,
            carpeta_modelos,
            f"modelo_{mejor_nombre.lower()}"
        )