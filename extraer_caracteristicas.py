import os
import numpy as np
import pandas as pd


def distancia_3d(punto1, punto2):
    return np.linalg.norm(punto2 - punto1)


def sacar_angulo_3_puntos(p1, p2, p3):
    """
    Calcula el ángulo en p2 formado por los segmentos p1-p2 y p3-p2
    """
    v1 = p1 - p2
    v2 = p3 - p2

    norma_v1 = np.linalg.norm(v1)
    norma_v2 = np.linalg.norm(v2)

    if norma_v1 == 0 or norma_v2 == 0:
        return 0.0

    cos_theta = np.dot(v1, v2) / (norma_v1 * norma_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angulo = np.arccos(cos_theta)
    return np.degrees(angulo)


def organizar_landmarks_de_fila(fila):
    puntos = []

    for i in range(21):
        x = fila[f"x_{i}"]
        y = fila[f"y_{i}"]
        z = fila[f"z_{i}"]
        puntos.append([x, y, z])

    return np.array(puntos, dtype=np.float32)


def normalizar_landmarks(puntos):
    """
    1. Traslada todo respecto a la muñeca (landmark 0)
    2. Escala usando la distancia muñeca - base dedo medio (landmark 9)
    """
    punto_muneca = puntos[0]
    puntos_centrados = puntos - punto_muneca

    distancia_referencia = np.linalg.norm(puntos[9] - puntos[0])

    if distancia_referencia == 0:
        return puntos_centrados

    puntos_normalizados = puntos_centrados / distancia_referencia
    return puntos_normalizados


def sacar_features_de_una_muestra(puntos_norm):
    features = {}

    # -----------------------------
    # 1. Coordenadas normalizadas
    # -----------------------------
    for i in range(21):
        features[f"xn_{i}"] = puntos_norm[i][0]
        features[f"yn_{i}"] = puntos_norm[i][1]
        features[f"zn_{i}"] = puntos_norm[i][2]

    # -----------------------------
    # 2. Distancias importantes
    # -----------------------------
    puntas_dedos = [4, 8, 12, 16, 20]
    nombres_puntas = ["pulgar", "indice", "medio", "anular", "menique"]

    # distancia de la muñeca a cada punta
    for nombre, idx in zip(nombres_puntas, puntas_dedos):
        features[f"dist_muneca_{nombre}"] = distancia_3d(puntos_norm[0], puntos_norm[idx])

    # distancias entre puntas de dedos vecinas
    pares_puntas = [
        ("pulgar_indice", 4, 8),
        ("indice_medio", 8, 12),
        ("medio_anular", 12, 16),
        ("anular_menique", 16, 20),
        ("pulgar_menique", 4, 20)
    ]

    for nombre, i, j in pares_puntas:
        features[f"dist_{nombre}"] = distancia_3d(puntos_norm[i], puntos_norm[j])

    # -----------------------------
    # 3. Ángulos de flexión por dedo
    # -----------------------------
    # índice: 5-6-8
    features["angulo_indice"] = sacar_angulo_3_puntos(
        puntos_norm[5], puntos_norm[6], puntos_norm[8]
    )

    # medio: 9-10-12
    features["angulo_medio"] = sacar_angulo_3_puntos(
        puntos_norm[9], puntos_norm[10], puntos_norm[12]
    )

    # anular: 13-14-16
    features["angulo_anular"] = sacar_angulo_3_puntos(
        puntos_norm[13], puntos_norm[14], puntos_norm[16]
    )

    # meñique: 17-18-20
    features["angulo_menique"] = sacar_angulo_3_puntos(
        puntos_norm[17], puntos_norm[18], puntos_norm[20]
    )

    # pulgar: 1-2-4
    features["angulo_pulgar"] = sacar_angulo_3_puntos(
        puntos_norm[1], puntos_norm[2], puntos_norm[4]
    )

    # -----------------------------
    # 4. Apertura general de la mano
    # -----------------------------
    features["apertura_mano"] = distancia_3d(puntos_norm[4], puntos_norm[20])

    return features


def sacar_features(df_landmarks):
    data_features = []

    for _, fila in df_landmarks.iterrows():
        puntos = organizar_landmarks_de_fila(fila)
        puntos_norm = normalizar_landmarks(puntos)

        features = sacar_features_de_una_muestra(puntos_norm)

        features["clase"] = fila["clase"]
        features["persona"] = fila["persona"]

        if "dataset" in df_landmarks.columns:
            features["dataset"] = fila["dataset"]

        data_features.append(features)

    df_features = pd.DataFrame(data_features)
    return df_features


if __name__ == "__main__":
    ruta_entrada = "DataSet/processed/landmarks.csv"
    ruta_salida = "DataSet/processed/features.csv"

    df_landmarks = pd.read_csv(ruta_entrada)
    df_features = sacar_features(df_landmarks)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_features.to_csv(ruta_salida, index=False)

    print("Features guardadas correctamente")
    print("Shape del dataframe:", df_features.shape)
    print(df_features.head())