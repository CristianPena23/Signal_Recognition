import os
import argparse
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog as skimage_hog

VOCALES = ["A", "E", "I", "O", "U"]


# Conexiones del esqueleto (necesarias para renderizar la imagen del esqueleto)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
_CANVAS = 64  # tamaño en píxeles del canvas para Hu/HOG


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


# ---------------------------------------------------------------------------
# Helpers para Hu Moments y HOG
# ---------------------------------------------------------------------------

def _landmarks_a_imagen(puntos_norm, tam=_CANVAS):
    """
    Proyecta los landmarks 2D (x, y) normalizados sobre un canvas en escala de grises.
    Dibuja las conexiones del esqueleto y un punto en cada articulación.
    """
    img = np.zeros((tam, tam), dtype=np.uint8)

    xy = puntos_norm[:, :2].copy()

    margen = 6
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    rango = xy_max - xy_min
    rango[rango == 0] = 1.0

    xy_scaled = (xy - xy_min) / rango * (tam - 2 * margen) + margen
    pts = np.clip(np.round(xy_scaled).astype(int), 0, tam - 1)

    for a, b in _HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), 255, 1)
    for pt in pts:
        cv2.circle(img, tuple(pt), 2, 255, -1)

    return img


def _hu_moments(img):
    """
    7 momentos de Hu log-escalados para mejorar el rango numérico.
    Signo conservado para distinguir positivos de negativos.
    """
    m = cv2.moments(img)
    hu = cv2.HuMoments(m).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def _hog_features(img):
    """
    HOG sobre el canvas de 64x64.
    Parámetros: 16x16 px/celda, bloques 2x2 celdas, 9 orientaciones → 324 features.
    """
    return skimage_hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )


# ---------------------------------------------------------------------------

def sacar_features_de_una_muestra(puntos_norm):
    features = {}

    # -----------------------------
    # 1. Coordenadas normalizadas (63)
    # -----------------------------
    for i in range(21):
        features[f"xn_{i}"] = puntos_norm[i][0]
        features[f"yn_{i}"] = puntos_norm[i][1]
        features[f"zn_{i}"] = puntos_norm[i][2]

    # -----------------------------
    # 2. Distancias importantes (10)
    # -----------------------------
    puntas_dedos = [4, 8, 12, 16, 20]
    nombres_puntas = ["pulgar", "indice", "medio", "anular", "menique"]

    for nombre, idx in zip(nombres_puntas, puntas_dedos):
        features[f"dist_muneca_{nombre}"] = distancia_3d(puntos_norm[0], puntos_norm[idx])

    pares_puntas = [
        ("pulgar_indice", 4, 8),
        ("indice_medio", 8, 12),
        ("medio_anular", 12, 16),
        ("anular_menique", 16, 20),
        ("pulgar_menique", 4, 20),
    ]
    for nombre, i, j in pares_puntas:
        features[f"dist_{nombre}"] = distancia_3d(puntos_norm[i], puntos_norm[j])

    # -----------------------------
    # 3. Ángulos de flexión por dedo (5)
    # -----------------------------
    features["angulo_indice"] = sacar_angulo_3_puntos(
        puntos_norm[5], puntos_norm[6], puntos_norm[8]
    )
    features["angulo_medio"] = sacar_angulo_3_puntos(
        puntos_norm[9], puntos_norm[10], puntos_norm[12]
    )
    features["angulo_anular"] = sacar_angulo_3_puntos(
        puntos_norm[13], puntos_norm[14], puntos_norm[16]
    )
    features["angulo_menique"] = sacar_angulo_3_puntos(
        puntos_norm[17], puntos_norm[18], puntos_norm[20]
    )
    features["angulo_pulgar"] = sacar_angulo_3_puntos(
        puntos_norm[1], puntos_norm[2], puntos_norm[4]
    )

    # -----------------------------
    # 4. Apertura general de la mano (1)
    # -----------------------------
    features["apertura_mano"] = distancia_3d(puntos_norm[4], puntos_norm[20])

    # -----------------------------
    # 5. Hu Moments sobre imagen del esqueleto (7)
    # -----------------------------
    img_esqueleto = _landmarks_a_imagen(puntos_norm)
    for i, v in enumerate(_hu_moments(img_esqueleto)):
        features[f"hu_{i}"] = float(v)

    # -----------------------------
    # 6. HOG sobre imagen del esqueleto (324)
    # -----------------------------
    for i, v in enumerate(_hog_features(img_esqueleto)):
        features[f"hog_{i}"] = float(v)

    return features


def sacar_features(df_landmarks):
    data_features = []
    total = len(df_landmarks)

    for n, (_, fila) in enumerate(df_landmarks.iterrows(), 1):
        puntos = organizar_landmarks_de_fila(fila)
        puntos_norm = normalizar_landmarks(puntos)

        features = sacar_features_de_una_muestra(puntos_norm)

        features["clase"] = fila["clase"]
        features["persona"] = fila["persona"]

        if "dataset" in df_landmarks.columns:
            features["dataset"] = fila["dataset"]

        data_features.append(features)

        if n % 500 == 0 or n == total:
            print(f"  {n}/{total} muestras procesadas...")

    df_features = pd.DataFrame(data_features)
    return df_features


def _contar_features(df):
    meta = {"clase", "persona", "dataset"}
    return df.shape[1] - sum(c in df.columns for c in meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae features de landmarks de mano."
    )
    parser.add_argument(
        "--vocales",
        action="store_true",
        help="Filtra solo las clases A, E, I, O, U y guarda archivos con sufijo _vocales.",
    )
    args = parser.parse_args()

    DIR_PROC = "DataSet/processed"
    os.makedirs(DIR_PROC, exist_ok=True)

    ruta_lm_completo = f"{DIR_PROC}/landmarks.csv"

    print("Cargando landmarks.csv...")
    df_lm = pd.read_csv(ruta_lm_completo)
    print(f"  Total muestras en landmarks.csv: {len(df_lm)}")

    if args.vocales:
        # ── Modo vocales ──────────────────────────────────────────────────────
        df_lm_filtrado = df_lm[df_lm["clase"].isin(VOCALES)].reset_index(drop=True)

        clases_encontradas = sorted(df_lm_filtrado["clase"].unique())
        clases_faltantes   = [c for c in VOCALES if c not in clases_encontradas]
        print(f"\nClases encontradas: {clases_encontradas}")
        if clases_faltantes:
            print(f"[AVISO] Clases no presentes en el dataset: {clases_faltantes}")

        print(f"Muestras tras filtrar: {len(df_lm_filtrado)}")
        print(df_lm_filtrado["clase"].value_counts().sort_index().to_string())

        ruta_lm_salida = f"{DIR_PROC}/landmarks_vocales.csv"
        df_lm_filtrado.to_csv(ruta_lm_salida, index=False)
        print(f"\nLandmarks filtrados guardados en: {ruta_lm_salida}")

        print(f"\nExtrayendo features para {len(df_lm_filtrado)} muestras...")
        df_ft = sacar_features(df_lm_filtrado)

        ruta_ft_salida = f"{DIR_PROC}/features_vocales.csv"
        df_ft.to_csv(ruta_ft_salida, index=False)
        print(f"\nFeatures guardadas en: {ruta_ft_salida}")
        print(f"Shape: {df_ft.shape}  ({_contar_features(df_ft)} features por muestra)")

    else:
        # ── Modo completo (comportamiento original) ───────────────────────────
        print(f"\nExtrayendo features para {len(df_lm)} muestras...")
        df_ft = sacar_features(df_lm)

        ruta_ft_salida = f"{DIR_PROC}/features.csv"
        df_ft.to_csv(ruta_ft_salida, index=False)
        print(f"\nFeatures guardadas en: {ruta_ft_salida}")
        print(f"Shape: {df_ft.shape}  ({_contar_features(df_ft)} features por muestra)")

    print(df_ft.head())
