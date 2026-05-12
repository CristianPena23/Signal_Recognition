import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import urllib.request
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")


def procesar_imagen(ruta_img, landmarker):
    img = cv2.imread(ruta_img)

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    resultado = landmarker.detect(mp_image)

    if not resultado.hand_landmarks:
        return None

    mano = resultado.hand_landmarks[0]

    puntos = [[lm.x, lm.y, lm.z] for lm in mano]

    return np.array(puntos)  # (21, 3)


def extraer_landmarks(df_metadata):
    _descargar_modelo()

    data = []

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        for _, row in df_metadata.iterrows():
            ruta = row["ruta"]

            puntos = procesar_imagen(ruta, landmarker)

            if puntos is None:
                continue

            fila = {}

            for j in range(21):
                fila[f"x_{j}"] = puntos[j][0]
                fila[f"y_{j}"] = puntos[j][1]
                fila[f"z_{j}"] = puntos[j][2]

            fila["clase"] = row["clase"]
            fila["persona"] = row["persona"]

            data.append(fila)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df_meta = pd.read_csv("DataSet/interim/metadata.csv")

    df_landmarks = extraer_landmarks(df_meta)
    df_landmarks.to_csv("DataSet/processed/landmarks.csv", index=False)

    print("Landmarks extraídos!")
    print(df_landmarks.head())
