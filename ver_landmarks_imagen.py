import cv2
import mediapipe as mp
import pandas as pd
import urllib.request
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")


def dibujar_landmarks_en_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)

    if img is None:
        print("No se pudo cargar la imagen:", ruta_imagen)
        return

    _descargar_modelo()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp_image)

        if not resultado.hand_landmarks:
            print("No se detectó mano en la imagen")
            cv2.imshow("Imagen sin landmarks", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        img_con_landmarks = img.copy()
        h, w = img.shape[:2]

        for mano_landmarks in resultado.hand_landmarks:
            puntos = [(int(lm.x * w), int(lm.y * h)) for lm in mano_landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img_con_landmarks, puntos[a], puntos[b], (0, 200, 0), 2)
            for x, y in puntos:
                cv2.circle(img_con_landmarks, (x, y), 5, (255, 0, 0), -1)

        cv2.imshow("Imagen con landmarks", img_con_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def mostrar_imagen_aleatoria_desde_metadata(ruta_metadata):
    df = pd.read_csv(ruta_metadata)

    if len(df) == 0:
        print("El metadata.csv está vacío")
        return

    fila = df.sample(1).iloc[0]
    ruta_imagen = fila["ruta"]

    print("Imagen seleccionada:")
    print("Ruta:", ruta_imagen)
    print("Clase:", fila["clase"])
    if "persona" in df.columns:
        print("Persona:", fila["persona"])
    if "dataset" in df.columns:
        print("Dataset:", fila["dataset"])

    dibujar_landmarks_en_imagen(ruta_imagen)


if __name__ == "__main__":
    ruta_metadata = "DataSet/interim/metadata.csv"
    mostrar_imagen_aleatoria_desde_metadata(ruta_metadata)
