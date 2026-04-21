import cv2
import mediapipe as mp
import pandas as pd
import random
import os


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def dibujar_landmarks_en_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)

    if img is None:
        print("No se pudo cargar la imagen:", ruta_imagen)
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        resultado = hands.process(img_rgb)

        if not resultado.multi_hand_landmarks:
            print("No se detectó mano en la imagen")
            cv2.imshow("Imagen sin landmarks", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        img_con_landmarks = img.copy()

        for mano_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_con_landmarks,
                mano_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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