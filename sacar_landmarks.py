import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_hands = mp.solutions.hands

def procesar_imagen(ruta_img, hands_model):
    img = cv2.imread(ruta_img)

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resultado = hands_model.process(img_rgb)

    if not resultado.multi_hand_landmarks:
        return None

    mano = resultado.multi_hand_landmarks[0]

    puntos = []

    for lm in mano.landmark:
        puntos.append([lm.x, lm.y, lm.z])

    return np.array(puntos)  # (21,3)


def extraer_landmarks(df_metadata):
    data = []

    with mp_hands.Hands(static_image_mode=True) as hands:

        for i, row in df_metadata.iterrows():
            ruta = row["ruta"]

            puntos = procesar_imagen(ruta, hands)

            if puntos is None:
                continue

            fila = {}

            # guardar puntos como features
            for i in range(21):
                fila[f"x_{i}"] = puntos[i][0]
                fila[f"y_{i}"] = puntos[i][1]
                fila[f"z_{i}"] = puntos[i][2]

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