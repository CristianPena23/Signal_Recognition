import cv2
import mediapipe as mp
import time
import urllib.request
import os
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter
from skimage.feature import hog as skimage_hog
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODELS_DIR                = "models"
RUTA_FEATURES_SELECCIONADAS = "DataSet/processed/features_vocales_seleccionadas.csv"
VOCALES = {"A", "E", "I", "O", "U"}
_CANVAS = 64  # tamaño del canvas para Hu/HOG

# Nombres de las 410 features en el mismo orden que _extraer_features() las produce.
# Permite mapear columnas del CSV de selección a índices del vector completo.
_META_COLS = {"clase", "persona", "dataset"}
_ALL_FEATURE_NAMES = (
    [f"{c}n_{i}" for i in range(21) for c in ("x", "y", "z")]          # 63
    + ["dist_muneca_pulgar", "dist_muneca_indice", "dist_muneca_medio",
       "dist_muneca_anular",  "dist_muneca_menique"]                      # 5
    + ["dist_pulgar_indice", "dist_indice_medio",  "dist_medio_anular",
       "dist_anular_menique", "dist_pulgar_menique"]                      # 5
    + ["angulo_indice", "angulo_medio", "angulo_anular",
       "angulo_menique", "angulo_pulgar"]                                  # 5
    + ["apertura_mano"]                                                    # 1
    + [f"hu_{i}"  for i in range(7)]                                      # 7
    + [f"hog_{i}" for i in range(324)]                                    # 324
)  # total: 410


def _cargar_indices_seleccionados(ruta_csv):
    """Lee solo el encabezado del CSV y devuelve los índices posicionales
    que corresponden a las features seleccionadas dentro del vector de 410."""
    cols = pd.read_csv(ruta_csv, nrows=0).columns.tolist()
    feats_sel = [c for c in cols if c not in _META_COLS]
    nombre_a_idx = {nombre: idx for idx, nombre in enumerate(_ALL_FEATURE_NAMES)}
    indices = np.array([nombre_a_idx[f] for f in feats_sel], dtype=int)
    return indices, feats_sel

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def descargar_modelo_si_falta():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")


def dibujar_landmarks(frame, resultado):
    h, w = frame.shape[:2]
    for hand_landmarks in resultado.hand_landmarks:
        puntos = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, puntos[a], puntos[b], (0, 200, 0), 2)
        for x, y in puntos:
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


# --- Extracción de features (mismo pipeline que extraer_caracteristicas.py) ---

def _distancia_3d(p1, p2):
    return np.linalg.norm(p2 - p1)


def _angulo_3_puntos(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _normalizar(puntos):
    centrados = puntos - puntos[0]
    dist_ref = np.linalg.norm(puntos[9] - puntos[0])
    return centrados if dist_ref == 0 else centrados / dist_ref


def _landmarks_a_imagen(puntos_norm, tam=_CANVAS):
    img = np.zeros((tam, tam), dtype=np.uint8)
    xy = puntos_norm[:, :2].copy()
    margen = 6
    rango = xy.max(axis=0) - xy.min(axis=0)
    rango[rango == 0] = 1.0
    pts = np.clip(
        np.round((xy - xy.min(axis=0)) / rango * (tam - 2 * margen) + margen).astype(int),
        0, tam - 1,
    )
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), 255, 1)
    for pt in pts:
        cv2.circle(img, tuple(pt), 2, 255, -1)
    return img


def _hu_moments(img):
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def _hog_features(img):
    return skimage_hog(
        img, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        visualize=False, feature_vector=True,
    )


def _extraer_features(puntos_norm):
    """Produce el vector de 410 features en el mismo orden que features_vocales.csv."""
    f = []
    # 63 coordenadas normalizadas (xn_i, yn_i, zn_i para i=0..20)
    f.extend(puntos_norm.flatten())
    # 5 distancias muñeca → punta de cada dedo
    for idx in [4, 8, 12, 16, 20]:
        f.append(_distancia_3d(puntos_norm[0], puntos_norm[idx]))
    # 5 distancias entre puntas vecinas
    for i, j in [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]:
        f.append(_distancia_3d(puntos_norm[i], puntos_norm[j]))
    # 5 ángulos de flexión (orden: indice, medio, anular, menique, pulgar)
    f.append(_angulo_3_puntos(puntos_norm[5],  puntos_norm[6],  puntos_norm[8]))
    f.append(_angulo_3_puntos(puntos_norm[9],  puntos_norm[10], puntos_norm[12]))
    f.append(_angulo_3_puntos(puntos_norm[13], puntos_norm[14], puntos_norm[16]))
    f.append(_angulo_3_puntos(puntos_norm[17], puntos_norm[18], puntos_norm[20]))
    f.append(_angulo_3_puntos(puntos_norm[1],  puntos_norm[2],  puntos_norm[4]))
    # 1 apertura general de la mano
    f.append(_distancia_3d(puntos_norm[4], puntos_norm[20]))
    # 7 Hu Moments + 324 HOG
    img = _landmarks_a_imagen(puntos_norm)
    f.extend(_hu_moments(img))
    f.extend(_hog_features(img))
    return np.array(f, dtype=np.float32)


# -------------------------------------------------------------------------------

def iniciar_deteccion_tiempo_real():
    descargar_modelo_si_falta()

    clf = joblib.load(os.path.join(MODELS_DIR, "modelo_randomforest_seleccionadas.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder_seleccionadas.pkl"))

    indices_sel, feats_sel = _cargar_indices_seleccionados(RUTA_FEATURES_SELECCIONADAS)
    print(f"Features seleccionadas: {len(feats_sel)}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)  # índice 1 = iVCam (iPhone)
    if not cap.isOpened():
        print("No se pudo abrir la webcam")
        return

    historial = deque(maxlen=15)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la webcam")
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.monotonic() * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]

            if resultado.hand_landmarks:
                dibujar_landmarks(frame, resultado)

                # Landmarks → numpy (21, 3)
                lms = resultado.hand_landmarks[0]
                puntos = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                puntos_norm  = _normalizar(puntos)
                feats_todas  = _extraer_features(puntos_norm)
                feats        = feats_todas[indices_sel].reshape(1, -1)
                pred = clf.predict(feats)[0]
                historial.append(le.inverse_transform([pred])[0])

                # Votación por mayoría sobre los últimos 15 frames
                conteo = Counter(historial)
                letra_votada, votos = conteo.most_common(1)[0]
                confianza = int(votos / len(historial) * 100)

                # Recuadro negro semitransparente en esquina inferior izquierda
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 175), (250, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Letra grande
                cv2.putText(frame, letra_votada, (15, h - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                # Porcentaje de confianza
                cv2.putText(frame, f"Confianza: {confianza}%", (15, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

            else:
                historial.clear()
                cv2.putText(frame, "Sin mano", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("LSC - Vocales seleccionadas (A E I O U)", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_deteccion_tiempo_real()
