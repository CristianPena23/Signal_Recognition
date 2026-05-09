import cv2
import mediapipe as mp
import time
import urllib.request
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Conexiones del esqueleto de la mano (índices de landmarks MediaPipe)
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


def iniciar_deteccion_tiempo_real():
    descargar_modelo_si_falta()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam")
        return

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

            if resultado.hand_landmarks:
                dibujar_landmarks(frame, resultado)
                cv2.putText(
                    frame, "Mano detectada", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    frame, "No se detecta mano", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                )

            cv2.imshow("Deteccion de landmarks en tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_deteccion_tiempo_real()
