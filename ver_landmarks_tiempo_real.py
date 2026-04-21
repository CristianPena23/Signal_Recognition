import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def iniciar_deteccion_tiempo_real():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la webcam")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame de la webcam")
                break

            # efecto espejo para que se vea más natural
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(frame_rgb)

            if resultado.multi_hand_landmarks:
                for mano_landmarks in resultado.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        mano_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                cv2.putText(
                    frame,
                    "Mano detectada",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No se detecta mano",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Deteccion de landmarks en tiempo real", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_deteccion_tiempo_real()