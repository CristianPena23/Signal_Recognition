#!/usr/bin/env python3
"""HAND2VOICE — Interfaz visual de reconocimiento de vocales en tiempo real."""

import cv2
import mediapipe as mp
import time
import urllib.request
import os
import numpy as np
import joblib
from collections import deque, Counter
from skimage.feature import hog as skimage_hog
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
APP_NAME     = "HAND2VOICE"
MODEL_PATH   = "hand_landmarker.task"
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
CLF_PATH     = os.path.join("models", "modelo_randomforest_seleccionadas.pkl")
LE_PATH      = os.path.join("models", "label_encoder_seleccionadas.pkl")
FEATURES_CSV = os.path.join("DataSet", "processed", "features_vocales_seleccionadas.csv")
_CANVAS      = 64
HISTORY_MAX  = 15

# ── Layout ────────────────────────────────────────────────────────────────────
TITLE_H = 52
CAM_W   = 640
CAM_H   = 570
PANEL_W = 420
TOT_W   = CAM_W + PANEL_W   # 1060
TOT_H   = TITLE_H + CAM_H   # 622

# Coordenada Y de inicio de cada sección dentro del panel derecho (570 px total)
S1_Y, S1_H = 0,   115   # VOCAL DETECTADA
S2_Y, S2_H = 115, 115   # DISTANCIAS muñeca→dedo
S6_Y, S6_H = 230, 95    # DISTANCIAS ENTRE PUNTAS
S3_Y, S3_H = 325, 115   # ÁNGULOS
S4_Y, S4_H = 440, 65    # APERTURA
S5_Y, S5_H = 505, 65    # CONFIANZA + HISTORIAL

# ── Colores BGR ───────────────────────────────────────────────────────────────
BG        = (18,  18,  28)
TITLE_BG  = (12,  12,  22)
TITLE_COL = (210, 175, 40)
SEC_BG    = (30,  30,  46)
SEP_COL   = (60,  60,  80)
HDR_COL   = (180, 150, 30)
GREEN     = (50,  215, 70)
ORANGE    = (25,  150, 255)
WHITE     = (225, 225, 225)
GRAY      = (115, 115, 125)
BAR_BG    = (48,  48,  60)
BAR_GRN   = (40,  190, 60)
BAR_BLU   = (190, 110, 25)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_META_COLS = {"clase", "persona", "dataset"}
_ALL_FEATURE_NAMES = (
    [f"{c}n_{i}" for i in range(21) for c in ("x", "y", "z")]          # 63
    + ["dist_muneca_pulgar", "dist_muneca_indice", "dist_muneca_medio",
       "dist_muneca_anular",  "dist_muneca_menique"]                     # 5
    + ["dist_pulgar_indice", "dist_indice_medio",  "dist_medio_anular",
       "dist_anular_menique", "dist_pulgar_menique"]                     # 5
    + ["angulo_indice", "angulo_medio", "angulo_anular",
       "angulo_menique", "angulo_pulgar"]                                 # 5
    + ["apertura_mano"]                                                   # 1
    + [f"hu_{i}"  for i in range(7)]                                     # 7
    + [f"hog_{i}" for i in range(324)]                                   # 324
)  # total 410


# ── Pipeline de features (idéntico a extraer_caracteristicas.py) ──────────────

def _dist(a, b):
    return float(np.linalg.norm(b - a))


def _angulo(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _normalizar(pts):
    c = pts - pts[0]
    d = np.linalg.norm(pts[9] - pts[0])
    return c if d == 0 else c / d


def _a_imagen(pn, tam=_CANVAS):
    img = np.zeros((tam, tam), np.uint8)
    xy = pn[:, :2].copy()
    m = 6
    rg = xy.max(0) - xy.min(0)
    rg[rg == 0] = 1.0
    p = np.clip(np.round((xy - xy.min(0)) / rg * (tam - 2 * m) + m).astype(int), 0, tam - 1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(p[a]), tuple(p[b]), 255, 1)
    for pt in p:
        cv2.circle(img, tuple(pt), 2, 255, -1)
    return img


def _hu(img):
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def _hog(img):
    return skimage_hog(
        img, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        visualize=False, feature_vector=True,
    )


def extraer(pn):
    """Produce el vector de 410 features y un dict de valores para la UI."""
    f = list(pn.flatten())  # 63 coordenadas normalizadas

    # 5 distancias muñeca → punta
    dists = {}
    for nombre, idx in zip(["pulgar", "indice", "medio", "anular", "menique"],
                           [4, 8, 12, 16, 20]):
        v = _dist(pn[0], pn[idx])
        dists[nombre] = v
        f.append(v)

    # 5 distancias entre puntas vecinas
    dists_puntas = {}
    for (i, j), nombre in zip(
        [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)],
        ["pulgar_indice", "indice_medio", "medio_anular", "anular_menique", "pulgar_menique"],
    ):
        v = _dist(pn[i], pn[j])
        dists_puntas[nombre] = v
        f.append(v)

    # 5 ángulos de flexión (orden: indice, medio, anular, menique, pulgar)
    angulos = {
        "indice":  _angulo(pn[5],  pn[6],  pn[8]),
        "medio":   _angulo(pn[9],  pn[10], pn[12]),
        "anular":  _angulo(pn[13], pn[14], pn[16]),
        "menique": _angulo(pn[17], pn[18], pn[20]),
        "pulgar":  _angulo(pn[1],  pn[2],  pn[4]),
    }
    f.extend([angulos["indice"], angulos["medio"], angulos["anular"],
              angulos["menique"], angulos["pulgar"]])

    # 1 apertura de mano
    apertura = _dist(pn[4], pn[20])
    f.append(apertura)

    # 7 Hu Moments + 324 HOG
    img = _a_imagen(pn)
    f.extend(_hu(img))
    f.extend(_hog(img))

    info = {"dists": dists, "dists_puntas": dists_puntas, "angulos": angulos, "apertura": apertura}
    return np.array(f, np.float32), info


# ── Selección de cámara (mismo esquema que ver_landmarks_tiempo_real.py) ──────

def _probar_camara(idx, resultado, timeout=5.0):
    import threading
    def _w():
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            time.sleep(3)
            ret, frame = cap.read()
            if ret and frame is not None:
                resultado[0] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    t = threading.Thread(target=_w, daemon=True)
    t.start()
    t.join(timeout=timeout)


def detectar_camaras(indices=(0, 1, 2)):
    print("\nDetectando camaras disponibles...")
    ok = []
    for idx in indices:
        r = [None]
        _probar_camara(idx, r)
        if r[0]:
            w, h = r[0]
            ok.append((idx, w, h))
            print(f"  [{idx}] OK  -  {w}x{h}")
        else:
            print(f"  [{idx}] no disponible")
    return ok


def seleccionar_camara():
    ok = detectar_camaras()
    if not ok:
        print("No se encontro ninguna camara.")
        return None
    if len(ok) == 1:
        print(f"\nCamara unica detectada: indice {ok[0][0]}")
        return ok[0][0]
    validos = {str(d[0]) for d in ok}
    print("Selecciona el indice de camara: ", end="", flush=True)
    while True:
        e = input().strip()
        if e in validos:
            return int(e)
        print(f"  Invalido. Opciones: {sorted(validos)}. Intenta de nuevo: ",
              end="", flush=True)


# ── Helpers de dibujo ─────────────────────────────────────────────────────────

_F  = cv2.FONT_HERSHEY_SIMPLEX
_FD = cv2.FONT_HERSHEY_DUPLEX


def tx(img, txt, x, y, s=0.5, c=WHITE, t=1):
    cv2.putText(img, txt, (x, y), _F, s, c, t, cv2.LINE_AA)


def barra(img, x, y, w, h, ratio, fg, bg=BAR_BG):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = max(0, int(w * float(np.clip(ratio, 0.0, 1.0))))
    if fill:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg, -1)


def hdr(panel, y, label):
    """Dibuja una barra de encabezado de sección."""
    cv2.rectangle(panel, (6, y + 2), (PANEL_W - 6, y + 18), SEC_BG, -1)
    tx(panel, label, 12, y + 14, s=0.44, c=HDR_COL)


# ── Panel derecho ─────────────────────────────────────────────────────────────

def dibujar_panel(panel, letra, conf, info, historial):
    panel[:] = BG

    # ── S1: VOCAL DETECTADA (y 0–115) ─────────────────────────────────────
    hdr(panel, S1_Y, "VOCAL DETECTADA")

    if letra:
        # Letra grande: baseline en y=107, escala 3 → ~84 px de altura → top ≈ 23
        cv2.putText(panel, letra, (22, S1_Y + 107),
                    _FD, 3.2, GREEN, 8, cv2.LINE_AA)
        pct = int(conf * 100)
        cv2.putText(panel, f"{pct}%", (155, S1_Y + 60),
                    _FD, 1.8, ORANGE, 3, cv2.LINE_AA)
        tx(panel, "de confianza", 155, S1_Y + 85, s=0.45, c=GRAY)
    else:
        tx(panel, "Sin mano detectada", 14, S1_Y + 65, s=0.65, c=(55, 55, 200), t=2)

    cv2.line(panel, (6, S2_Y - 2), (PANEL_W - 6, S2_Y - 2), SEP_COL, 1)

    # ── S2: DISTANCIAS (y 115–230) ────────────────────────────────────────
    hdr(panel, S2_Y, "DISTANCIAS  (muneca -> dedo)")

    dists   = info.get("dists", {})
    etiq_d  = [("Pulgar ", "pulgar"), ("Indice ", "indice"),
               ("Medio  ", "medio"),  ("Anular ", "anular"), ("Menique", "menique")]
    dy = S2_Y + 22
    for lbl, key in etiq_d:
        v = dists.get(key, 0.0)
        tx(panel, f"{lbl}: {v:.3f}", 10, dy + 11, s=0.46, c=WHITE)
        barra(panel, 188, dy + 1, PANEL_W - 198, 13, v / 2.8, BAR_GRN)
        dy += 18

    cv2.line(panel, (6, S6_Y - 2), (PANEL_W - 6, S6_Y - 2), SEP_COL, 1)

    # ── S6: DISTANCIAS ENTRE PUNTAS (y 230–325) ──────────────────────────
    hdr(panel, S6_Y, "DISTANCIAS ENTRE PUNTAS")

    dp     = info.get("dists_puntas", {})
    etiq_p = [("Pulgar -> Indice ", "pulgar_indice"),
              ("Indice -> Medio  ", "indice_medio"),
              ("Medio  -> Anular ", "medio_anular"),
              ("Anular -> Menique", "anular_menique")]
    dy = S6_Y + 22
    for lbl, key in etiq_p:
        v = dp.get(key, 0.0)
        tx(panel, f"{lbl}: {v:.3f}", 10, dy + 11, s=0.46, c=WHITE)
        barra(panel, 215, dy + 1, PANEL_W - 225, 13, v / 2.5, BAR_GRN)
        dy += 17

    cv2.line(panel, (6, S3_Y - 2), (PANEL_W - 6, S3_Y - 2), SEP_COL, 1)

    # ── S3: ÁNGULOS DE FLEXIÓN (y 325–440) ───────────────────────────────
    hdr(panel, S3_Y, "ANGULOS DE FLEXION (grados)")

    angulos = info.get("angulos", {})
    etiq_a  = [("Pulgar ", "pulgar"), ("Indice ", "indice"),
               ("Medio  ", "medio"),  ("Anular ", "anular"), ("Menique", "menique")]
    dy = S3_Y + 22
    for lbl, key in etiq_a:
        v = angulos.get(key, 0.0)
        tx(panel, f"{lbl}: {v:5.1f}", 10, dy + 11, s=0.46, c=WHITE)
        tx(panel, "°", 182, dy + 11, s=0.46, c=GRAY)
        barra(panel, 188, dy + 1, PANEL_W - 198, 13, v / 180.0, BAR_BLU)
        dy += 18

    cv2.line(panel, (6, S4_Y - 2), (PANEL_W - 6, S4_Y - 2), SEP_COL, 1)

    # ── S4: APERTURA DE MANO (y 440–505) ─────────────────────────────────
    hdr(panel, S4_Y, "APERTURA DE MANO")

    ap = info.get("apertura", 0.0)
    tx(panel, f"{ap:.3f}", 10, S4_Y + 42, s=0.65, c=ORANGE, t=2)
    barra(panel, 82, S4_Y + 28, PANEL_W - 92, 16, ap / 2.5, BAR_GRN)

    cv2.line(panel, (6, S5_Y - 2), (PANEL_W - 6, S5_Y - 2), SEP_COL, 1)

    # ── S5: CONFIANZA + HISTORIAL (y 505–570) ────────────────────────────
    hdr(panel, S5_Y, "CONFIANZA")

    tx(panel, f"{int(conf * 100)}%", 10, S5_Y + 40, s=0.65, c=GREEN, t=2)
    barra(panel, 62, S5_Y + 26, PANEL_W - 72, 14, conf, BAR_GRN)

    # Últimas 5 predicciones
    tx(panel, "Hist:", 10, S5_Y + 58, s=0.4, c=GRAY)
    hist5 = list(historial)[-5:]
    xh = 58
    for pred in hist5:
        col = GREEN if pred == letra else GRAY
        cv2.rectangle(panel, (xh, S5_Y + 44), (xh + 26, S5_Y + 62), SEC_BG, -1)
        cv2.rectangle(panel, (xh, S5_Y + 44), (xh + 26, S5_Y + 62), col, 1)
        tx(panel, pred, xh + 6, S5_Y + 59, s=0.52, c=col, t=1)
        xh += 32


# ── Panel izquierdo (landmarks sobre cámara) ──────────────────────────────────

def dibujar_cam(frame, resultado):
    h, w = frame.shape[:2]
    for hlms in resultado.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hlms]

        # Bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 18
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 210, 200), 2)

        # Conexiones del esqueleto
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)

        # Landmarks
        for x, y in pts:
            cv2.circle(frame, (x, y), 5, (255, 80, 0), -1)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Descargar modelo MediaPipe si falta
    if not os.path.exists(MODEL_PATH):
        print("Descargando hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  OK")

    # Cargar índices de features seleccionadas
    print("Cargando features seleccionadas ...")
    cols      = pd.read_csv(FEATURES_CSV, nrows=0).columns.tolist()
    feats_sel = [c for c in cols if c not in _META_COLS]
    n2i       = {n: i for i, n in enumerate(_ALL_FEATURE_NAMES)}
    idx_sel   = np.array([n2i[f] for f in feats_sel], dtype=int)
    print(f"  {len(feats_sel)} features seleccionadas")

    # Cargar clasificador y label encoder
    print("Cargando modelo ...")
    clf = joblib.load(CLF_PATH)
    le  = joblib.load(LE_PATH)
    print(f"  {type(clf).__name__}  ({clf.n_features_in_} features de entrada)")

    # Seleccionar cámara
    idx_cam = seleccionar_camara()
    if idx_cam is None:
        return

    cap = cv2.VideoCapture(idx_cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {idx_cam}")
        return

    time.sleep(2)
    for _ in range(30):
        if cap.read()[0]:
            break
    else:
        print("La camara no entrega frames.")
        cap.release()
        return

    # Configurar MediaPipe Hand Landmarker
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    historial = deque(maxlen=HISTORY_MAX)
    letra, conf, info = "", 0.0, {}
    panel = np.zeros((CAM_H, PANEL_W, 3), np.uint8)

    print(f"\n{'=' * 50}")
    print(f"  {APP_NAME}  —  Presiona ESC para salir")
    print(f"{'=' * 50}\n")

    with vision.HandLandmarker.create_from_options(opts) as lm_det:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))

            ts  = int(time.monotonic() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm_det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )

            if res.hand_landmarks:
                dibujar_cam(frame, res)

                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]],
                    dtype=np.float32,
                )
                pn = _normalizar(pts)
                fv, info = extraer(pn)

                pred = clf.predict(fv[idx_sel].reshape(1, -1))[0]
                historial.append(le.inverse_transform([pred])[0])

                c = Counter(historial)
                letra, votos = c.most_common(1)[0]
                conf = votos / len(historial)
            else:
                historial.clear()
                letra, conf, info = "", 0.0, {}

            # Renderizar panel derecho
            dibujar_panel(panel, letra, conf, info, historial)

            # Componer canvas completo
            canvas = np.zeros((TOT_H, TOT_W, 3), np.uint8)

            # Barra de título
            canvas[:TITLE_H, :] = TITLE_BG
            cv2.putText(canvas, APP_NAME,
                        (TOT_W // 2 - 160, 36),
                        _FD, 1.6, TITLE_COL, 2, cv2.LINE_AA)
            tx(canvas, "Reconocimiento de Vocales en Lengua de Senas",
               TOT_W // 2 - 150, 50, s=0.38, c=GRAY)
            cv2.line(canvas, (0, TITLE_H), (TOT_W, TITLE_H), SEP_COL, 1)

            # Panel izquierdo: video
            canvas[TITLE_H:, :CAM_W] = frame
            # Panel derecho: info
            canvas[TITLE_H:, CAM_W:] = panel

            # Separador vertical
            cv2.line(canvas, (CAM_W, TITLE_H), (CAM_W, TOT_H), SEP_COL, 2)

            cv2.imshow(APP_NAME, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
