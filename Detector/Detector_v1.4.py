#!/usr/bin/env python3
import cv2
import numpy as np
import math

# ==========================================================
# CONFIG
# ==========================================================
VIDEO_PATH = "../Generador de videos/video_real.mp4"
GAMMA = 1.3
MAX_JUMP = 200            # tolerancia mayor para match predicción ↔ punto
MAX_ASSIGN_DIST = 300     # distancia máxima para asignar detecciones a LEDs
STABILITY_FRAMES = 2

# HSV más amplio para mejorar detección
HSV_LOWER = np.array([115, 40, 40])
HSV_UPPER = np.array([180, 255, 255])


# ==========================================================
# UTILIDADES
# ==========================================================
def apply_gamma(frame, gamma):
    invG = 1.0 / gamma
    table = np.array([(i/255.0)**invG * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def circular_score(cnt):
    area = cv2.contourArea(cnt)
    if area < 5:
        return -1
    (_, _), r = cv2.minEnclosingCircle(cnt)
    return area / (math.pi*r*r)


# ==========================================================
# KALMAN
# ==========================================================
def create_kalman():
    k = cv2.KalmanFilter(4, 2)
    k.measurementMatrix = np.eye(2,4, dtype=np.float32)
    k.transitionMatrix = np.array([
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=np.float32)
    k.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
    k.errorCovPost = np.eye(4, dtype=np.float32)
    return k


kalman_filters = [create_kalman(), create_kalman(), create_kalman()]
measurements = [np.zeros((2,1), dtype=np.float32) for _ in range(3)]

last_valid = [None, None, None]   # últimas posiciones válidas (detección o predicción)


# ==========================================================
# DETECCIÓN FLEXIBLE (1 → 3 LEDs)
# ==========================================================
def detectar_leds(frame):
    frame_gamma = apply_gamma(frame, GAMMA)
    blur = cv2.GaussianBlur(frame_gamma,(7,7),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    cnts, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in cnts:
        if circular_score(c) < 0.18:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        pts.append((cx,cy))

    return pts, mask


# ==========================================================
# ASIGNACIÓN POR DISTANCIA (detecciones → 3 LEDs)
# ==========================================================
def asignar_puntos(preds, detecciones):
    asignados = [None, None, None]
    usados = set()

    for led_id in range(3):
        px, py = preds[led_id]
        mejor_d = 99999
        mejor_p = None
        mejor_i = None

        for i, (x,y) in enumerate(detecciones):
            if i in usados:
                continue
            d = math.dist((px,py),(x,y))
            if d < mejor_d and d < MAX_ASSIGN_DIST:
                mejor_d = d
                mejor_i = i
                mejor_p = (x,y)

        if mejor_i is not None:
            usados.add(mejor_i)
            asignados[led_id] = mejor_p

    return asignados


# ==========================================================
# MAIN
# ==========================================================
def main():
    global last_valid

    cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_PATH != "" else 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detecciones, mask = detectar_leds(frame)

        # PREDICCIÓN DE LOS 3 LEDs
        preds = []
        for kf in kalman_filters:
            p = kf.predict()
            preds.append((int(p[0]), int(p[1])))

        # ASIGNACIÓN
        asignados = asignar_puntos(preds, detecciones)

        # CORREGIR SOLO LOS LEDs DETECTADOS
        for i in range(3):
            if asignados[i] is not None:
                x, y = asignados[i]
                measurements[i][0][0] = x
                measurements[i][1][0] = y
                kalman_filters[i].correct(measurements[i])
                last_valid[i] = (x,y)
            else:
                # Si se pierde → usar predicción
                last_valid[i] = preds[i]

        # =======================================================
        # DIBUJAR
        # =======================================================
        out = frame.copy()

        for i in range(3):
            x,y = last_valid[i]
            if asignados[i] is not None:
                cv2.circle(out, (x,y), 7, (0,0,255), -1)  # REAL rojo
            else:
                cv2.circle(out, (x,y), 9, (0,255,255), 2) # PREDICCIÓN perdido (amarillo)

        # Triángulo con líneas verdes (predicción general)
        cv2.line(out,last_valid[0],last_valid[1],(0,255,0),2)
        cv2.line(out,last_valid[1],last_valid[2],(0,255,0),2)
        cv2.line(out,last_valid[2],last_valid[0],(0,255,0),2)

        cv2.putText(out,"KALMAN PREDICTIVO (si falta LED)",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow("Crudo", frame)
        cv2.imshow("Mascara HSV", mask)
        cv2.imshow("Tracking", out)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
