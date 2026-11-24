#!/usr/bin/env python3
"""
LED Triangle Tracker v3
=======================
Basado en el código original que YA FUNCIONABA.
Mejoras incrementales sin romper la detección base.
"""

import cv2
import numpy as np
import math

# ==========================================================
# CONFIG (mismos valores que funcionaban)
# ==========================================================
VIDEO_PATH = "../Generador de videos/video_real.mp4"
GAMMA = 1.3
MAX_ASSIGN_DIST = 300

# HSV original que funcionaba
HSV_LOWER = np.array([115, 40, 40])
HSV_UPPER = np.array([180, 255, 255])

# Geometría del triángulo (se calibra automáticamente)
GEOMETRY_TOLERANCE = 0.30  # 30% tolerancia en proporciones de lados


# ==========================================================
# UTILIDADES (igual que original)
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
    return area / (math.pi * r * r)


# ==========================================================
# KALMAN (igual que original)
# ==========================================================
def create_kalman():
    k = cv2.KalmanFilter(4, 2)
    k.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    k.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    k.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
    k.errorCovPost = np.eye(4, dtype=np.float32)
    return k


kalman_filters = [create_kalman(), create_kalman(), create_kalman()]
measurements = [np.zeros((2, 1), dtype=np.float32) for _ in range(3)]
last_valid = [None, None, None]


# ==========================================================
# MODELO GEOMÉTRICO DEL TRIÁNGULO
# ==========================================================
class TriangleGeometry:
    """
    Aprende y usa la geometría del triángulo para:
    1. Validar que 3 puntos formen EL triángulo correcto
    2. Inferir un vértice faltante cuando solo hay 2 detectados
    """
    
    def __init__(self):
        self.calibrated = False
        # Ratios de los lados (ordenados de menor a mayor)
        self.side_ratios = None
        # Historial para calibración robusta
        self.ratio_history = []
        self.calibration_frames = 20
    
    def _get_side_ratios(self, p0, p1, p2):
        """Calcula ratios de lados normalizados."""
        d01 = math.dist(p0, p1)
        d12 = math.dist(p1, p2)
        d20 = math.dist(p2, p0)
        
        sides = sorted([d01, d12, d20])
        if sides[2] < 20:  # Triángulo muy pequeño
            return None
        
        # Normalizar por el lado más largo
        return [s / sides[2] for s in sides]
    
    def calibrate(self, points):
        """Actualiza calibración con un triángulo detectado."""
        if len(points) != 3 or any(p is None for p in points):
            return
        
        ratios = self._get_side_ratios(points[0], points[1], points[2])
        if ratios is None:
            return
        
        self.ratio_history.append(ratios)
        
        if len(self.ratio_history) >= self.calibration_frames:
            # Usar mediana para robustez
            arr = np.array(self.ratio_history[-self.calibration_frames:])
            self.side_ratios = np.median(arr, axis=0).tolist()
            self.calibrated = True
    
    def validate_triangle(self, points):
        """
        Verifica si los 3 puntos forman un triángulo consistente.
        Retorna: (es_válido, confianza 0-1)
        """
        if not self.calibrated:
            return True, 0.5
        
        if len(points) != 3 or any(p is None for p in points):
            return False, 0.0
        
        ratios = self._get_side_ratios(points[0], points[1], points[2])
        if ratios is None:
            return False, 0.0
        
        # Comparar con ratios calibrados
        diff = sum(abs(r - c) for r, c in zip(ratios, self.side_ratios))
        
        if diff < GEOMETRY_TOLERANCE:
            return True, 1.0 - diff / GEOMETRY_TOLERANCE
        return False, 0.0
    
    def infer_third_vertex(self, p0, p1, last_p2):
        """
        Dado dos vértices detectados y la última posición del tercero,
        infiere dónde debería estar el tercer vértice.
        """
        if not self.calibrated or last_p2 is None:
            return last_p2
        
        # Distancia entre los dos puntos conocidos
        d01 = math.dist(p0, p1)
        if d01 < 20:
            return last_p2
        
        # Vector del lado conocido
        v01 = (p1[0] - p0[0], p1[1] - p0[1])
        
        # Punto medio
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        
        # Vector perpendicular normalizado
        perp_len = math.sqrt(v01[0]**2 + v01[1]**2)
        perp = (-v01[1] / perp_len, v01[0] / perp_len)
        
        # Estimar altura del triángulo usando los ratios conocidos
        # El ratio medio nos da una idea de la "forma" del triángulo
        avg_ratio = sum(self.side_ratios) / 3
        est_height = d01 * avg_ratio * 0.7  # Factor empírico
        
        # Dos candidatos (arriba y abajo del segmento)
        cand1 = (mid[0] + perp[0] * est_height, mid[1] + perp[1] * est_height)
        cand2 = (mid[0] - perp[0] * est_height, mid[1] - perp[1] * est_height)
        
        # Elegir el más cercano a la última posición conocida
        d1 = math.dist(cand1, last_p2)
        d2 = math.dist(cand2, last_p2)
        
        return cand1 if d1 < d2 else cand2


geometry = TriangleGeometry()


# ==========================================================
# DETECCIÓN (igual que original, funciona bien)
# ==========================================================
def detectar_leds(frame):
    frame_gamma = apply_gamma(frame, GAMMA)
    blur = cv2.GaussianBlur(frame_gamma, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in cnts:
        if circular_score(c) < 0.18:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        pts.append((cx, cy))

    return pts, mask


# ==========================================================
# ASIGNACIÓN (igual que original)
# ==========================================================
def asignar_puntos(preds, detecciones):
    asignados = [None, None, None]
    usados = set()

    for led_id in range(3):
        px, py = preds[led_id]
        mejor_d = 99999
        mejor_i = None

        for i, (x, y) in enumerate(detecciones):
            if i in usados:
                continue
            d = math.dist((px, py), (x, y))
            if d < mejor_d and d < MAX_ASSIGN_DIST:
                mejor_d = d
                mejor_i = i

        if mejor_i is not None:
            usados.add(mejor_i)
            asignados[led_id] = detecciones[mejor_i]

    return asignados


# ==========================================================
# MAIN
# ==========================================================
def main():
    global last_valid

    cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_PATH != "" else 0)
    
    frame_count = 0
    stats = {'detected': 0, 'predicted': 0, 'geometric': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        detecciones, mask = detectar_leds(frame)

        # PREDICCIÓN de los 3 LEDs
        preds = []
        for kf in kalman_filters:
            p = kf.predict()
            preds.append((int(p[0]), int(p[1])))

        # ASIGNACIÓN
        asignados = asignar_puntos(preds, detecciones)
        
        # Contar cuántos se detectaron
        n_detectados = sum(1 for a in asignados if a is not None)
        
        # Estados para visualización
        estados = ['PRED', 'PRED', 'PRED']  # Por defecto predicción

        # CORREGIR Kalman con detecciones
        for i in range(3):
            if asignados[i] is not None:
                x, y = asignados[i]
                measurements[i][0][0] = x
                measurements[i][1][0] = y
                kalman_filters[i].correct(measurements[i])
                last_valid[i] = (x, y)
                estados[i] = 'DET'
                stats['detected'] += 1
            else:
                # Usar predicción
                last_valid[i] = preds[i]
                stats['predicted'] += 1

        # =====================================================
        # MEJORA GEOMÉTRICA: inferir LED faltante
        # =====================================================
        if n_detectados == 2 and geometry.calibrated:
            # Encontrar cuál falta y cuáles están
            detected_ids = [i for i in range(3) if asignados[i] is not None]
            missing_id = [i for i in range(3) if asignados[i] is None][0]
            
            p0 = last_valid[detected_ids[0]]
            p1 = last_valid[detected_ids[1]]
            last_missing = preds[missing_id]  # La predicción Kalman
            
            # Inferir posición geométrica
            inferred = geometry.infer_third_vertex(p0, p1, last_missing)
            
            if inferred is not None:
                # Validar que el triángulo resultante sea coherente
                test_triangle = [None, None, None]
                test_triangle[detected_ids[0]] = p0
                test_triangle[detected_ids[1]] = p1
                test_triangle[missing_id] = inferred
                
                is_valid, conf = geometry.validate_triangle(test_triangle)
                
                if is_valid and conf > 0.3:
                    last_valid[missing_id] = (int(inferred[0]), int(inferred[1]))
                    estados[missing_id] = 'GEO'
                    stats['geometric'] += 1

        # Calibrar geometría cuando tenemos 3 detecciones buenas
        if n_detectados == 3:
            geometry.calibrate([last_valid[0], last_valid[1], last_valid[2]])

        # =======================================================
        # DIBUJAR
        # =======================================================
        out = frame.copy()

        # Colores según estado
        colores = {
            'DET': (0, 255, 0),    # Verde - detectado
            'PRED': (0, 255, 255), # Amarillo - predicción Kalman
            'GEO': (0, 165, 255),  # Naranja - inferido geométricamente
        }

        for i in range(3):
            if last_valid[i] is None:
                continue
            x, y = last_valid[i]
            color = colores[estados[i]]
            
            if estados[i] == 'DET':
                cv2.circle(out, (x, y), 7, color, -1)
            else:
                cv2.circle(out, (x, y), 9, color, 2)
            
            # Etiqueta
            cv2.putText(out, f"LED{i} ({estados[i]})", (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Triángulo
        if all(v is not None for v in last_valid):
            cv2.line(out, last_valid[0], last_valid[1], (100, 100, 100), 1)
            cv2.line(out, last_valid[1], last_valid[2], (100, 100, 100), 1)
            cv2.line(out, last_valid[2], last_valid[0], (100, 100, 100), 1)

        # Info
        calib_text = "CALIBRADO" if geometry.calibrated else "CALIBRANDO..."
        cv2.putText(out, f"Geometria: {calib_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if geometry.calibrated else (0, 165, 255), 2)
        cv2.putText(out, f"Frame: {frame_count} | Detectados: {n_detectados}/3", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Leyenda
        cv2.circle(out, (20, 80), 6, colores['DET'], -1)
        cv2.putText(out, "Detectado", (35, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colores['DET'], 1)
        cv2.circle(out, (120, 80), 6, colores['PRED'], -1)
        cv2.putText(out, "Predicho", (135, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colores['PRED'], 1)
        cv2.circle(out, (220, 80), 6, colores['GEO'], -1)
        cv2.putText(out, "Geometrico", (235, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colores['GEO'], 1)

        cv2.imshow("Original", frame)
        cv2.imshow("Mascara HSV", mask)
        cv2.imshow("Tracking", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            geometry.__init__()
            for i in range(3):
                kalman_filters[i] = create_kalman()
                last_valid[i] = None
            print("Reset!")
        elif key == ord('s'):
            total = sum(stats.values())
            if total > 0:
                print(f"\nEstadísticas ({frame_count} frames):")
                for k, v in stats.items():
                    print(f"  {k}: {v} ({100*v/total:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()