#!/usr/bin/env python3
"""
LED Linear Tracker v1.6
=======================
Video: 20260414_164706.mp4 (6 LEDs blancos en linea, parpadeo 15 Hz)

Estrategia robusta:
  FASE 1 (ADQUISICION):
    - Detecta blobs brillantes en CADA frame con varios umbrales.
    - ACUMULA blobs en el tiempo (porque parpadean: no se ven todos a la vez).
    - Busca 6 puntos COLINEALES y EQUIESPACIADOS entre los candidatos
      (RANSAC: prueba cada par como extremos del segmento de 6 LEDs).
    - Cuando los encuentra, LOCKEA.

  FASE 2 (LOCKED):
    - Para cada uno de los 6 LEDs hace busqueda local (ROI) alrededor de
      su ultima posicion conocida.
    - Si encuentra blob brillante en la ROI -> actualiza posicion.
    - Si no encuentra -> mantiene la posicion (LED en off).
    - NUNCA baja de 6 LEDs. Si querés re-adquirir, presiona 'r'.

Controles:
  Trackbars en ventana "Controles": Brillo, Area min/max, Circ, ROI radius.
  q       = salir
  r       = reset / re-adquisicion
  d       = debug en consola
  ESPACIO = pausa / reanudar
  +       = mas rapido (x1.5)
  -       = mas lento  (/1.5)
  0       = velocidad original (1x)
"""

import cv2
import numpy as np
import math
import time

# ==========================================================
# CONFIG
# ==========================================================
VIDEO_PATH = "../Generador de videos/20260414_164706.mp4"
N_LEDS = 6

# Valores iniciales (modificables con trackbars)
INIT_BRIGHTNESS = 180
INIT_MIN_AREA = 80
INIT_MAX_AREA = 2000
INIT_CIRC_X100 = 55         # circularidad minima (0-100). LEDs son redondos
INIT_ROI_RADIUS = 45

# Multiples umbrales que se prueban en ADQUISICION (de mas alto a mas bajo).
# Menos niveles = mas rapido. 4 alcanza para cubrir el rango de parpadeo.
ACQUISITION_THRESHOLDS = [200, 160, 120, 80]

# Tolerancias para reconocer la fila de 6 LEDs durante ADQUISICION
LINE_PERP_TOL = 0.10     # desviacion perpendicular relativa al largo
SPACING_TOL = 0.30       # tolerancia en espaciado relativo

# ---- Sanity check durante LOCKED (mas estrictos) ----
SANITY_SPACING_TOL = 0.30        # max desviacion del espaciado vs referencia
SANITY_PERP_TOL = 0.08           # max desviacion perpendicular de la linea
SANITY_NO_DET_FRAMES = 45        # frames seguidos sin ningun LED detectado real

# Acumulador temporal de blobs (para juntar parpadeos)
ACCUMULATOR_FRAMES = 25
ACCUMULATOR_MERGE_DIST = 12


# ==========================================================
# DETECCION DE BLOBS BRILLANTES
# ==========================================================
def detect_blobs(gray_blur, threshold, min_area, max_area, circ_min):
    _, mask = cv2.threshold(gray_blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        # Filtro de circularidad: los LEDs son redondos, los cables/reflejos no.
        if circ_min > 0:
            (_, _), r = cv2.minEnclosingCircle(c)
            if r < 1:
                continue
            if area / (math.pi * r * r) < circ_min:
                continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        pts.append((cx, cy))
    return pts, mask


def detect_multi_threshold(frame, min_area, max_area, circ_min):
    """Corre el detector con varios umbrales y devuelve la UNION de blobs."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)   # kernel chico = mas rapido
    all_blobs = []
    main_mask = None
    for th in ACQUISITION_THRESHOLDS:
        pts, mask = detect_blobs(blur, th, min_area, max_area, circ_min)
        if main_mask is None:
            main_mask = mask
        for p in pts:
            if not any(math.dist(p, q) < ACCUMULATOR_MERGE_DIST for q in all_blobs):
                all_blobs.append(p)
    return all_blobs, main_mask


def detect_single_threshold(frame, threshold, min_area, max_area, circ_min):
    """Detector rapido con UN solo umbral. Usado en fase LOCKED."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return detect_blobs(blur, threshold, min_area, max_area, circ_min)


def check_configuration_valid(positions, reference_spacings):
    """
    Verifica que las 6 posiciones formen una fila valida:
      a) Colineales (desviacion perpendicular < SANITY_PERP_TOL del largo).
      b) Espaciado coherente con la referencia (permitiendo escala uniforme).
    Devuelve (ok: bool, motivo: str).
    """
    pts = np.array(positions, dtype=np.float32)
    # --- Colinealidad: ajusto recta por SVD y mido desviacion perpendicular ---
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    perp = np.array([-direction[1], direction[0]])
    perp_dists = [abs(float(np.dot(p - mean, perp))) for p in pts]
    max_perp = max(perp_dists)
    total_len = float(math.dist(positions[0], positions[-1]))
    if total_len < 10:
        return False, "colapso"
    if max_perp / total_len > SANITY_PERP_TOL:
        return False, f"no-colineal ({max_perp/total_len:.2f})"

    # --- Espaciado vs referencia (con escala uniforme) ---
    current = [math.dist(positions[i], positions[i + 1])
               for i in range(len(positions) - 1)]
    if reference_spacings is None or sum(reference_spacings) <= 0:
        return True, ""
    scale = sum(current) / sum(reference_spacings)
    max_dev = max(abs(c - scale * r) / (scale * r)
                  for c, r in zip(current, reference_spacings))
    if max_dev > SANITY_SPACING_TOL:
        return False, f"espaciado-roto ({max_dev:.2f})"
    return True, ""


# ==========================================================
# ACUMULADOR TEMPORAL
# ==========================================================
class BlobAccumulator:
    """Junta blobs detectados en frames sucesivos para vencer el parpadeo."""

    def __init__(self, history_frames=ACCUMULATOR_FRAMES, merge_dist=ACCUMULATOR_MERGE_DIST):
        self.blobs = []  # cada elemento: [x, y, last_age, hits]
        self.age = 0
        self.history = history_frames
        self.merge_dist = merge_dist

    def add(self, new_blobs):
        self.age += 1
        # Eliminar viejos
        self.blobs = [b for b in self.blobs if self.age - b[2] <= self.history]
        for nx, ny in new_blobs:
            merged = False
            for b in self.blobs:
                if math.dist((nx, ny), (b[0], b[1])) < self.merge_dist:
                    # Promediar posicion suavemente
                    b[0] = int(0.7 * b[0] + 0.3 * nx)
                    b[1] = int(0.7 * b[1] + 0.3 * ny)
                    b[2] = self.age
                    b[3] += 1
                    merged = True
                    break
            if not merged:
                self.blobs.append([int(nx), int(ny), self.age, 1])

    def positions(self, min_hits=1):
        return [(b[0], b[1]) for b in self.blobs if b[3] >= min_hits]

    def reset(self):
        self.blobs = []
        self.age = 0


# ==========================================================
# RANSAC: 6 puntos COLINEALES y EQUIESPACIADOS
# ==========================================================
def find_n_equispaced_collinear(blobs, n=N_LEDS):
    """
    Busca n puntos colineales y equiespaciados entre los candidatos.
    Prueba cada par de blobs como (primer, ultimo) y verifica que haya
    n-2 blobs en medio en posiciones equiespaciadas.

    Devuelve lista ordenada de n puntos o None.
    """
    if len(blobs) < n:
        return None
    pts = np.array(blobs, dtype=np.float32)
    nb = len(pts)
    best = None
    best_residual = float('inf')

    for i in range(nb):
        for j in range(nb):
            if i == j:
                continue
            a, b = pts[i], pts[j]
            v = b - a
            length = float(np.linalg.norm(v))
            if length < 40:
                continue
            step_vec = v / (n - 1)
            step_len = length / (n - 1)
            v_perp = np.array([-v[1], v[0]]) / length

            chosen = [None] * n
            chosen[0] = pts[i]
            chosen[n - 1] = pts[j]
            used = {i, j}
            residual = 0.0
            valid = True

            for k in range(1, n - 1):
                expected = a + k * step_vec
                best_m, best_d = -1, float('inf')
                for m in range(nb):
                    if m in used:
                        continue
                    d = float(np.linalg.norm(pts[m] - expected))
                    if d < best_d:
                        best_d = d
                        best_m = m
                tol = SPACING_TOL * step_len
                if best_m < 0 or best_d > tol:
                    valid = False
                    break
                chosen[k] = pts[best_m]
                used.add(best_m)
                residual += best_d

            if not valid:
                continue

            max_perp = max(abs(np.dot(p - a, v_perp)) for p in chosen)
            if max_perp / length > LINE_PERP_TOL:
                continue

            if residual < best_residual:
                best_residual = residual
                best = [(int(p[0]), int(p[1])) for p in chosen]

    return best


# ==========================================================
# TRACKING DE CUERPO RIGIDO (fase LOCKED)
# ==========================================================
def assign_blobs_to_leds(predicted, blobs, max_dist):
    """Asigna greedy cada LED a un blob (los mas cercanos primero)."""
    assigned = [None] * len(predicted)
    used = set()
    candidatos = []
    for i, p in enumerate(predicted):
        for j, b in enumerate(blobs):
            d = math.dist(p, b)
            if d < max_dist:
                candidatos.append((d, i, j))
    candidatos.sort()
    for d, i, j in candidatos:
        if assigned[i] is not None or j in used:
            continue
        assigned[i] = blobs[j]
        used.add(j)
    return assigned


def update_locked_rigid(frame, locked_positions, velocities,
                        threshold, min_area, max_area, circ_min, max_match_dist):
    """
    Tracking rigido + velocidad:
      1. Detecta blobs (UN solo umbral + circularidad) en todo el frame.
      2. Predice cada LED con su posicion + velocidad.
      3. Empareja blobs detectados con LEDs (greedy por distancia).
      4. Para LEDs no encontrados: aplica la traslacion MEDIANA de los
         encontrados (cuerpo rigido), de modo que todos se mueven con
         la camara incluso aunque algunos esten apagados.
      5. Suaviza posiciones y actualiza velocidad.
    Devuelve: (new_positions, new_velocities, estados, blobs_detectados, mask)
      estados[i] in {'DET','RIGID','PRED'}
    """
    blobs, mask = detect_single_threshold(frame, threshold, min_area, max_area, circ_min)

    # Predicciones con velocidad
    predicted = [(locked_positions[i][0] + velocities[i][0],
                  locked_positions[i][1] + velocities[i][1]) for i in range(N_LEDS)]
    predicted_int = [(int(p[0]), int(p[1])) for p in predicted]

    assigned = assign_blobs_to_leds(predicted_int, blobs, max_match_dist)

    matched_ids = [i for i in range(N_LEDS) if assigned[i] is not None]
    estados = ['PRED'] * N_LEDS

    # Traslacion mediana del cuerpo rigido
    if len(matched_ids) >= 1:
        tx_list = sorted(assigned[i][0] - predicted[i][0] for i in matched_ids)
        ty_list = sorted(assigned[i][1] - predicted[i][1] for i in matched_ids)
        median_tx = tx_list[len(tx_list) // 2]
        median_ty = ty_list[len(ty_list) // 2]
    else:
        median_tx = 0.0
        median_ty = 0.0

    new_positions = []
    new_velocities = []
    for i in range(N_LEDS):
        if assigned[i] is not None:
            target = assigned[i]
            estados[i] = 'DET'
        elif len(matched_ids) >= 1:
            target = (predicted[i][0] + median_tx, predicted[i][1] + median_ty)
            estados[i] = 'RIGID'
        else:
            target = predicted[i]
            estados[i] = 'PRED'

        # Suavizado de posicion (mas peso al target nuevo cuando es deteccion directa)
        alpha = 0.75 if estados[i] == 'DET' else 0.55
        sx = (1 - alpha) * locked_positions[i][0] + alpha * target[0]
        sy = (1 - alpha) * locked_positions[i][1] + alpha * target[1]

        # Velocidad suavizada (filtro exponencial)
        vx_new = sx - locked_positions[i][0]
        vy_new = sy - locked_positions[i][1]
        vx = 0.5 * velocities[i][0] + 0.5 * vx_new
        vy = 0.5 * velocities[i][1] + 0.5 * vy_new

        new_positions.append((int(sx), int(sy)))
        new_velocities.append((vx, vy))

    return new_positions, new_velocities, estados, blobs, mask


# ==========================================================
# MAIN
# ==========================================================
def nothing(_):
    pass


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {VIDEO_PATH}")
        return

    # Velocidad de reproduccion: leer FPS del video para respetarlo
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0 or fps_video > 240:
        fps_video = 30.0   # fallback si el contenedor no reporta fps
    frame_interval = 1.0 / fps_video   # segundos por frame a velocidad 1x
    speed_factor = 1.0                 # multiplicador modificable con +/-/0
    paused = False
    print(f"Video FPS: {fps_video:.2f}  intervalo: {frame_interval*1000:.1f} ms")

    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", 420, 200)
    cv2.createTrackbar("Brillo",    "Controles", INIT_BRIGHTNESS, 255, nothing)
    cv2.createTrackbar("Area min",  "Controles", INIT_MIN_AREA, 1000, nothing)
    cv2.createTrackbar("Area max",  "Controles", INIT_MAX_AREA, 8000, nothing)
    cv2.createTrackbar("Circ x100", "Controles", INIT_CIRC_X100, 100, nothing)
    cv2.createTrackbar("ROI",       "Controles", INIT_ROI_RADIUS, 200, nothing)

    accumulator = BlobAccumulator()
    locked = False
    locked_positions = [None] * N_LEDS
    velocities = [(0.0, 0.0)] * N_LEDS
    reference_spacings = None     # espaciados al momento de lockear (para sanity check)
    no_det_streak = 0             # frames seguidos sin ningun LED en estado DET
    frame_count = 0

    # Para frame-skip y medicion de FPS real
    wall_start = time.perf_counter()
    expected_video_time = 0.0     # segundos transcurridos del video que deberiamos haber procesado
    fps_real_smoothed = fps_video
    last_display_t = time.perf_counter()
    frames_skipped_total = 0

    last_frame = None
    while True:
        t_start = time.perf_counter()

        if not paused:
            # ---- FRAME-SKIP por wall-clock ----
            # Si vamos atrasados respecto al tiempo "real" del video, saltamos
            # frames sin procesarlos para mantener velocidad real.
            elapsed_wall = (time.perf_counter() - wall_start) * speed_factor
            behind_s = elapsed_wall - expected_video_time
            frames_to_skip = max(0, int(behind_s / frame_interval) - 1)
            frames_to_skip = min(frames_to_skip, 30)  # cap por seguridad

            for _ in range(frames_to_skip):
                if not cap.grab():
                    break
                expected_video_time += frame_interval
                frame_count += 1
                frames_skipped_total += 1

            ret, frame = cap.read()
            if not ret:
                # El video termino: reiniciar y resetear el tracker (la escena
                # teletransporta al frame 0 y las posiciones/velocidades viejas
                # confunden la asignacion rigida).
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                locked = False
                accumulator.reset()
                locked_positions = [None] * N_LEDS
                velocities = [(0.0, 0.0)] * N_LEDS
                no_det_streak = 0
                wall_start = time.perf_counter()
                expected_video_time = 0.0
                print(f"Frame {frame_count}: video reiniciado, re-adquisicion")
                continue
            frame_count += 1
            expected_video_time += frame_interval
            last_frame = frame
        else:
            # En pausa, reusar el ultimo frame para que la ventana siga viva
            frame = last_frame
            if frame is None:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                continue

        th = cv2.getTrackbarPos("Brillo", "Controles")
        min_a = cv2.getTrackbarPos("Area min", "Controles")
        max_a = cv2.getTrackbarPos("Area max", "Controles")
        circ_min = cv2.getTrackbarPos("Circ x100", "Controles") / 100.0
        roi_r = cv2.getTrackbarPos("ROI", "Controles")

        out = frame.copy()

        if not locked:
            # ================== FASE ADQUISICION ==================
            blobs_now, mask_now = detect_multi_threshold(frame, min_a, max_a, circ_min)
            accumulator.add(blobs_now)
            candidates = accumulator.positions(min_hits=1)

            result = find_n_equispaced_collinear(candidates, N_LEDS)
            if result is not None:
                locked_positions = result
                velocities = [(0.0, 0.0)] * N_LEDS
                reference_spacings = [math.dist(result[i], result[i + 1])
                                      for i in range(N_LEDS - 1)]
                locked = True
                print(f"Frame {frame_count}: LOCKED 6 LEDs en posiciones:")
                for i, p in enumerate(locked_positions):
                    print(f"  L{i}: {p}")

            # ---- Visualizacion ----
            for (x, y) in blobs_now:
                cv2.circle(out, (x, y), 4, (255, 255, 0), -1)  # azul claro: blob actual
            for (x, y) in candidates:
                cv2.circle(out, (x, y), 8, (0, 255, 255), 1)   # amarillo: acumulado
            if result is not None:
                for i, (x, y) in enumerate(result):
                    cv2.circle(out, (x, y), 11, (0, 255, 0), 2)
                    cv2.putText(out, f"L{i}", (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(out, f"BUSCANDO  blobs ahora:{len(blobs_now)}  "
                             f"acumulados:{len(candidates)}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("Mascara", mask_now)
        else:
            # ================== FASE LOCKED (tracking rigido) ==================
            # max_match_dist depende del ROI (trackbar). Sube si la camara se mueve mucho.
            locked_positions, velocities, estados, blobs_now, mask_now = \
                update_locked_rigid(frame, locked_positions, velocities,
                                    th, min_a, max_a, circ_min, max_match_dist=roi_r)

            n_det = sum(1 for s in estados if s == 'DET')
            n_rigid = sum(1 for s in estados if s == 'RIGID')

            # ---- Contador de frames sin ningun LED detectado real ----
            if n_det == 0:
                no_det_streak += 1
            else:
                no_det_streak = 0

            # ---- SANITY CHECKS: disparan re-adquisicion automatica ----
            #   a) Configuracion geometrica rota: no colineales o espaciado raro.
            #   b) Demasiados frames seguidos sin detecciones reales (tracker
            #      "fantasma" arrastrado por el cuerpo rigido lejos de los LEDs).
            need_reacquire = False
            motivo = ""
            ok, why = check_configuration_valid(locked_positions, reference_spacings)
            if not ok:
                need_reacquire = True
                motivo = why
            elif no_det_streak >= SANITY_NO_DET_FRAMES:
                need_reacquire = True
                motivo = f"sin-detecciones ({no_det_streak}f)"

            if need_reacquire:
                print(f"Frame {frame_count}: configuracion invalida "
                      f"[{motivo}], re-adquiriendo...")
                locked = False
                accumulator.reset()
                locked_positions = [None] * N_LEDS
                velocities = [(0.0, 0.0)] * N_LEDS
                reference_spacings = None
                no_det_streak = 0
                continue

            # ---- Visualizacion ----
            colores = {
                'DET':   (0, 255, 0),     # Verde: detectado
                'RIGID': (0, 165, 255),   # Naranja: inferido por cuerpo rigido
                'PRED':  (0, 255, 255),   # Amarillo: solo prediccion (sin blob cerca)
            }
            for (x, y) in blobs_now:
                cv2.circle(out, (x, y), 3, (255, 255, 0), -1)  # blobs detectados
            for i in range(N_LEDS):
                x, y = locked_positions[i]
                color = colores[estados[i]]
                if estados[i] == 'DET':
                    cv2.circle(out, (x, y), 7, color, -1)
                else:
                    cv2.circle(out, (x, y), 9, color, 2)
                cv2.putText(out, f"L{i}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            for i in range(N_LEDS - 1):
                cv2.line(out, locked_positions[i], locked_positions[i + 1],
                         (120, 120, 120), 1)

            cv2.putText(out, f"LOCKED  det:{n_det}  rigid:{n_rigid}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("Mascara", mask_now)

        # ---- Medir FPS real (suavizado) ----
        now = time.perf_counter()
        dt_disp = now - last_display_t
        last_display_t = now
        if dt_disp > 0:
            fps_inst = 1.0 / dt_disp
            fps_real_smoothed = 0.9 * fps_real_smoothed + 0.1 * fps_inst

        cv2.putText(out, f"Frame: {frame_count}  (skip total: {frames_skipped_total})",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        speed_label = "PAUSA" if paused else f"x{speed_factor:.2f}"
        cv2.putText(out,
                    f"Vel: {speed_label}  base:{fps_video:.1f}fps  real:{fps_real_smoothed:.1f}fps",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Original", frame)
        cv2.imshow("Tracking", out)

        # ---- Esperar para respetar la velocidad del video ----
        # target = intervalo entre frames del video / factor de velocidad
        # le restamos lo que tardo el procesamiento; si estamos atrasados,
        # el frame-skip de arriba ya se encargara en la proxima vuelta.
        elapsed = time.perf_counter() - t_start
        target = frame_interval / max(speed_factor, 0.01)
        wait_ms = max(1, int((target - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            locked = False
            accumulator.reset()
            locked_positions = [None] * N_LEDS
            velocities = [(0.0, 0.0)] * N_LEDS
            no_det_streak = 0
            wall_start = time.perf_counter()
            expected_video_time = 0.0
            print("Reset / re-adquisicion")
        elif key == ord('d'):
            if not locked:
                print(f"[debug] f{frame_count} blobs={len(blobs_now)} "
                      f"acumulados={len(accumulator.positions())} "
                      f"brillo={th} area=[{min_a},{max_a}]")
            else:
                print(f"[debug] f{frame_count} LOCKED  positions={locked_positions}")
        elif key == ord(' '):
            paused = not paused
            print("Pausa" if paused else "Reanudar")
        elif key == ord('+') or key == ord('='):
            speed_factor = min(8.0, speed_factor * 1.5)
            print(f"Velocidad: x{speed_factor:.2f}")
        elif key == ord('-') or key == ord('_'):
            speed_factor = max(0.1, speed_factor / 1.5)
            print(f"Velocidad: x{speed_factor:.2f}")
        elif key == ord('0'):
            speed_factor = 1.0
            wall_start = time.perf_counter()
            expected_video_time = frame_count * frame_interval
            print("Velocidad: x1.00 (original)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
