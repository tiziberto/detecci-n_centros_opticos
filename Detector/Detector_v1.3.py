import cv2
import numpy as np
import math
from collections import deque

# --------------------------------------------------------
# RANGOS HSV para LED ROJO (ligeramente ampliados)
HSV_LOWER_RED_1 = np.array([0, 50, 50])
HSV_UPPER_RED_1 = np.array([15, 255, 255])
HSV_LOWER_RED_2 = np.array([165, 50, 50])
HSV_UPPER_RED_2 = np.array([180, 255, 255])

# --- PARÁMETROS DE ROBUSTEZ ---
VENTANA_FRAMES = 2
UMBRAL_PARPADEO = 10
MIN_CIRCULARIDAD_PARPADEO = 0.65
MIN_CIRCULARIDAD_HSV = 0.45
KERNEL_GAUSS = (3, 3)
MIN_AREA_CONTOUR = 10

# Parámetros Kalman (pueden tunearse)
PROCESS_NOISE = 1e-2   # cov de proceso (más alto = más confianza en medición que en modelo)
MEASUREMENT_NOISE = 1e-1  # cov de medición (más alto = mediciones menos confiables)

# Parámetros de zoom dinámico suavizado
ZOOM_ALPHA = 0.2       # suavizado exponencial del centro/size (0..1)
ZOOM_MIN_SIZE = 80     # tamaño mínimo del roi en px
ZOOM_PADDING = 1.6     # multiplicador para dar algo de margen al ROI

# --------------------------------------------------------

def crear_kalman():
    """
    Crea un KalmanFilter con estado [x, y, vx, vy] y medición [x, y].
    """
    kf = cv2.KalmanFilter(4, 2, 0)  # stateDim=4, measDim=2
    # transición (dt = 1 frame)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # medición extrae x,y del estado
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Covarianzas
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE

    # Inicializaciones razonables (se reescriben cuando se inicializa realmente)
    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf

def asociar_detecciones_a_filtros(predicciones, detecciones):
    """
    Asociacion greedy por distancia mínima (compatible con NumPy 1.x).
    predicciones: list de (x,y) predichos por cada kalman
    detecciones: list de (x,y)
    Devuelve: lista de pares (idx_filtro, idx_det) asociados, lista de filtros sin detección, lista de detecciones sin filtro
    """
    if len(predicciones) == 0 or len(detecciones) == 0:
        return [], list(range(len(predicciones))), list(range(len(detecciones)))

    # construir matriz de distancias
    D = np.zeros((len(predicciones), len(detecciones)), dtype=np.float32)
    for i, p in enumerate(predicciones):
        for j, d in enumerate(detecciones):
            D[i, j] = math.hypot(p[0] - d[0], p[1] - d[1])

    assigned = []
    used_pred = set()
    used_det = set()

    # greedy: tomar la pareja mínima iterativamente
    while True:
        D_safe = np.where(np.isnan(D), np.inf, D)
        flat_min_idx = int(np.argmin(D_safe))
        i, j = divmod(flat_min_idx, D.shape[1])
        min_val = D_safe[i, j]
        if np.isinf(min_val):
            break
        assigned.append((i, j))
        used_pred.add(i)
        used_det.add(j)
        D[i, :] = np.nan
        D[:, j] = np.nan
        if len(used_pred) == len(predicciones) or len(used_det) == len(detecciones):
            break

    filtros_sin_det = [i for i in range(len(predicciones)) if i not in used_pred]
    det_sin_filtro = [j for j in range(len(detecciones)) if j not in used_det]
    return assigned, filtros_sin_det, det_sin_filtro

def obtener_roi_zoom(pts, frame_shape, prev_center_size):
    """
    Calcula ROI cuadrado (centro, size) a partir de un conjunto de puntos (lista de (x,y)),
    y aplica suavizado exponencial con prev_center_size = (cx, cy, size).
    Devuelve recorte (x1,y1,x2,y2) y nuevo prev_center_size.
    """
    h, w = frame_shape[:2]
    if len(pts) == 0:
        return None, prev_center_size

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    size_x = (maxx - minx) * ZOOM_PADDING
    size_y = (maxy - miny) * ZOOM_PADDING
    size = max(size_x, size_y, ZOOM_MIN_SIZE)

    # suavizado exponencial
    if prev_center_size is None:
        scx, scy, ssize = cx, cy, size
    else:
        scx = ZOOM_ALPHA * cx + (1 - ZOOM_ALPHA) * prev_center_size[0]
        scy = ZOOM_ALPHA * cy + (1 - ZOOM_ALPHA) * prev_center_size[1]
        ssize = ZOOM_ALPHA * size + (1 - ZOOM_ALPHA) * prev_center_size[2]

    half = ssize / 2.0
    x1 = int(max(0, scx - half))
    y1 = int(max(0, scy - half))
    x2 = int(min(w - 1, scx + half))
    y2 = int(min(h - 1, scy + half))

    return (x1, y1, x2, y2), (scx, scy, ssize)

def imprimir_datos_kalman_console(kalman_filters):
    """
    Imprime datos de estado de cada kalman en consola.
    """
    for i, kf in enumerate(kalman_filters):
        s = kf.statePost.flatten()
        print(f"KF[{i}] -> x={s[0]:.1f}, y={s[1]:.1f}, vx={s[2]:.2f}, vy={s[3]:.2f}")

def detectar_centroides_leds_hibrido_mejorado_con_kalman(ruta_video, num_leds=3):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video en {ruta_video}")
        return

    total_frames_en_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Iniciando procesamiento de video ({ancho}x{alto}) -> frames: {total_frames_en_video}")

    # buffers / estados
    frame_buffer = deque(maxlen=VENTANA_FRAMES)
    kernel_morph = np.ones((3, 3), np.uint8)

    kalman_filters = []
    kalman_initialized = False
    last_successful_locations = []

    # controles de reproducción
    frame_idx = 0
    autoplay = True   # si True avanza automáticamente; 'p' toggles pause
    exit_flag = False

    # zoom state
    prev_center_size = None

    total_frames_procesados = 0
    frames_con_deteccion_completa = 0

    while not exit_flag:
        # asegurarse de que frame_idx esté en rango
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= total_frames_en_video:
            print("Llegaste al final del video.")
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_curr_bgr = cap.read()
        if not ret:
            print(f"No se pudo leer frame {frame_idx}.")
            break

        total_frames_procesados += 1
        frame_curr = frame_curr_bgr.copy()
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(gray_curr)

        # preparar máscara HSV siempre (para mostrar)
        hsv = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, HSV_LOWER_RED_1, HSV_UPPER_RED_1)
        mask_r2 = cv2.inRange(hsv, HSV_LOWER_RED_2, HSV_UPPER_RED_2)
        mask_hsv = cv2.bitwise_or(mask_r1, mask_r2)
        mask_hsv_disp = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)

        if len(frame_buffer) < VENTANA_FRAMES:
            # mostrar ventanas
            cv2.imshow("RAW", frame_curr_bgr)
            cv2.imshow("PROCESADA", frame_curr_bgr)
            cv2.imshow("MASK_HSV", mask_hsv_disp)
            # manejar keys en modo autoplay o pause
            key = cv2.waitKey(0 if not autoplay else 30) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('n'):
                frame_idx += 1
                continue
            if key == ord('b'):
                frame_idx = max(0, frame_idx - 1)
                continue
            if key == ord('p'):
                autoplay = not autoplay
                continue
            if key == ord('r'):
                kalman_filters = []
                kalman_initialized = False
                last_successful_locations = []
                print("Reset de tracking ejecutado.")
                continue
            if autoplay:
                frame_idx += 1
            continue

        # -----------------------
        # DETECCIÓN POR PARPADEO
        # -----------------------
        blurred_curr = cv2.GaussianBlur(frame_buffer[-1], KERNEL_GAUSS, 0)
        blurred_prev = cv2.GaussianBlur(frame_buffer[-2], KERNEL_GAUSS, 0)
        diff_frame = cv2.absdiff(blurred_curr, blurred_prev)
        _, thresholded = cv2.threshold(diff_frame, UMBRAL_PARPADEO, 255, cv2.THRESH_BINARY)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        contours_parpadeo, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        led_locations_parpadeo = []
        for c in contours_parpadeo:
            area = cv2.contourArea(c)
            if area < MIN_AREA_CONTOUR:
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < MIN_CIRCULARIDAD_PARPADEO:
                continue
            mask_blob = np.zeros(diff_frame.shape, dtype=np.uint8)
            cv2.drawContours(mask_blob, [c], -1, 255, -1)
            roi_intensity = cv2.bitwise_and(diff_frame, diff_frame, mask=mask_blob)
            M = cv2.moments(roi_intensity)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                led_locations_parpadeo.append(((cX, cY), "P"))

        # -----------------------
        # MÉTODO HSV (FALLBACK)
        # -----------------------
        led_locations_hsv = []
        if len(led_locations_parpadeo) < num_leds:
            mask_hsv_proc = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
            contours_hsv, _ = cv2.findContours(mask_hsv_proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gray_curr_intensity = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2GRAY)

            for c in contours_hsv:
                area = cv2.contourArea(c)
                if area < MIN_AREA_CONTOUR:
                    continue
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity < MIN_CIRCULARIDAD_HSV:
                    continue
                mask_blob = np.zeros(gray_curr_intensity.shape, dtype=np.uint8)
                cv2.drawContours(mask_blob, [c], -1, 255, -1)
                roi_intensity = cv2.bitwise_and(gray_curr_intensity, gray_curr_intensity, mask=mask_blob)
                M = cv2.moments(roi_intensity)
                if M["m00"] != 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    led_locations_hsv.append(((cX, cY), "H"))

        # -----------------------
        # FUSIÓN DE DETECCIONES
        # -----------------------
        final_led_locations = list(led_locations_parpadeo)
        for (cX_hsv, cY_hsv), method in led_locations_hsv:
            is_duplicate = False
            for (cX_p, cY_p), _ in led_locations_parpadeo:
                if math.hypot(cX_p - cX_hsv, cY_p - cY_hsv) < 10:
                    is_duplicate = True
                    break
            if not is_duplicate and len(final_led_locations) < num_leds:
                final_led_locations.append(((cX_hsv, cY_hsv), method))

        detection_method_summary = "FALLO"

        # -----------------------
        # TRACKING: KALMAN (manejo correcto de predict/correct)
        # -----------------------
        if not kalman_initialized and len(final_led_locations) == num_leds:
            # inicializar kalman con las detecciones completas
            kalman_filters = []
            for (cX, cY), _ in final_led_locations:
                kf = crear_kalman()
                kf.statePost = np.array([[np.float32(cX)], [np.float32(cY)], [0.0], [0.0]], dtype=np.float32)
                kalman_filters.append(kf)
            kalman_initialized = True
            last_successful_locations = final_led_locations.copy()
            frames_con_deteccion_completa += 1
            detection_method_summary = "Inicialización Kalman"

        elif kalman_initialized:
            # 1) Predict una vez por filtro y guardar predicciones sin alterar más
            predictions = []
            for kf in kalman_filters:
                pred_state = kf.predict()  # predict actualiza statePre internamente
                px = float(pred_state[0, 0])
                py = float(pred_state[1, 0])
                predictions.append((px, py, pred_state))

            # 2) Asociar detecciones actuales a predicciones
            detecciones_coords = [(cX, cY) for ((cX, cY), _) in final_led_locations]
            assigned, filtros_sin_det, det_sin_filtro = asociar_detecciones_a_filtros(
                [(p[0], p[1]) for p in predictions], detecciones_coords
            )

            # 3) Aplicar corrections a filtros asignados (correct usa la medición)
            filtros_corregidos = set()
            for idx_filtro, idx_det in assigned:
                meas = np.array([[np.float32(detecciones_coords[idx_det][0])],
                                 [np.float32(detecciones_coords[idx_det][1])]], dtype=np.float32)
                kalman_filters[idx_filtro].correct(meas)
                filtros_corregidos.add(idx_filtro)

            # 4) Construir final_led_locations a partir de estadoPost si corregido, sino del pred almacenado
            new_final_locations = []
            for i, kf in enumerate(kalman_filters):
                if i in filtros_corregidos:
                    state = kf.statePost
                    tag = "C"  # corregido
                else:
                    # usar la prediccion guardada (predictions[i][2])
                    pred_state = predictions[i][2]
                    state = pred_state
                    tag = "K"  # predicción (no corregida)
                x = float(state[0, 0])
                y = float(state[1, 0])
                new_final_locations.append(((x, y), tag))
            final_led_locations = new_final_locations

            # contar frames con detección real si hubieron al menos num_leds detecciones
            if len(detecciones_coords) >= num_leds:
                frames_con_deteccion_completa += 1
                last_successful_locations = [((float(kf.statePost[0, 0]), float(kf.statePost[1, 0])), "K")
                                             for kf in kalman_filters]

            detection_method_summary = "Kalman"

        else:
            # Kalman no inicializado y deteccion incompleta
            if len(final_led_locations) == num_leds:
                last_successful_locations = final_led_locations.copy()
                frames_con_deteccion_completa += 1
                detection_method_summary = "Fusión"
            elif len(last_successful_locations) == num_leds:
                final_led_locations = last_successful_locations.copy()
                detection_method_summary = "MEMORIA P."
            else:
                detection_method_summary = "FALLO INICIAL"

        # -----------------------
        # DIBUJO Y OVERLAYS
        # -----------------------
        proc_vis = frame_curr_bgr.copy()

        # dibujar leds y etiquetas
        for i, ((cX, cY), method_tag) in enumerate(final_led_locations):
            if method_tag == "M":
                color = (0, 165, 255)  # naranja para memoria
                tag = "M"
            elif method_tag in ("P", "C"):  # Parpadeo o corregido por Kalman
                color = (0, 255, 0)
                tag = method_tag
            elif method_tag == "H":
                color = (255, 0, 0)
                tag = "H"
            elif method_tag == "K":  # Predicción Kalman
                color = (0, 200, 200)
                tag = "K"
            else:
                color = (200, 200, 200)
                tag = method_tag

            cv2.circle(proc_vis, (int(round(cX)), int(round(cY))), 4, color, -1)
            cv2.putText(proc_vis, f"L{i+1} ({tag})", (int(round(cX)) + 8, int(round(cY)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Mostrar predicciones guardadas (cruces amarillas) y datos Kalman
        if kalman_initialized:
            # mostrar datos Kalman en la imagen
            for i, kf in enumerate(kalman_filters):
                # usar statePost (si fue corregido) o statePre (si no)
                s_post = kf.statePost.flatten()
                px = int(round(float(s_post[0])))
                py = int(round(float(s_post[1])))
                vx = float(s_post[2])
                vy = float(s_post[3])
                cv2.drawMarker(proc_vis, (px, py), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                cv2.putText(proc_vis, f"KF{i}: x={px},y={py} v=({vx:.1f},{vy:.1f})", (10, 60 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # estado y estadísticas
        status_text = f"Frame {frame_idx+1}/{total_frames_en_video} | Det: {len(final_led_locations)}/{num_leds} | M: {detection_method_summary}"
        cv2.putText(proc_vis, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # -----------------------
        # ZOOM dinámico suavizado (opción 3 pedida)
        # -----------------------
        pts_for_zoom = [ (x,y) for ((x,y), _) in final_led_locations ]
        roi_rect, prev_center_size = obtener_roi_zoom(pts_for_zoom, frame_curr_bgr.shape, prev_center_size)
        zoom_vis = None
        if roi_rect is not None:
            x1, y1, x2, y2 = roi_rect
            zoom_vis = frame_curr_bgr[y1:y2, x1:x2].copy()
            if zoom_vis.size == 0:
                zoom_vis = None

        # -----------------------
        # MOSTRAR VENTANAS: RAW, PROCESADA, MASK_HSV, ZOOM
        # -----------------------
        cv2.imshow("RAW", frame_curr_bgr)
        cv2.imshow("PROCESADA", proc_vis)
        cv2.imshow("MASK_HSV", mask_hsv_disp)
        if zoom_vis is not None:
            # escalar zoom a ventana fija para mejor visual
            zoom_show = cv2.resize(zoom_vis, (300, 300), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("ZOOM", zoom_show)
        else:
            # mostrar imagen negra si no hay zoom
            cv2.imshow("ZOOM", np.zeros((300,300,3), dtype=np.uint8))

        # imprimir datos Kalman en consola si estamos en pausa (o cada N frames, aquí cada frame)
        if not autoplay:
            if kalman_initialized:
                print(f"\n--- FRAME {frame_idx+1} ---")
                imprimir_datos_kalman_console(kalman_filters)

        # -----------------------
        # TECLAS / CONTROLES
        # -----------------------
        # modo espera: si autoplay True, espera breve; si False, espera indefinidamente por tecla
        key = cv2.waitKey(0 if not autoplay else 30) & 0xFF

        if key == 27 or key == ord('q'):
            # ESC o q => salir
            exit_flag = True
            break
        elif key == ord('n'):
            frame_idx += 1
        elif key == ord('b'):
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('p'):
            # toggle pausa automática
            autoplay = not autoplay
            print("Autoplay:", autoplay)
        elif key == ord('r'):
            # reset tracking (reinicia kalman)
            kalman_filters = []
            kalman_initialized = False
            last_successful_locations = []
            prev_center_size = None
            print("Reset de tracking ejecutado.")
        else:
            # ninguna tecla de control: si autoplay avanza, si pauso se queda (ya manejado por waitKey param)
            if autoplay:
                frame_idx += 1
            else:
                # en pausa y tecla desconocida -> seguir en la misma imagen hasta que apriete 'n' o 'b' u otra
                pass

    # fin loop
    cap.release()
    cv2.destroyAllWindows()

    # REPORTE
    if total_frames_procesados > 0:
        porcentaje_precision = (frames_con_deteccion_completa / total_frames_procesados) * 100
        print("\n" + "="*50)
        print("         REPORTE DE PRECISIÓN DE DETECCIÓN ")
        print("="*50)
        print(f"Frames Totales en Video:    {total_frames_en_video}")
        print(f"Frames Procesados:          {total_frames_procesados}")
        print("-" * 50)
        print(f"Frames con {num_leds} LEDs detectados REALMENTE: {frames_con_deteccion_completa}")
        print(f"Tasa de Detección REAL:     {porcentaje_precision:.2f}%")
        print("="*50 + "\n")
    else:
        print("No se procesaron frames.")


# --- USAR EL CÓDIGO ---
if __name__ == '__main__':
    VIDEO_PATH = '../Generador de videos/simulacion_leds_vision.mp4'
    # VIDEO_PATH = '../Generador de videos/simulacion_leds_30fps.mp4'
    detectar_centroides_leds_hibrido_mejorado_con_kalman(
        ruta_video=VIDEO_PATH,
        num_leds=3
    )
