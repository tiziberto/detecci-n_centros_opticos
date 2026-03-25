import cv2
import numpy as np
from collections import deque

# RANGOS HSV para LED ROJO (Se necesitan dos rangos para el color rojo)
HSV_LOWER_RED_1 = np.array([0, 50, 150]) 
HSV_UPPER_RED_1 = np.array([10, 255, 255])
HSV_LOWER_RED_2 = np.array([170, 50, 150])
HSV_UPPER_RED_2 = np.array([180, 255, 255])

VENTANA_FRAMES = 2

# --- CLASE PARA EL SEGUIMIENTO INDIVIDUAL DE UN LED (FILTRO DE KALMAN) ---
class LedTracker:
    def __init__(self, x_init, y_init, dt=1.0/30.0):
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Matriz de Transición de Estado (A) - Depende de dt
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                            [0, 1, 0, dt],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)

        # ✅ AJUSTE CLAVE 1: AUMENTAR EL RUIDO DE PROCESO (Q)
        # Esto hace que el modelo de movimiento sea menos confiable, haciendo que
        # la predicción se actualice más rápido ante nuevas mediciones.
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 1e-2 # Aumentado de 1e-4 a 1e-2
        
        # ✅ AJUSTE CLAVE 2: REDUCIR EL RUIDO DE MEDICIÓN (R)
        # Esto hace que el filtro confíe MÁS en la detección del frame actual,
        # corrigiendo rápidamente el retraso.
        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.1 # Reducido de 1.0 a 0.1
        
        # Vector de Estado Inicial
        self.kf.statePost = np.array([[x_init], [y_init], [0.0], [0.0]], np.float32) 

        self.last_known_pos = (x_init, y_init)
        self.missed_detections = 0
        self.id = id(self)
        
    def predict(self):
        predicted = self.kf.predict()
        x_pred, y_pred = predicted[0, 0], predicted[1, 0]
        return (x_pred, y_pred)

    def update(self, measurement):
        measured = np.array([[np.float32(measurement[0])], 
                             [np.float32(measurement[1])]])
        self.kf.correct(measured)
        
        self.last_known_pos = measurement
        self.missed_detections = 0
        return self.last_known_pos
    
    def get_current_position(self):
        x = self.kf.statePost[0, 0]
        y = self.kf.statePost[1, 0]
        return (x, y)
    
    def get_velocity(self):
        vx = self.kf.statePost[2, 0]
        vy = self.kf.statePost[3, 0]
        return (vx, vy)


def asociar_detecciones_con_trackers(detecciones, trackers, max_dist=50):
    """
    Asocia las detecciones recién encontradas (centroides) con los trackers existentes
    utilizando la distancia euclidiana a la posición PREDICHA del tracker.
    """
    if not detecciones:
        return {}, [] 

    # 1. Obtener posiciones predichas
    predicciones = [t.predict() for t in trackers]
    
    # 2. Calcular matriz de costo (distancia)
    num_det = len(detecciones)
    num_trk = len(trackers)
    cost_matrix = np.full((num_det, num_trk), np.inf)

    for i, (det_x, det_y) in enumerate(detecciones):
        for j, (trk_pred_x, trk_pred_y) in enumerate(predicciones):
            dist = np.sqrt((det_x - trk_pred_x)**2 + (det_y - trk_pred_y)**2)
            if dist < max_dist:
                cost_matrix[i, j] = dist

    # 3. Asignación (Greedy simple)
    assignments = {} 
    unassigned_detections = list(range(num_det)) # <-- Nombre correcto
    assigned_trackers = set()

    for _ in range(min(num_det, num_trk)):
        min_cost = np.inf
        best_det = -1
        best_trk = -1
        
        for i in unassigned_detections:
            for j in range(num_trk):
                if j not in assigned_trackers and cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_det = i
                    best_trk = j
        
        if best_det != -1:
            assignments[best_trk] = best_det
            assigned_trackers.add(best_trk)
            unassigned_detections.remove(best_det)
        else:
            break
    
    # ✅ CORRECCIÓN FINAL: Usar el nombre de variable correcto: unassigned_detections
    return assignments, unassigned_detections

def detectar_centroides_leds_con_prediccion(ruta_video, num_leds=3, umbral=30, max_dist_asociacion=75, max_missed_frames=10):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video en {ruta_video}")
        return

    # --- INICIALIZACIÓN ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0 
    
    trackers = [] 
    
    total_frames_procesados = 0
    frames_con_deteccion_completa = 0
    total_frames_en_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame_prev = cap.read()
    if not ret:
        print("Error: Video vacío o no se pudo leer el primer frame.")
        return
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Iniciando procesamiento de video ({ancho}x{alto}, FPS: {fps:.2f}). Presiona 'q' para salir.")
    
    frame_buffer = [gray_prev] * VENTANA_FRAMES 
    kernel = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, frame_curr = cap.read()
        if not ret: break

        total_frames_procesados += 1

        frame_curr_bgr = frame_curr.copy() 
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        
        frame_buffer.append(gray_curr)
        frame_buffer.pop(0)

        if len(frame_buffer) < VENTANA_FRAMES:
            cv2.imshow('Deteccion de LEDS (Frame Actual)', frame_curr_bgr)
            cv2.waitKey(1)
            continue

        # -------------------------------------------------------------
        # 1. 🔍 DETECCIÓN HÍBRIDA (Observaciones)
        # -------------------------------------------------------------
        
        led_locations_obs = [] 
        
        # Intento Principal: Parpadeo
        diff_frame = cv2.absdiff(frame_buffer[-1], frame_buffer[-2])
        _, thresholded = cv2.threshold(diff_frame, umbral, 255, cv2.THRESH_BINARY)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours_principal, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_principal:
            if cv2.contourArea(c) < 5: continue
            mask_blob = np.zeros(diff_frame.shape, dtype=np.uint8)
            cv2.drawContours(mask_blob, [c], -1, 255, -1)
            roi_intensity = cv2.bitwise_and(diff_frame, diff_frame, mask=mask_blob)
            M = cv2.moments(roi_intensity) 
            
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                led_locations_obs.append(((cX, cY), "Parpadeo"))

        num_found_principal = len(led_locations_obs)

        # Fallback: Segmentación HSV
        if num_found_principal < num_leds:
            hsv = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2HSV)
            mask_r1 = cv2.inRange(hsv, HSV_LOWER_RED_1, HSV_UPPER_RED_1)
            mask_r2 = cv2.inRange(hsv, HSV_LOWER_RED_2, HSV_UPPER_RED_2)
            mask_hsv = cv2.bitwise_or(mask_r1, mask_r2)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours_hsv, _ = cv2.findContours(mask_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours_hsv:
                if cv2.contourArea(c) > 10: 
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]
                        
                        is_duplicate = False
                        for (px, py), _ in led_locations_obs:
                             if np.sqrt((cX-px)**2 + (cY-py)**2) < 20:
                                 is_duplicate = True
                                 break
                        if not is_duplicate and len(led_locations_obs) < num_leds:
                            led_locations_obs.append(((cX, cY), "HSV"))


        # -------------------------------------------------------------
        # 2. 🔗 ASOCIACIÓN Y SEGUIMIENTO (PREDICCIÓN + CORRECCIÓN)
        # -------------------------------------------------------------
        
        current_detections = [loc for loc, method in led_locations_obs]
        
        if trackers:
            # 2.1. Asociación
            assignments, unassigned_detections_idx = asociar_detecciones_con_trackers(
                current_detections, trackers, max_dist=max_dist_asociacion
            )
            
            # 2.2. Actualizar (Corregir) trackers existentes
            for trk_idx, det_idx in assignments.items():
                tracker = trackers[trk_idx]
                location = current_detections[det_idx]
                tracker.update(location)
            
            # 2.3. Manejar trackers sin detección (solo Predicción)
            for trk_idx in range(len(trackers)):
                if trk_idx not in assignments:
                    tracker = trackers[trk_idx]
                    # La predicción ya se hizo en asociar_detecciones_con_trackers
                    tracker.missed_detections += 1
                    
            # 2.4. Eliminar trackers perdidos
            trackers = [t for t in trackers if t.missed_detections < max_missed_frames]
            
            # 2.5. Inicializar nuevos trackers
            for det_idx in unassigned_detections_idx:
                 if len(trackers) < num_leds: 
                    new_tracker = LedTracker(current_detections[det_idx][0], current_detections[det_idx][1], dt)
                    trackers.append(new_tracker)

        # 2.6. Inicialización (Primer frame)
        else:
            for i, (location, _) in enumerate(led_locations_obs):
                 if len(trackers) < num_leds:
                    new_tracker = LedTracker(location[0], location[1], dt)
                    trackers.append(new_tracker)
        
        if len(trackers) == num_leds:
             frames_con_deteccion_completa += 1
        
        # -------------------------------------------------------------
        # 3. 📊 DIBUJO DE RESULTADOS
        # -------------------------------------------------------------
        
        # 
        
        for i, tracker in enumerate(trackers):
            cX, cY = tracker.get_current_position()
            vx, vy = tracker.get_velocity()
            
            is_predicted = tracker.missed_detections > 0
            
            color = (0, 255, 0) if not is_predicted else (0, 0, 255) 
            tag = "C" if not is_predicted else "P" 
            
            cv2.circle(frame_curr_bgr, (int(cX), int(cY)), 8, color, 2)
            cv2.circle(frame_curr_bgr, (int(cX), int(cY)), 3, color, -1)
            
            # Dibujar vector de velocidad (dirección y magnitud)
            end_x = int(cX + vx * 5)
            end_y = int(cY + vy * 5)
            cv2.arrowedLine(frame_curr_bgr, (int(cX), int(cY)), (end_x, end_y), color, 2, tipLength=0.3)
            
            status_text = f"L{i+1} ({tag}): V({vx:.1f}, {vy:.1f})"
            cv2.putText(frame_curr_bgr, status_text, (int(cX) + 15, int(cY) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        status_text = f"Trackers Activos: {len(trackers)}/{num_leds} | Detecciones Observadas: {len(current_detections)}"
        cv2.putText(frame_curr_bgr, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Deteccion de LEDS (Con Prediccion de Kalman)', frame_curr_bgr)
        
        gray_prev = gray_curr.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    # --- REPORTE DE ESTADÍSTICAS ---
    if total_frames_procesados > 0:
        porcentaje_precision = (frames_con_deteccion_completa / total_frames_procesados) * 100
        
        print("\n" + "="*50)
        print("         ✅ REPORTE DE PRECISIÓN DE SEGUIMIENTO ✅")
        print("="*50)
        print(f"Frames Totales en Video:    {total_frames_en_video}")
        print(f"Frames Procesados:          {total_frames_procesados}")
        print("-" * 50)
        print(f"Frames con 3 LEDs seguidos: {frames_con_deteccion_completa}")
        print(f"Tasa de Seguimiento (100%): {porcentaje_precision:.2f}%")
        print("="*50 + "\n")
    else:
        print("No se procesaron frames.")


# --- USAR EL CÓDIGO ---

if __name__ == '__main__':
    # Modifica la ruta de tu video
    VIDEO_PATH = '../Generador de videos/simulacion_leds_30fps.mp4' 
 
    
    
    detectar_centroides_leds_con_prediccion(
        ruta_video=VIDEO_PATH,
        num_leds=3,
        umbral=30,
        # Se aumentó max_dist_asociacion a 75 para tolerar movimientos rápidos
        max_dist_asociacion=75, 
        max_missed_frames=10      
    )