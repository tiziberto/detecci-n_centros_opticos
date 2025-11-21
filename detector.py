import cv2
import numpy as np
import math

# RANGOS HSV para LED ROJO (ligeramente ampliados en brillo para robustez)
# (H: Tono, S: Saturación, V: Brillo)
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

# ----------------------------------------

def detectar_centroides_leds_hibrido_mejorado(ruta_video, num_leds=3):
    """
    Implementa la detección híbrida con centroides ponderados y una memoria
    PERSISTENTE: si no detecta los 3 LEDs, usa la última posición válida.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video en {ruta_video}")
        return

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
    print(f"Iniciando procesamiento de video ({ancho}x{alto}). Presiona 'q' para salir.")
    
    frame_buffer = [gray_prev] * VENTANA_FRAMES 
    kernel_morph = np.ones((3, 3), np.uint8)

    # --- VARIABLE DE ESTADO PARA LA MEMORIA PERSISTENTE ---
    # Almacena las ubicaciones del ÚLTIMO frame que tuvo detección completa.
    last_successful_locations = [] 
    # ------------------------------------------------------

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
        # MÉTODO 1: Parpadeo + CENTROIDE PONDERADO 
        # -------------------------------------------------------------
        
        blurred_curr = cv2.GaussianBlur(frame_buffer[-1], KERNEL_GAUSS, 0)
        blurred_prev = cv2.GaussianBlur(frame_buffer[-2], KERNEL_GAUSS, 0)
        
        diff_frame = cv2.absdiff(blurred_curr, blurred_prev)
        _, thresholded = cv2.threshold(diff_frame, UMBRAL_PARPADEO, 255, cv2.THRESH_BINARY)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        
        contours_parpadeo, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        led_locations_parpadeo = []
        
        for i, c in enumerate(contours_parpadeo):
            area = cv2.contourArea(c)
            if area < MIN_AREA_CONTOUR: continue
            
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity < MIN_CIRCULARIDAD_PARPADEO: continue

            mask_blob = np.zeros(diff_frame.shape, dtype=np.uint8)
            cv2.drawContours(mask_blob, [c], -1, 255, -1)
            roi_intensity = cv2.bitwise_and(diff_frame, diff_frame, mask=mask_blob)

            M = cv2.moments(roi_intensity) 
            
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                led_locations_parpadeo.append(((cX, cY), "P"))
            
        
        # -------------------------------------------------------------
        # MÉTODO 2: Segmentación HSV (FALLBACK / RELLENO) 
        # -------------------------------------------------------------
        
        led_locations_hsv = [] 
        if len(led_locations_parpadeo) < num_leds:
            
            hsv = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2HSV)
            mask_r1 = cv2.inRange(hsv, HSV_LOWER_RED_1, HSV_UPPER_RED_1)
            mask_r2 = cv2.inRange(hsv, HSV_LOWER_RED_2, HSV_UPPER_RED_2)
            mask_hsv = cv2.bitwise_or(mask_r1, mask_r2)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
            
            contours_hsv, _ = cv2.findContours(mask_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            gray_curr_intensity = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2GRAY) 

            for i, c in enumerate(contours_hsv):
                area = cv2.contourArea(c)
                if area < MIN_AREA_CONTOUR: continue

                perimeter = cv2.arcLength(c, True)
                circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
                if circularity < MIN_CIRCULARIDAD_HSV: continue

                mask_blob = np.zeros(gray_curr_intensity.shape, dtype=np.uint8)
                cv2.drawContours(mask_blob, [c], -1, 255, -1)
                roi_intensity = cv2.bitwise_and(gray_curr_intensity, gray_curr_intensity, mask=mask_blob)
                
                M = cv2.moments(roi_intensity)
                if M["m00"] != 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    led_locations_hsv.append(((cX, cY), "H"))

        # -------------------------------------------------------------
        # FUSIÓN Y SELECCIÓN FINAL DE CENTROIDES 
        # -------------------------------------------------------------
        
        final_led_locations = list(led_locations_parpadeo) 
        
        for (cX_hsv, cY_hsv), method in led_locations_hsv:
            
            is_duplicate = False
            for (cX_p, cY_p), _ in led_locations_parpadeo:
                distance = math.sqrt((cX_p - cX_hsv)**2 + (cY_p - cY_hsv)**2)
                if distance < 10: 
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(final_led_locations) < num_leds:
                final_led_locations.append(((cX_hsv, cY_hsv), method))


        # -------------------------------------------------------------
        # APLICACIÓN DE LA LÓGICA DE MEMORIA PERSISTENTE
        # -------------------------------------------------------------
        
        detection_method_summary = "FALLO" # Inicialización por defecto
        
        # Caso 1: Detección COMPLETA y exitosa
        if len(final_led_locations) == num_leds:
            frames_con_deteccion_completa += 1
            # ACTUALIZAR la memoria. Esto ocurre solo si la detección es completa.
            last_successful_locations = final_led_locations.copy()
            detection_method_summary = "Fusión" if len(final_led_locations) > len(led_locations_parpadeo) else "Parpadeo"

        # Caso 2: Detección FALLIDA, pero hay una memoria válida (¡Persistente!)
        elif len(final_led_locations) < num_leds and len(last_successful_locations) == num_leds:
            
            # USAR la memoria. Sobrescribe la detección incompleta con la última ubicación válida.
            final_led_locations = last_successful_locations.copy()
            detection_method_summary = "MEMORIA P." # P. de Persistente
            
        # Caso 3: Fallo completo sin memoria válida (solo en los primeros frames)
        else:
            detection_method_summary = "FALLO INICIAL"
            
        # -------------------------------------------------------------
        # DIBUJO DE RESULTADOS
        # -------------------------------------------------------------
        
        for i, ((cX, cY), method_tag) in enumerate(final_led_locations):
            # Lógica de color y etiqueta para la visualización
            if detection_method_summary == "MEMORIA P.":
                 color = (0, 165, 255) # Naranja/Amarillo para la memoria
                 method_tag = "M" 
            elif method_tag == "P":
                 color = (0, 255, 0) # Verde para Parpadeo
            else: # method_tag == "H"
                 color = (255, 0, 0) # Azul para HSV
            
            cv2.circle(frame_curr_bgr, (int(cX), int(cY)), 4, color, -1)
            cv2.putText(frame_curr_bgr, f"L{i+1} ({method_tag})", (int(cX) + 8, int(cY) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        status_text = f"Det. LEDs: {len(final_led_locations)}/{num_leds} | M: {detection_method_summary}"
        cv2.putText(frame_curr_bgr, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Deteccion de LEDS (Frame Actual)', frame_curr_bgr)
        
        gray_prev = gray_curr.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    # --- REPORTE DE ESTADÍSTICAS ---
    if total_frames_procesados > 0:
        # Nota: 'frames_con_deteccion_completa' ahora solo cuenta los frames donde el algoritmo
        # REALMENTE detectó los 3 LEDs (sin memoria), lo cual es la métrica de precisión más honesta.
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
    VIDEO_PATH = '../Generador de videos/simulacion_leds_30fps.mp4' 
 
    
    detectar_centroides_leds_hibrido_mejorado(
        ruta_video=VIDEO_PATH,
        num_leds=3
    )
