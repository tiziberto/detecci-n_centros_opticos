import cv2
import numpy as np

# RANGOS HSV para LED ROJO (Se necesitan dos rangos para el color rojo)
HSV_LOWER_RED_1 = np.array([0, 50, 150]) 
HSV_UPPER_RED_1 = np.array([10, 255, 255])
HSV_LOWER_RED_2 = np.array([170, 50, 150])
HSV_UPPER_RED_2 = np.array([180, 255, 255])

# Definición del ciclo del LED para el método robusto (Aproximación por buffer)
VENTANA_FRAMES = 2

def detectar_centroides_leds_hibrido_mejorado(ruta_video, num_leds=3, umbral=30, radio_eliminacion=25):
    """
    Combina la Detección de Parpadeo Sincronizado con Centroide Ponderado
    (Precisión Subpíxel Reforzada) y un método de reserva (HSV).
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video en {ruta_video}")
        return

    # --- CONTADORES DE ESTADÍSTICAS ---
    total_frames_procesados = 0
    frames_con_deteccion_completa = 0 # Contador para 100% de precisión (3 de 3 LEDs detectados)
    total_frames_en_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # -----------------------------------

    ret, frame_prev = cap.read()
    if not ret:
        print("Error: Video vacío o no se pudo leer el primer frame.")
        return
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Iniciando procesamiento de video ({ancho}x{alto}). Presiona 'q' para salir.")
    
    frame_buffer = [gray_prev] * VENTANA_FRAMES 
    kernel = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, frame_curr = cap.read()
        if not ret: break

        total_frames_procesados += 1 # Contar cada frame procesado

        frame_curr_bgr = frame_curr.copy() 
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        
        frame_buffer.append(gray_curr)
        frame_buffer.pop(0)

        if len(frame_buffer) < VENTANA_FRAMES:
            cv2.imshow('Deteccion de LEDS (Frame Actual)', frame_curr_bgr)
            cv2.waitKey(1)
            continue

        # -------------------------------------------------------------
        # 🚀 MÉTODO PRINCIPAL: Parpadeo + CENTROIDE PONDERADO
        # -------------------------------------------------------------
        
        diff_frame = cv2.absdiff(frame_buffer[-1], frame_buffer[-2]) # F_n - F_n-1
        _, thresholded = cv2.threshold(diff_frame, umbral, 255, cv2.THRESH_BINARY)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        led_locations = []
        
        for i, c in enumerate(contours):
            if i >= num_leds: break

            if cv2.contourArea(c) < 5: 
                continue

            mask_blob = np.zeros(diff_frame.shape, dtype=np.uint8)
            cv2.drawContours(mask_blob, [c], -1, 255, -1)
            
            roi_intensity = cv2.bitwise_and(diff_frame, diff_frame, mask=mask_blob)

            M = cv2.moments(roi_intensity) 
            
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                led_locations.append((cX, cY))
            
        num_found_principal = len(led_locations)
        detection_method = "Parpadeo"

        # -------------------------------------------------------------
        # 🚨 MÉTODO DE RESERVA (FALLBACK): Segmentación HSV
        # -------------------------------------------------------------
        
        if num_found_principal < num_leds:
            detection_method = "HSV Fallback"
            
            hsv = cv2.cvtColor(frame_curr_bgr, cv2.COLOR_BGR2HSV)
            mask_r1 = cv2.inRange(hsv, HSV_LOWER_RED_1, HSV_UPPER_RED_1)
            mask_r2 = cv2.inRange(hsv, HSV_LOWER_RED_2, HSV_UPPER_RED_2)
            mask_hsv = cv2.bitwise_or(mask_r1, mask_r2)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(mask_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            led_locations = [] 
            for i, c in enumerate(contours):
                if cv2.contourArea(c) > 10 and i < num_leds: 
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]
                        led_locations.append((cX, cY))

        # --- ACTUALIZAR CONTEO DE PRECISIÓN ---
        if len(led_locations) == num_leds:
            frames_con_deteccion_completa += 1
        # ---------------------------------------
        
        # -------------------------------------------------------------
        # 📊 DIBUJO DE RESULTADOS
        # -------------------------------------------------------------

        for i, (cX, cY) in enumerate(led_locations):
            color = (0, 255, 0) if detection_method == "Parpadeo" else (255, 0, 0)
            method_tag = "P" if detection_method == "Parpadeo" else "H"
            
            cv2.circle(frame_curr_bgr, (int(cX), int(cY)), 5, color, -1)
            cv2.putText(frame_curr_bgr, f"L{i+1} ({method_tag})", (int(cX) + 10, int(cY) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        status_text = f"Det. LEDs: {len(led_locations)}/{num_leds} | M: {detection_method}"
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
        porcentaje_precision = (frames_con_deteccion_completa / total_frames_procesados) * 100
        
        print("\n" + "="*50)
        print("         ✅ REPORTE DE PRECISIÓN DE DETECCIÓN ✅")
        print("="*50)
        print(f"Frames Totales en Video:    {total_frames_en_video}")
        print(f"Frames Procesados:          {total_frames_procesados}")
        print("-" * 50)
        print(f"Frames con 3 LEDs detectados: {frames_con_deteccion_completa}")
        print(f"Tasa de Detección (100%):  {porcentaje_precision:.2f}%")
        print("="*50 + "\n")
    else:
        print("No se procesaron frames.")


# --- USAR EL CÓDIGO ---

if __name__ == '__main__':
    # La ruta de tu video
    #VIDEO_PATH = '../Generador de videos/simulacion_leds_60fps.mp4' 
    #VIDEO_PATH = '../Generador de videos/simulacion_leds_120fps.mp4' 
    # VIDEO_PATH = '../Generador de videos/simulacion_leds_ciclo_3on3off.mp4' 
    # VIDEO_PATH = '../Generador de videos/simulacion_leds_parpadeo.mp4' 
    # VIDEO_PATH = '../Generador de videos/simulacion_leds_fondo_dinamico.mp4' 
    # VIDEO_PATH = '../Generador de videos/simulacion_leds_fondo_suave.mp4'
    VIDEO_PATH = '../Generador de videos/simulacion_leds_30fps.mp4' 
 
    
    
    detectar_centroides_leds_hibrido_mejorado(
        ruta_video=VIDEO_PATH,
        num_leds=3,
        umbral=30,
        radio_eliminacion=25
    )
