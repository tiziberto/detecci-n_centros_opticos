import cv2
import numpy as np
import time

# --- Parámetros del Video ---
FPS = 30
DURACION_SEGUNDOS = 10
TOTAL_FRAMES = FPS * DURACION_SEGUNDOS

# --- Parámetros de Simulación ---
ANCHO, ALTO = 640, 480

MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 220
PERIODO_FONDO_FRAMES = FPS * 5 

# --- Placa y LEDs ---
LARGO_PLACA = 80
ANCHO_PLACA = 40
RADIO_LED = 5

# Color Violeta (BGR: Azul, Verde, Rojo)
COLOR_LED_ON = (255, 0, 255) 
COLOR_LED_OFF = (60, 60, 60) 

Y_POS = 0

led_relativos = np.array([
    [-15, Y_POS],  # LED 1: Izquierda
    [  0, Y_POS],  # LED 2: Centro
    [ 15, Y_POS]   # LED 3: Derecha
], dtype=np.float32)


# --- Ángulo de visión (0 = frontal, 1 = casi perfil) ---
ANGULO_VISION = 0.5   # <<< podés cambiar este valor

# --- Función de transformación 2D + perspectiva ---
def transformar_punto_con_vision(p_rel, cx, cy, angulo_rot, vision):
    # rotación normal
    R = np.array([
        [np.cos(angulo_rot), -np.sin(angulo_rot)],
        [np.sin(angulo_rot),  np.cos(angulo_rot)]
    ])
    p = np.dot(R, p_rel.T).T

    # proyección tipo perspectiva
    px = p[0]
    py = p[1]

    # Efecto de visión
    py2 = py * (1 - vision) + px * vision * 0.35

    # Trasladar a la posición del centro
    return int(cx + px), int(cy + py2)


# --- Inicialización del video ---
NOMBRE_VIDEO = 'simulacion_leds_violeta_15hz.mp4' # Nombre actualizado
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(NOMBRE_VIDEO, fourcc, FPS, (ANCHO, ALTO))

# --- Generación del video ---
print("Generando video con LEDs en línea recta, Violeta a 15 Hz (1 frame ON / 1 frame OFF)...")

inicio = time.time()

for frame_idx in range(TOTAL_FRAMES):

    # Fondo dinámico
    t_norm = (frame_idx % PERIODO_FONDO_FRAMES) / PERIODO_FONDO_FRAMES
    sin_wave = (np.sin(t_norm * 2 * np.pi) + 1) / 2
    brillo = int(MIN_BRIGHTNESS + (MAX_BRIGHTNESS - MIN_BRIGHTNESS) * sin_wave)
    frame = np.full((ALTO, ANCHO, 3), (brillo, brillo, brillo), dtype=np.uint8)

    # Movimiento placa
    cx = 300 + 150 * np.sin(frame_idx * 0.05)
    cy = 200 + 100 * np.cos(frame_idx * 0.07)
    ang = frame_idx * 0.04

    # ==========================================================
    # CAMBIO: Lógica de Parpadeo 1 frame ON / 1 frame OFF (15 Hz)
    # Si el índice del frame es par (0, 2, 4...), está encendido.
    # Si el índice del frame es impar (1, 3, 5...), está apagado.
    led_is_on = (frame_idx % 2) == 0 
    # ==========================================================
    color_led = COLOR_LED_ON if led_is_on else COLOR_LED_OFF

    # Dibujar LEDs con efecto de visión
    for p_rel in led_relativos:
        lx, ly = transformar_punto_con_vision(p_rel, cx, cy, ang, ANGULO_VISION)
        cv2.circle(frame, (lx, ly), RADIO_LED, color_led, -1)

    out.write(frame)

out.release()
tiempo_total = time.time() - inicio

print("------------------------------------------------")
print(f"Video generado: {NOMBRE_VIDEO}")
print(f"Frecuencia de parpadeo: 15 Hz")
print(f"Ángulo de visión usado: {ANGULO_VISION}")
print(f"Tiempo total: {tiempo_total:.2f} segundos")
print("------------------------------------------------")