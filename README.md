# 🔴 Detección de Centros Ópticos de LEDs

Sistema de visión por computadora para detectar y rastrear LEDs en tiempo real utilizando OpenCV y Filtro de Kalman.

---

## 📁 Estructura del Proyecto

```
Deteccion Centros Opticos/
├── Detector/
│   ├── Detector_v1.0.py          # Detección híbrida básica (Parpadeo + HSV)
│   ├── Detector_v1.1.py          # Fusión de métodos + filtro circularidad
│   ├── Detector_v1.2.py          # Memoria persistente de última posición
│   ├── Detector_v1.3.py          # Kalman + zoom dinámico + controles interactivos
│   └── Detector_v1.4.py          # Versión final optimizada con Kalman
│
└── Generador de videos/
    ├── generador.py              # Script para generar videos de prueba
    ├── simulacion_leds_30fps.mp4
    ├── simulacion_leds_60fps.mp4
    ├── simulacion_leds_120fps.mp4
    ├── simulacion_leds_parpadeo.mp4
    ├── simulacion_leds_fondo_dinamico.mp4
    ├── simulacion_leds_fondo_suave.mp4
    ├── simulacion_leds_ciclo_3on3off.mp4
    ├── simulacion_leds_violeta_1hz.mp4
    ├── simulacion_leds_violeta_15hz.mp4
    ├── simulacion_leds_vision.mp4
    └── video_real.mp4
```

---

## ⚙️ Requisitos

### Dependencias

```bash
pip install opencv-python numpy
```

### Versiones probadas
- Python 3.8+
- OpenCV 4.5+
- NumPy 1.20+

---

## 🚀 Instrucciones de Uso

### 1. Generar un video de prueba (opcional)

```bash
cd "Generador de videos"
python generador.py
```

Esto crea un video con 3 LEDs violeta que se mueven, rotan y parpadean sobre un fondo dinámico.

### 2. Ejecutar el detector

```bash
cd Detector
python Detector_v1.4.py
```

### 3. Controles durante la ejecución

| Tecla | Acción |
|-------|--------|
| `q` | Salir del programa |

---

## 🔬 Versión Final: Detector_v1.4.py

### Descripción

La versión 1.4 implementa un sistema robusto de tracking con:

1. **Corrección Gamma** - Mejora el contraste para detectar LEDs tenues
2. **Segmentación HSV** - Detección por color (violeta/magenta)
3. **Filtro de Circularidad** - Descarta contornos no circulares
4. **Filtro de Kalman (x3)** - Un filtro por cada LED para predicción de posición

### Pipeline de Detección

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE DETECCIÓN                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Frame de entrada                                           │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────┐                                            │
│   │  Corrección │  ◄── Gamma = 1.3 (mejora contraste)        │
│   │    Gamma    │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────┐                                            │
│   │  Gaussian   │  ◄── Kernel 7x7 (reduce ruido)             │
│   │    Blur     │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────┐                                            │
│   │ Conversión  │                                            │
│   │  BGR → HSV  │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────┐                                            │
│   │  Máscara    │  ◄── H: 115-180, S: 40-255, V: 40-255      │
│   │    HSV      │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────┐                                            │
│   │  Encontrar  │                                            │
│   │  Contornos  │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────┐                                            │
│   │  Filtro de  │  ◄── circular_score > 0.18                 │
│   │Circularidad │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────┐                                            │
│   │  Cálculo    │  ◄── Momentos de imagen (subpíxel)         │
│   │ Centroides  │                                            │
│   └──────┬──────┘                                            │
│          ▼                                                   │
│   ┌─────────────────────────────────────┐                    │
│   │      FILTRO DE KALMAN (x3)          │                    │
│   │                                     │                    │
│   │  Para cada LED:                     │                    │
│   │  1. Predecir posición               │                    │
│   │  2. Asignar detección más cercana   │                    │
│   │  3. Corregir si hay detección       │                    │
│   │     o mantener predicción           │                    │
│   │                                     │                    │
│   └─────────────────────────────────────┘                    │
│                        │                                     │
│                        ▼                                     │
│              Posiciones de 3 LEDs                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Parámetros de Configuración

Editar al inicio de `Detector_v1.4.py`:

```python
VIDEO_PATH = "../Generador de videos/video_real.mp4"  # Ruta al video
GAMMA = 1.3              # Corrección de brillo (>1 = más claro)
MAX_JUMP = 200           # Tolerancia predicción ↔ detección
MAX_ASSIGN_DIST = 300    # Distancia máxima para asignar detección a LED

# Rango HSV para detección de color (violeta/magenta)
HSV_LOWER = np.array([115, 40, 40])
HSV_UPPER = np.array([180, 255, 255])
```

### Ventanas de Visualización

El detector muestra 3 ventanas simultáneas:

| Ventana | Contenido |
|---------|-----------|
| **Crudo** | Frame original sin procesar |
| **Mascara HSV** | Máscara binaria de detección por color |
| **Tracking** | Resultado final con LEDs y triángulo |

### Códigos de Color en Tracking

| Color | Significado |
|-------|-------------|
| 🔴 Círculo rojo sólido | LED detectado (medición real) |
| 🟡 Círculo amarillo vacío | LED predicho (no detectado en este frame) |
| 🟢 Líneas verdes | Triángulo formado por los 3 LEDs |

---

## 📹 Generador de Videos

El script `generador.py` crea videos sintéticos con LEDs simulados.

### Parámetros Configurables

```python
# Video
FPS = 30                      # Frames por segundo
DURACION_SEGUNDOS = 10        # Duración total
ANCHO, ALTO = 640, 480        # Resolución

# Fondo dinámico
MIN_BRIGHTNESS = 50           # Brillo mínimo del fondo
MAX_BRIGHTNESS = 220          # Brillo máximo del fondo
PERIODO_FONDO_FRAMES = FPS*5  # Ciclo de variación del fondo

# LEDs
COLOR_LED_ON = (255, 0, 255)  # Violeta (BGR)
COLOR_LED_OFF = (60, 60, 60)  # Gris oscuro
RADIO_LED = 5                 # Radio en píxeles

# Perspectiva
ANGULO_VISION = 0.5           # 0 = frontal, 1 = casi perfil
```

### Frecuencias de Parpadeo

El generador puede crear diferentes patrones de parpadeo modificando la lógica:

```python
# 15 Hz (1 frame ON / 1 frame OFF a 30 FPS)
led_is_on = (frame_idx % 2) == 0

# 1 Hz (15 frames ON / 15 frames OFF a 30 FPS)
led_is_on = (frame_idx % 30) < 15

# 3 ON / 3 OFF
led_is_on = (frame_idx % 6) < 3
```

### Videos Incluidos

| Archivo | Descripción |
|---------|-------------|
| `simulacion_leds_30fps.mp4` | Estándar 30 FPS |
| `simulacion_leds_60fps.mp4` | Alta velocidad |
| `simulacion_leds_120fps.mp4` | Muy alta velocidad |
| `simulacion_leds_parpadeo.mp4` | Énfasis en parpadeo |
| `simulacion_leds_fondo_dinamico.mp4` | Fondo con brillo variable |
| `simulacion_leds_fondo_suave.mp4` | Variaciones suaves de fondo |
| `simulacion_leds_ciclo_3on3off.mp4` | Ciclo 3 frames ON / 3 OFF |
| `simulacion_leds_violeta_1hz.mp4` | Parpadeo lento (1 Hz) |
| `simulacion_leds_violeta_15hz.mp4` | Parpadeo rápido (15 Hz) |
| `simulacion_leds_vision.mp4` | Con efecto de perspectiva |
| `video_real.mp4` | Captura real para pruebas |

---

## 🔧 Filtro de Kalman - Detalles Técnicos

### Modelo de Estado

El filtro usa un modelo de **velocidad constante** con 4 variables:

```
Estado = [x, y, vx, vy]

Donde:
  x, y   = posición actual
  vx, vy = velocidad (píxeles/frame)
```

### Matrices del Filtro

```python
# Matriz de Transición (asume dt = 1 frame)
transitionMatrix = [
    [1, 0, 1, 0],   # x_new = x + vx
    [0, 1, 0, 1],   # y_new = y + vy
    [0, 0, 1, 0],   # vx_new = vx
    [0, 0, 0, 1]    # vy_new = vy
]

# Matriz de Medición (solo medimos posición)
measurementMatrix = [
    [1, 0, 0, 0],   # medimos x
    [0, 1, 0, 0]    # medimos y
]
```

### Ajuste de Ruidos

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `processNoiseCov` | 0.05 | Confianza en el modelo de movimiento |
| `measurementNoiseCov` | 0.2 | Confianza en las mediciones |

- **Mayor processNoiseCov** → Más reactivo a cambios bruscos
- **Menor measurementNoiseCov** → Confía más en las detecciones

---

## 📊 Evolución de Versiones

| Versión | Características Principales |
|---------|----------------------------|
| **v1.0** | Detección híbrida: Parpadeo (diferencia de frames) + HSV fallback |
| **v1.1** | Fusión de métodos, filtro de circularidad, centroide ponderado |
| **v1.2** | Memoria persistente: usa última posición válida si falla detección |
| **v1.3** | Filtro de Kalman, zoom dinámico suavizado, controles interactivos (n/b/p/r) |
| **v1.4** | **Versión final**: Kalman optimizado, código simplificado, mejor rendimiento |

---

## 🐛 Solución de Problemas

### El detector no encuentra LEDs

1. Verificar que `VIDEO_PATH` apunta al archivo correcto
2. Ajustar `GAMMA` (aumentar si los LEDs son tenues)
3. Modificar rangos `HSV_LOWER` y `HSV_UPPER` según el color de los LEDs

### Tabla de referencia HSV por color

| Color | H min | H max |
|-------|-------|-------|
| Rojo (bajo) | 0 | 10 |
| Rojo (alto) | 170 | 180 |
| Naranja | 10 | 25 |
| Amarillo | 25 | 35 |
| Verde | 35 | 85 |
| Azul | 85 | 130 |
| **Violeta/Magenta** | **115** | **180** |

### Los trackers saltan entre LEDs

- Reducir `MAX_ASSIGN_DIST` (ej: de 300 a 150)
- Los LEDs pueden estar demasiado cerca entre sí

### El seguimiento se pierde frecuentemente

- Verificar que el parpadeo no es demasiado lento para el FPS del video
- Ajustar los parámetros del Kalman para mayor tolerancia

---

## 📝 Notas Adicionales

- El sistema está optimizado para **3 LEDs** pero puede modificarse cambiando `num_leds`
- Los videos de simulación usan color **violeta (BGR: 255, 0, 255)** por defecto
- El detector v1.4 está preparado para usar cámara en vivo si `VIDEO_PATH = ""`

---

## 👤 Autor

Tiziano Bertorello
