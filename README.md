# Detección de Centros Ópticos

Visión por computadora para medir, cuadro a cuadro y con precisión subpíxel,
la posición de los centros ópticos de los LEDs de un módulo lineal en video.

Es un problema de **metrología**, no de detección: no alcanza con decir "hay un
LED por acá". Hay que devolver una coordenada estable, con el mismo índice
físico en todos los cuadros, y saber cuánto vale esa coordenada.

El proyecto pasó por ocho versiones y dos objetos distintos. Lo que terminó
gobernando el diseño fue un detalle físico: **los LEDs parpadean a 15 Hz y la
cámara filma a 30,012 fps**, es decir, el parpadeo cae exactamente sobre la
frecuencia de Nyquist.

---

## Resultados de la versión 1.7

Medidos sobre nueve videos reales de 1440 × 1440 a 30,012 fps, 4 440 cuadros,
147,9 segundos de material, con el módulo cruzando el encuadre completo.

| Métrica | Valor |
|---|---|
| Cobertura (cuadros con los 6 centros) | **93,6 %** |
| Cobertura por video | 80,7 % a 100 % |
| Repetibilidad (ruido de medición) | **0,13 px** mediana, 0,26 px p95 |
| Desvío respecto de la fila ideal equiespaciada | 0,27 px RMS |
| Diferencia sistemática entre regímenes | **0,90 px** (ver más abajo) |
| Cuadros sin solución (el detector se abstiene) | 6,4 % |

### Cómo se resolvió cada cuadro

| Estado | Vía | Cuadros | % |
|---|---|---|---|
| `LED` | Blobs saturados + RANSAC | 2 937 | 66,1 % |
| `BAR` | Búsqueda local | 1 041 | 23,4 % |
| `MOD` | Forma del módulo | 154 | 3,5 % |
| `ACQ` | Adquisición | 26 | 0,6 % |
| `COAST` | Sin medición, sostiene la pose | 239 | 5,4 % |
| `LOST` | Sin solución | 43 | 1,0 % |

### Por video

| Video | Cuadros | Segundos | Cobertura | Espaciado | Movimiento |
|---|---|---|---|---|---|
| 172332 | 329 | 11,0 | 100,0 % | 32,0 px | 0,8 px/cuadro |
| 172546 | 368 | 12,3 | 100,0 % | 32,4 px | 27,3 |
| 172649 | 324 | 10,8 | 88,9 % | 32,0 px | 1,1 |
| 172729 | 445 | 14,8 | 92,1 % | 32,3 px | 18,0 |
| 173420 | 482 | 16,1 | 99,4 % | 31,8 px | 0,2 |
| 173527 | 589 | 19,6 | 98,5 % | 32,8 px | 32,5 |
| 173649 | 509 | 17,0 | 99,0 % | 31,1 px | 41,0 |
| 173754 | 553 | 18,4 | 94,2 % | 27,1 px | 37,6 |
| 174058 | 841 | 28,0 | 80,7 % | 24,5 px | 20,9 |

---

## El problema de los 15 Hz

La frecuencia de Nyquist de una cámara de 30,012 fps es 15,006 Hz. El parpadeo
medido en los videos es de 15,01 Hz. El parpadeo cae **sobre** Nyquist, así que
lo que se captura deja de depender de la señal y pasa a depender de la fase
entre el obturador y el parpadeo. Como los dos relojes no están sincronizados,
la fase deriva y el régimen cambia a lo largo de cada video.

Hay dos regímenes, y los dos aparecen:

- **Encendido.** Los LEDs saturan y el halo inunda los huecos. El contraste
  lente/hueco cae a 37. Las seis manchas se funden entre sí.
- **Atenuado.** Los LEDs siguen emitiendo pero mucho más débil: la lente queda
  en 95 a 226 niveles de gris contra una barra en 17. El contraste sube a 200.
  **No se apagan del todo**, y este régimen es fotométricamente el más limpio.

Además, una minoría de cuadros cae justo sobre la conmutación y, por el
obturador rodante, el módulo queda partido dentro del mismo cuadro: parte de
las lentes encendidas y parte apagadas.

El 23,4 % de los cuadros se resolvió en el régimen atenuado. Sin tratarlo
explícitamente, la cobertura caería aproximadamente a la mitad.

---

## Cómo funciona la versión 1.7

Los 6 centros se modelan como un cuerpo rígido de **4 grados de libertad**, no
como 12 incógnitas independientes:

```
p_i = c + (i - 2,5) · s · u(θ)        i = 0 … 5
```

con `c` el centro, `s` el espaciado y `θ` el ángulo. Reparametrizado como
`p_k = c + k·d` el ajuste es lineal y se resuelve por mínimos cuadrados en una
sola operación.

Cada cuadro se mide por tres vías independientes, todas pasan por el mismo
refinamiento y la misma validación, y se adopta la de mayor contraste que sea
consistente con el seguimiento:

| Vía | Método | Cuándo sirve | Estado |
|---|---|---|---|
| A | Blobs saturados a varios umbrales + RANSAC de fila equiespaciada | LEDs encendidos y separados | `LED` |
| B | Forma del módulo por momentos centrales, sobre el cuadro reducido 4× | Encendidos y atenuados | `MOD` |
| C | Búsqueda local en rejilla, con escala y ángulo casi congelados | Requiere seguimiento previo | `BAR` |

Las vías A y B son **mediciones globales sin información previa**: son las que
impiden que el modelo derive. La vía C es la barata y resuelve la mayoría de
los cuadros atenuados.

Sobre la medición elegida:

1. **Refinamiento**: centroide ponderado por intensidad en cada lente, con
   umbral relativo al fondo local y corrección acotada al 6 % del espaciado.
2. **Ajuste**: mínimos cuadrados lineales de los 4 GDL.
3. **Validación**: contraste lente/hueco, colinealidad, uniformidad de
   espaciado y escala dentro de rango.

Si la validación no pasa, el detector **se abstiene** en vez de devolver una
posición mala. Para una aplicación de medición, un dato malo sin marcar es peor
que no tener dato.

---

## Nivel de certeza

Lo que está medido, y cómo:

- **Repetibilidad, 0,13 px.** En cinco tramos casi inmóviles (300 cuadros) se
  quitó el movimiento real con un polinomio de grado 3 por coordenada; el
  residuo es ruido de medición. σ = 0,116 px a lo largo de la fila y 0,097 px
  perpendicular.
- **El supuesto de equiespaciado es válido a 0,27 px.** Se promediaron hasta
  200 cuadros llevados a una pose canónica usando sólo el centro y el ángulo
  (nunca el espaciado), y sobre la imagen apilada se midieron los 6 centros con
  centroide ponderado, sin modelo de fila y sin tope. El desvío respecto de la
  fila ideal es 0,27 px RMS, 0,47 px máximo.
- **El espaciado coincide con la medición independiente a 0,03 px.**
- **Hay un sesgo sistemático de 0,90 px entre los dos regímenes.** Medido por
  tripletas encendido-atenuado-encendido, que cancelan el movimiento suave
  hasta segundo orden, con control de signo invertido: +0,77 px a lo largo de
  la fila, +0,47 px perpendicular y +0,40 px de espaciado (medianas, n = 602,
  intervalos de confianza del 95 % por bootstrap más angostos que 0,02 px).

**Lo que no está verificado:** la exactitud absoluta. No hay en el proyecto un
patrón calibrado ni un conjunto etiquetado. El encabezado del código declara un
error medio de 0,24 px contra 216 centros etiquetados a mano, pero ese archivo
no está, así que esa cifra no es reproducible. El sesgo de 0,90 px entre
regímenes dice que al menos uno de los dos arrastra cerca de un píxel de error,
y sin referencia externa no se puede decidir cuál.

---

## Estructura

```
Detector/
  Detector_v1.0.py … Detector_v1.7.py   las ocho versiones
  DetectorFiltroKalman.py               filtro de Kalman, línea aparte
  salida/                               9 CSV con los centros medidos
Generador de videos/
  generador.py                          generador sintético
  simulacion_leds_*.mp4                 banco de pruebas controlado
  video_real.mp4                        primer video real, etapa de 3 LEDs
```

Los nueve videos reales de prueba (320 MB) no están en el repositorio.

### Evolución

| Versión | Fecha | Qué agregó |
|---|---|---|
| 1.0 | nov 2025 | Diferencia de cuadros + HSV de reserva, centroide ponderado |
| 1.1 | nov 2025 | Filtros de circularidad, fusión real de ambos métodos |
| 1.2 | nov 2025 | Memoria persistente de la última posición válida |
| 1.3 | dic 2025 | Filtro de Kalman por LED, asignación voraz, herramientas de inspección |
| 1.4 | dic 2025 | Primer video real: gamma, HSV violeta, 3 filtros fijos. Se cae la diferencia de cuadros |
| 1.5 | abr 2026 | Modelo geométrico del triángulo, calibración automática, inferencia del tercer vértice |
| 1.6 | may 2026 | Cambio de objeto a 6 LEDs en línea. RANSAC, cuerpo rígido. Se cae el Kalman |
| 1.7 | ago 2026 | Rediseño alrededor del parpadeo: 3 vías, 4 GDL, validación con abstención |

---

## Uso

Requiere Python 3 con OpenCV y NumPy.

```bash
pip install opencv-python numpy
```

```bash
python3 Detector/Detector_v1.7.py                            # menú de videos, con ventana
python3 Detector/Detector_v1.7.py video.mp4                  # un video, con ventana
python3 Detector/Detector_v1.7.py video.mp4 --sin-ventana    # un video, sin ventana
python3 Detector/Detector_v1.7.py --todos                    # todos, sin ventana
python3 Detector/Detector_v1.7.py video.mp4 --csv out.csv    # exporta los centros
python3 Detector/Detector_v1.7.py video.mp4 --grabar out.mp4
python3 Detector/Detector_v1.7.py --carpeta "otra/carpeta" --todos
```

`--sin-ventana` no figura en el encabezado del archivo. Es la opción que hace
falta para cronometrar un video suelto o para llamar al detector desde un
script.

### Reproducibilidad

El detector es determinista: volver a correr los videos 172332, 173527 y 174058
produjo CSV idénticos byte a byte a los de `Detector/salida/`. Velocidad medida
con reloj de pared: de 62 a 100 cuadros por segundo según el video, es decir
entre 2,1 y 3,3 veces más rápido que tiempo real.

Controles en modo ventana: `q`/`ESC` salir, `ESPACIO` pausa, `r` re-adquirir,
`n`/`p` video siguiente y anterior, `+`/`-` velocidad, `0` velocidad original,
`m` máscara, `d` diagnóstico por consola, `g` guardar el cuadro anotado.

### Formato del CSV

| Columna | Contenido |
|---|---|
| `frame` | Índice del cuadro, desde cero |
| `t_seg` | Tiempo en segundos |
| `estado` | `ACQ`, `LED`, `MOD`, `BAR`, `COAST` o `LOST` |
| `espaciado_px` | Distancia entre LEDs contiguos. Vacío si no hubo solución |
| `angulo_deg` | Ángulo de la fila respecto de la horizontal |
| `contraste` | Diferencia de gris entre lentes y huecos |
| `x0 … y5` | Los seis centros, con tres decimales. Vacíos en `COAST` y `LOST` |

---

## Limitaciones conocidas

- **No hay verdad de referencia.** La exactitud absoluta no es verificable.
- **El modelo asume que la fila se ve recta.** Los 4 GDL cubren traslación,
  rotación en el plano y escala, pero no perspectiva. Una vista muy oblicua
  rompería la uniformidad del espaciado.
- **La cantidad de LEDs es un parámetro**, no algo que el detector deduzca.
- **El régimen depende de la fase.** Si la fase queda alineada con el régimen
  difícil durante todo un video, el sistema trabaja siempre en el peor caso.
- **Las versiones anteriores no fueron medidas sobre el material nuevo.** La
  comparación 23 % contra 93,6 % viene del encabezado del código de la v1.7, no
  de una corrida reproducida.

### Próximos pasos

1. Construir un conjunto etiquetado de referencia y versionarlo acá.
2. Filmar a 60 fps o desincronizar el parpadeo. Eso elimina el problema en la
   captura en vez de en el algoritmo, y es la mejora de mayor impacto por menor
   esfuerzo.
3. Resolver el sesgo de 0,90 px entre regímenes, que hoy es el mayor término de
   error conocido.
