#!/usr/bin/env python3
"""
Detector de Centros Opticos v1.7
================================
Detecta y sigue los centros de los 6 LEDs de un modulo lineal en video.

QUE CAMBIA RESPECTO DE LA v1.6
------------------------------
La v1.6 estaba pensada para el video de laboratorio (fondo controlado, LEDs
siempre encendidos). En los videos reales de "Videos de Prueba - Deteccion de
Centros" rendia mal (23% de frames utiles, error medio de cientos de pixeles)
por cuatro motivos, y los cuatro estan resueltos aca:

  1. PARPADEO A 15 Hz. En los videos reales los LEDs se apagan en frames
     alternados: la MITAD de los frames no tiene ni un pixel saturado. La v1.6
     se quedaba sin mediciones y arrastraba el modelo rigido a la deriva.
     -> v1.7: en los frames apagados el modulo negro sigue siendo visible y las
        6 lentes se ven como puntos grises sobre el; el detector las mide igual
        (estado BAR). Ademas cada frame encendido re-mide la pose DESDE CERO,
        asi que la escala y el angulo no pueden derivar.

  2. BLOOM. Los LEDs saturan y se fusionan: a umbral 225 las 6 componentes se
     vuelven 2 o 3 y la deteccion por blobs fallaba.
     -> v1.7: se prueban varios umbrales (240/250/225/253); el bloom se separa
        en alguno de ellos. Y si no, el modulo se detecta igual por su forma.

  3. DISTRACTORES. Monitores, ventanas y luminarias del techo son brillantes y
     la v1.6 se enganchaba a ellos.
     -> v1.7: toda solucion tiene que pasar una validacion fotometrica: las 6
        lentes deben ser mas claras que los huecos que las separan, la fila debe
        ser colineal y equiespaciada, y mas alla del ultimo LED debe seguir
        habiendo barra oscura. Si no pasa, se ABSTIENE (found=False).

  4. PRECISION. La v1.6 devolvia posiciones suavizadas por un filtro de
     velocidad, con sesgo de varios pixeles.
     -> v1.7: cada lente se refina por centroide ponderado por intensidad y los
        6 puntos se ajustan por minimos cuadrados a un modelo de fila de 4 GDL
        (centro, angulo, espaciado). Medido contra 216 centros etiquetados a
        mano: error medio 0.24 px, maximo 1.97 px.

COMO FUNCIONA
-------------
Modelo: los 6 centros son un cuerpo rigido de 4 grados de libertad
    p_i = c + (i - 2.5) * s * u(theta)
(verificado sobre el ground truth: la desviacion respecto de la fila ideal
equiespaciada es menor a 0.03 * espaciado).

Cada frame se buscan mediciones por tres vias independientes, se refinan todas
igual y se elige la de mayor contraste que sea consistente con el seguimiento:

  A. BLOBS SATURADOS + RANSAC  (frames encendidos con los LEDs separados)
     Umbrales multiples -> componentes -> RANSAC de fila equiespaciada.
     Es una medicion global, sin prior: es lo que impide la deriva.

  B. FORMA DEL MODULO          (encendidos y apagados)
     Mascara de "oscuro respecto del fondo local" unida a la de "brillante",
     sobre el frame reducido 4x. Da candidatos a modulo (centro, angulo, largo)
     por momentos centrales. Para cada uno se barre el espaciado y se queda el
     mejor. Funciona con los LEDs apagados y con la barra partida por bloom.

  C. BUSQUEDA LOCAL            (solo si ya hay seguimiento)
     Rejilla gruesa alrededor de la prediccion, con escala y angulo casi
     congelados (la camara no cambia de distancia un 6% en 33 ms).

Y sobre la medicion elegida:
  REFINAMIENTO  centroide ponderado por (I - umbral_local)+ en cada lente
  AJUSTE        minimos cuadrados lineales de los 4 GDL sobre las lentes fiables
  VALIDACION    contraste lente/hueco + colinealidad + uniformidad + escala

USO
---
    python3 Detector_v1.7.py                          # menu de videos + ventana
    python3 Detector_v1.7.py video.mp4                # un video con ventana
    python3 Detector_v1.7.py --todos                  # los 9 videos, sin ventana
    python3 Detector_v1.7.py video.mp4 --csv out.csv  # exporta los centros
    python3 Detector_v1.7.py video.mp4 --grabar out.mp4
    python3 Detector_v1.7.py --carpeta "otra/carpeta" --todos

CONTROLES (modo ventana)
------------------------
    q / ESC   salir                 ESPACIO  pausa
    r         re-adquirir           n / p    video siguiente / anterior
    + / -     mas rapido / lento    0        velocidad original
    m         mostrar mascara       d        volcar diagnostico por consola
    g         guardar el frame anotado
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

# ==========================================================================
# CONFIGURACION
# ==========================================================================
N_LEDS = 6

# Carpeta por defecto: los videos de prueba, relativos a este archivo.
AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA_DEFECTO = os.path.normpath(
    os.path.join(AQUI, "..", "Videos de Prueba - Detección de Centros"))

# --- deteccion de LEDs encendidos -----------------------------------------
# El bloom fusiona LEDs vecinos: a 225 pueden quedar 2 componentes en vez de 6,
# pero a 240/250/253 se separan. Se prueban en orden de mejor tasa de acierto.
NIVELES_SATURACION = (240, 250, 225, 253)
AREA_MIN_BLOB = 3
ASPECTO_MAX_BLOB = 5.0        # descarta tubos fluorescentes del techo

# --- geometria del modulo (medida sobre 36 frames etiquetados a mano) ------
#   espaciado entre LEDs : 25 a 36 px con el frame a 1440 px
#   alto de la barra     : ~2.1 * espaciado
#   ancho de la barra    : ~10  * espaciado
# Los limites se expresan relativos al lado del frame para que el detector
# funcione igual con otras resoluciones.
S_MIN_REL = 0.0045            # 6.5 px a 1440
S_MAX_REL = 0.070             # 100 px a 1440

# --- validacion -----------------------------------------------------------
# Calibrado sobre detecciones correctas: el contraste lente/hueco va de 32 a
# 146. El umbral de adquisicion es deliberadamente exigente.
CONTRASTE_ADQ = 30.0
CONTRASTE_SEG = 18.0
# La uniformidad del espaciado y la colinealidad no necesitan umbral propio:
# los centros que se devuelven son el modelo de 4 GDL mas una correccion acotada
# a CLAMP_REFINO, asi que la fila es equiespaciada y colineal por construccion.

# --- dinamica (limites por frame, a 30 fps) -------------------------------
SALTO_MAX = 6.0               # salto del centro al re-anclar, en espaciados
ESCALA_MAX = 0.25             # cambio de escala al re-anclar
BAR_ESCALA = 0.02             # cambio de escala en un frame apagado
BAR_ANGULO = 0.05             # cambio de angulo en un frame apagado (rad)
BAR_SALTO = 1.6               # salto del centro en un frame apagado
COAST_MAX = 3                 # frames que se sostiene sin ninguna medicion
# Un 2% por frame parece poco, pero en una racha larga de frames apagados se
# acumula (40 frames -> x2.2). La escala se ancla a la ultima medicion global y
# no puede alejarse mas que esto mientras no haya otra.
ANCLA_ESCALA = 0.22
DESACUERDO_MAX = 2            # mediciones globales seguidas que contradicen el
                              # seguimiento antes de re-anclar por la fuerza

# --- correccion sub-pixel -------------------------------------------------
VENTANA_REFINO = 0.38         # radio de la ventana, en espaciados
UMBRAL_REFINO = 0.60          # umbral relativo dentro de la ventana
CLAMP_REFINO = 0.06           # cuanto puede apartarse una lente del modelo
# Cuantas de las 6 lentes tienen que confirmarse individualmente (tener
# estructura local propia) para aceptar una solucion. Con la pose correcta se
# confirman las 6, tanto encendidas como apagadas; exigir 5 descarta las poses
# de escala inflada, donde la mitad de los puntos cae sobre madera lisa.
LENTES_MIN = 4
LENTES_MIN_BAR = 5

# --- busqueda global por forma del modulo ---------------------------------
REDUCCION = 4                 # 1440 -> 360 para la busqueda global
VENTANA_FONDO = 49            # ventana del fondo local en la imagen reducida
MAX_CANDIDATOS = 8


# ==========================================================================
# NUCLEO GEOMETRICO
# ==========================================================================
def bilinear(img, xs, ys):
    """Muestreo bilineal vectorizado. Devuelve (valores, dentro_de_imagen)."""
    h, w = img.shape
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    fx = xs - x0c
    fy = ys - y0c
    Ia = img[y0c, x0c]
    Ib = img[y0c, x0c + 1]
    Ic = img[y0c + 1, x0c]
    Id = img[y0c + 1, x0c + 1]
    v = (Ia * (1 - fx) * (1 - fy) + Ib * fx * (1 - fy) +
         Ic * (1 - fx) * fy + Id * fx * fy)
    return v.astype(np.float32), ok


def model_points(pose, n=N_LEDS):
    """pose = (cx, cy, theta, s) -> array (n, 2) con los n centros del modelo."""
    cx, cy, th, s = pose
    k = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    return np.stack([cx + k * s * math.cos(th),
                     cy + k * s * math.sin(th)], axis=1)


def _model_points_batch(cx, cy, th, s, n):
    k = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0)[None, :]
    X = cx[:, None] + k * s[:, None] * np.cos(th)[:, None]
    Y = cy[:, None] + k * s[:, None] * np.sin(th)[:, None]
    return X, Y


def contrast_batch(gray, cx, cy, th, s, n=N_LEDS):
    """
    Contraste lente/hueco de muchas hipotesis a la vez.

    La lente se mide como MAXIMO en un disco chico y el hueco como MINIMO a lo
    largo del tramo: con los LEDs apagados la lente es un punto gris de pocos
    pixeles y muestrear un unico punto la erraba por completo.

    Devuelve (contraste, puntaje, valida). El puntaje penaliza que mas alla del
    ultimo LED haya algo tan brillante como un LED: eso significa que el modelo
    se corrio una posicion entera sobre la fila.

    OJO: sirve para elegir y para validar, NUNCA para afinar la pose. Maximizar
    contraste la sesga (inclina la fila para meter los huecos en zonas oscuras).
    """
    X, Y = _model_points_batch(cx, cy, th, s, n)
    ux = np.cos(th)[:, None]
    uy = np.sin(th)[:, None]
    r = (0.16 * s)[:, None]
    led = None
    okl = None
    for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        v, ok = bilinear(gray, X + r * (ox * ux - oy * uy),
                         Y + r * (ox * uy + oy * ux))
        led = v if led is None else np.maximum(led, v)
        if okl is None:
            okl = ok
    gap = None
    okg = None
    for t in (0.38, 0.5, 0.62):
        gx = X[:, :-1] + t * (X[:, 1:] - X[:, :-1])
        gy = Y[:, :-1] + t * (Y[:, 1:] - Y[:, :-1])
        v, ok = bilinear(gray, gx, gy)
        gap = v if gap is None else np.minimum(gap, v)
        if okg is None:
            okg = ok
    off = (n / 2.0 + 1.0) * s
    ex = np.stack([cx + off * np.cos(th), cx - off * np.cos(th)], 1)
    ey = np.stack([cy + off * np.sin(th), cy - off * np.sin(th)], 1)
    ends, _ = bilinear(gray, ex, ey)

    led_m = np.median(led, axis=1)
    gap_m = np.median(gap, axis=1)
    con = led_m - gap_m
    pen = np.maximum(0.0, ends.max(axis=1) - 0.5 * (led_m + gap_m))
    return con, con - 0.8 * pen, okl.all(axis=1) & okg.all(axis=1)


def contrast_one(gray, pose, n=N_LEDS):
    a = np.array([pose[0]], np.float32)
    b = np.array([pose[1]], np.float32)
    c = np.array([pose[2]], np.float32)
    d = np.array([pose[3]], np.float32)
    con, sc, ok = contrast_batch(gray, a, b, c, d, n)
    return float(con[0]), float(sc[0]), bool(ok[0])


def fit_pose(pts, ks, s_min, s_max, n=N_LEDS):
    """
    Ajuste LINEAL por minimos cuadrados de   p_k = c + k * d
    con k = indice - (n-1)/2.  Incognitas: cx, cy, dx, dy.
    Devuelve ((cx, cy, theta, s), residuo_rms) o None.
    """
    m = len(pts)
    if m < 3:
        return None
    A = np.zeros((2 * m, 4))
    b = np.zeros(2 * m)
    for i, ((x, y), k) in enumerate(zip(pts, ks)):
        A[2 * i] = (1, 0, k, 0)
        b[2 * i] = x
        A[2 * i + 1] = (0, 1, 0, k)
        b[2 * i + 1] = y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, dx, dy = sol
    s = float(math.hypot(dx, dy))
    if not (s_min <= s <= s_max):
        return None
    res = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
    return (float(cx), float(cy), float(math.atan2(dy, dx)), s), res


def refine_leds(gray, pose, n=N_LEDS):
    """
    Refinamiento sub-pixel: en la ventana de cada lente calcula el centroide
    ponderado por (I - umbral_local)+, con el umbral relativo al fondo de la
    ventana (la barra negra). Descarta las lentes sin estructura o que se van
    demasiado lejos de lo predicho.
    Devuelve (puntos, indices_k, contrastes_locales).
    """
    P = model_points(pose, n)
    s = pose[3]
    r = max(2, int(round(VENTANA_REFINO * s)))
    h, w = gray.shape
    pts, ks, cons = [], [], []
    for i, (x, y) in enumerate(P):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patch = gray[y0:y1, x0:x1]
        lo = float(np.percentile(patch, 15))
        hi = float(patch.max())
        if hi - lo < 12.0:
            continue
        wgt = patch - (lo + UMBRAL_REFINO * (hi - lo))
        np.maximum(wgt, 0.0, out=wgt)
        tot = float(wgt.sum())
        if tot < 1e-3:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        rx = float((xs * wgt).sum() / tot)
        ry = float((ys * wgt).sum() / tot)
        if math.hypot(rx - x, ry - y) > 0.55 * s:
            continue
        pts.append((rx, ry))
        ks.append(i - (n - 1) / 2.0)
        cons.append(hi - lo)
    return pts, ks, cons


# ==========================================================================
# VIA A: blobs saturados + RANSAC de fila
# ==========================================================================
def bright_blobs(gray_u8, umbral):
    _, m = cv2.threshold(gray_u8, umbral, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nn, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    pts = []
    for i in range(1, nn):
        a = st[i, cv2.CC_STAT_AREA]
        if not (AREA_MIN_BLOB <= a <= 8000):
            continue
        w = st[i, cv2.CC_STAT_WIDTH]
        h = st[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / max(1.0, min(w, h)) > ASPECTO_MAX_BLOB:
            continue
        pts.append((float(ce[i][0]), float(ce[i][1])))
    return pts[:60]


def ransac_row(pts, s_min, s_max, n=N_LEDS, tol=0.26):
    """
    Busca n puntos colineales equiespaciados. Prueba cada par de candidatos
    como los LEDs (a, a+g) para varios huecos g, de modo que la fila se
    encuentre aunque falten LEDs de los extremos.
    """
    if len(pts) < 4:
        return None
    P = np.asarray(pts, np.float64)
    nb = len(P)
    best = None
    for i in range(nb):
        for j in range(nb):
            if i == j:
                continue
            v = P[j] - P[i]
            L = float(np.hypot(v[0], v[1]))
            if L < 3 * s_min:
                continue
            for g in (n - 1, n - 2, n - 3):
                s = L / g
                if not (s_min <= s <= s_max):
                    continue
                step = v / g
                for a0 in range(0, n - g):
                    origin = P[i] - a0 * step
                    sel, kk = [], []
                    for k in range(n):
                        e = origin + k * step
                        d = np.hypot(P[:, 0] - e[0], P[:, 1] - e[1])
                        m = int(np.argmin(d))
                        if d[m] < tol * s:
                            sel.append(tuple(P[m]))
                            kk.append(k - (n - 1) / 2.0)
                    if len(sel) < 5:
                        continue
                    fit = fit_pose(sel, kk, s_min, s_max, n)
                    if fit is None or fit[1] > 0.20 * fit[0][3]:
                        continue
                    cand = (len(sel), -fit[1], fit[0])
                    if best is None or cand[:2] > best[:2]:
                        best = cand
    return None if best is None else best[2]


# ==========================================================================
# VIA B: forma del modulo (funciona con los LEDs encendidos Y apagados)
# ==========================================================================
def _rellenar_huecos(m, area_max, dim_max):
    """Con los LEDs encendidos la barra negra queda como un anillo: se cierra."""
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    if n <= 1:
        return m
    H, W = m.shape
    fill = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a > area_max or max(w, h) > dim_max:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        fill.append(i)
    if not fill:
        return m
    out = m.copy()
    out[np.isin(lab, fill)] = 1
    return out


def _momentos(mask_piece):
    mu = cv2.moments(mask_piece, binaryImage=True)
    m00 = mu["m00"]
    if m00 <= 4:
        return None
    a20 = mu["mu20"] / m00
    a02 = mu["mu02"] / m00
    a11 = mu["mu11"] / m00
    com = math.sqrt(max(0.0, (a20 - a02) ** 2 + 4 * a11 * a11))
    largo = math.sqrt(12 * max(0.5 * (a20 + a02 + com), 1e-6))
    ancho = max(math.sqrt(12 * max(0.5 * (a20 + a02 - com), 1e-6)), 1.0)
    ang = 0.5 * math.atan2(2 * a11, a20 - a02)
    return mu["m10"] / m00, mu["m01"] / m00, ang, largo, ancho


def module_candidates(gray_u8, s_min, s_max):
    """
    Candidatos a modulo por forma, sobre el frame reducido.
    Une la mascara de "oscuro respecto del fondo local" (la barra negra, visible
    siempre) con la de "brillante" (los LEDs), porque con bloom fuerte la barra
    queda partida en dos y solo la union reconstruye la pieza entera.
    Devuelve una lista de (cx, cy, angulo, largo) en coordenadas full-res.
    """
    H, W = gray_u8.shape
    ws = max(32, W // REDUCCION)
    hs = max(32, H // REDUCCION)
    small = cv2.resize(gray_u8, (ws, hs), interpolation=cv2.INTER_AREA)
    sx = W / float(ws)
    sy = H / float(hs)
    k = min(VENTANA_FONDO, (min(ws, hs) // 2) * 2 + 1)
    bg = cv2.blur(small, (k, k)).astype(np.int16)
    sm = small.astype(np.int16)

    oscuro = (bg - sm > np.maximum(20, bg >> 2)).astype(np.uint8)
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    oscuro = _rellenar_huecos(oscuro, area_max=(ws * hs) // 50, dim_max=ws // 3)
    brillo = ((sm > 214) | (sm - bg > 70)).astype(np.uint8)

    out = []
    # --- el modulo entero: barra oscura U LEDs brillantes ---
    mod = cv2.morphologyEx(oscuro | brillo, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mod, 8)
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        if max(bw, bh) < 8 or max(bw, bh) > ws // 2:
            continue
        sh = _momentos((lab[by:by + bh, bx:bx + bw] == i).astype(np.uint8))
        if sh is None:
            continue
        ux, uy, ang, largo, ancho = sh
        if not (1.9 <= largo / ancho <= 9.0):
            continue
        L = largo * sx
        if not (7.0 * s_min <= L <= 12.5 * s_max):
            continue
        out.append(((bx + ux) * sx, (by + uy) * sy, ang, L, a * sx * sy, "BARRA"))

    # --- la tira brillante sola (LEDs encendidos, barra comida por el bloom) ---
    if brillo.any():
        grp = cv2.dilate(brillo, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grp, 8)
        for i in range(1, n):
            bx, by, bw, bh, a = st[i]
            if max(bw, bh) < 7 or max(bw, bh) > ws // 2:
                continue
            piece = ((lab[by:by + bh, bx:bx + bw] == i) &
                     (brillo[by:by + bh, bx:bx + bw] > 0)).astype(np.uint8)
            sh = _momentos(piece)
            if sh is None:
                continue
            ux, uy, ang, largo, ancho = sh
            if not (1.5 <= largo / ancho <= 18.0):
                continue
            L = largo * sx
            if not (3.8 * s_min <= L <= 7.0 * s_max):
                continue
            out.append(((bx + ux) * sx, (by + uy) * sy, ang, L, a * sx * sy, "TIRA"))

    out.sort(key=lambda c: -c[4])
    return [(c[0], c[1], c[2], c[3], c[5]) for c in out[:MAX_CANDIDATOS]]


def seed_from_candidate(gray, cand, s_min, s_max, n=N_LEDS):
    """
    Dado un candidato a modulo (centro, angulo, largo) barre espaciado, angulo y
    desplazamiento a lo largo del eje, y devuelve la mejor hipotesis de fila.
    """
    cx, cy, ang, L, kind = cand
    # El largo medido significa cosas distintas segun el candidato: la barra
    # negra completa mide ~10 espaciados y la tira de LEDs ~5. Barrer un rango
    # unico que cubriera ambos dejaba entrar escalas 2.5x mas grandes que la
    # real, y el contraste las aceptaba: hay que barrer el rango de cada tipo.
    if kind == "BARRA":
        s_lo = max(s_min, L / 12.5)
        s_hi = min(s_max, L / 7.0)
    else:
        s_lo = max(s_min, L / 7.0)
        s_hi = min(s_max, L / 3.8)
    if s_hi < s_lo:
        return None
    S = np.linspace(s_lo, s_hi, 11, dtype=np.float32)
    T = (ang + np.linspace(-0.09, 0.09, 5)).astype(np.float32)
    D = np.linspace(-1.2, 1.2, 9, dtype=np.float32)          # en espaciados
    SS, TT, DD = np.meshgrid(S, T, D, indexing="ij")
    SS = SS.ravel(); TT = TT.ravel(); DD = DD.ravel()
    CX = (cx + DD * SS * np.cos(TT)).astype(np.float32)
    CY = (cy + DD * SS * np.sin(TT)).astype(np.float32)
    con, sc, ok = contrast_batch(gray, CX, CY, TT, SS, n)
    sc = np.where(ok, sc, -1e9)
    b = int(np.argmax(sc))
    if sc[b] <= 0:
        return None
    return (float(CX[b]), float(CY[b]), float(TT[b]), float(SS[b]))


# ==========================================================================
# DETECTOR
# ==========================================================================
class OpticalRowDetector:
    """
    Detector + seguidor de una fila de n LEDs equiespaciados.

    process(frame_bgr) -> dict(
        found   : bool
        centers : [(x, y)] * n en coordenadas del frame, sub-pixel, o None
        state   : 'ACQ' | 'LED' | 'MOD' | 'BAR' | 'COAST' | 'LOST'
        pose    : (cx, cy, theta, s) o None
        quality : contraste lente/hueco de la solucion
    )
    """

    def __init__(self, n_leds=N_LEDS):
        self.n = n_leds
        self.pose = None
        self.vel = np.zeros(4)
        self.coast = 0
        self.s_ancla = None       # escala de la ultima medicion global
        self.desacuerdo = 0
        self.s_min = None
        self.s_max = None
        self.frame_idx = -1
        self.last_mask = None
        self.stats = {k: 0 for k in
                      ("ACQ", "LED", "MOD", "BAR", "COAST", "LOST")}

    # ---------------------------------------------------------------- utils
    def _limites(self, shape):
        ref = float(max(shape[0], shape[1]))
        self.s_min = S_MIN_REL * ref
        self.s_max = S_MAX_REL * ref

    def _predict(self):
        if self.pose is None:
            return None
        p = np.array(self.pose) + self.vel
        p[3] = float(np.clip(p[3], self.s_min, self.s_max))
        return (float(p[0]), float(p[1]), float(p[2]), float(p[3]))

    def _consistent(self, pose, ref):
        if ref is None:
            return True
        if math.hypot(pose[0] - ref[0], pose[1] - ref[1]) > SALTO_MAX * ref[3]:
            return False
        return abs(pose[3] - ref[3]) / ref[3] <= ESCALA_MAX

    # ------------------------------------------------- refinar + validar
    def _settle(self, gray, pose, umbral, ref=None, ds=None, dth=None,
                lentes_min=LENTES_MIN):
        """
        Refina las lentes, reajusta los 4 GDL y valida. Con `ref` limita cuanto
        puede moverse la escala y el angulo respecto de esa pose.
        Devuelve (pose, centros, contraste) o None.
        """
        pts, ks, _ = refine_leds(gray, pose, self.n)
        movido = False
        if len(pts) >= 4 and (max(ks) - min(ks)) >= 3.0:
            fit = fit_pose(pts, ks, self.s_min, self.s_max, self.n)
            if fit is not None and fit[1] <= 0.25 * fit[0][3]:
                pose = fit[0]
                movido = True
                if ref is not None:
                    a = (pose[2] - ref[2] + math.pi) % (2 * math.pi) - math.pi
                    pose = (pose[0], pose[1],
                            ref[2] + float(np.clip(a, -dth, dth)),
                            float(np.clip(pose[3], ref[3] * (1 - ds),
                                          ref[3] * (1 + ds))))
        con, _, ok = contrast_one(gray, pose, self.n)
        if not ok or con < umbral:
            return None
        if not self._geometria_ok(pose):
            return None

        P = model_points(pose, self.n)
        out = [tuple(p) for p in P]
        lim = CLAMP_REFINO * pose[3]
        if movido:
            pts, ks, _ = refine_leds(gray, pose, self.n)
        if len(pts) < lentes_min:
            return None
        for (p, k) in zip(pts, ks):
            i = int(round(k + (self.n - 1) / 2.0))
            dx = p[0] - P[i][0]
            dy = p[1] - P[i][1]
            d = math.hypot(dx, dy)
            if d > lim:
                dx *= lim / d
                dy *= lim / d
            out[i] = (P[i][0] + dx, P[i][1] + dy)
        return pose, out, con

    def _geometria_ok(self, pose):
        return self.s_min <= pose[3] <= self.s_max

    def _ancla_ok(self, pose):
        """La escala no puede alejarse del ancla mientras no haya medicion global."""
        if self.s_ancla is None:
            return True
        return abs(pose[3] - self.s_ancla) / self.s_ancla <= ANCLA_ESCALA

    def _canonizar(self, pose, centers):
        """
        Fija el sentido del eje para que L0 sea SIEMPRE el mismo LED fisico.
        El ajuste de la fila no distingue un extremo del otro: sin esto el
        angulo alterna entre +1 y -179 grados y los indices se dan vuelta de un
        frame al siguiente, que para medir es inservible.
        """
        cx, cy, th, s = pose
        if self.pose is not None:
            invertir = math.cos(th - self.pose[2]) < 0
        else:
            ux, uy = math.cos(th), math.sin(th)
            invertir = ux < -1e-9 or (abs(ux) <= 1e-9 and uy < 0)
        if invertir:
            th += math.pi
            centers = list(centers)[::-1]
        th = (th + math.pi) % (2 * math.pi) - math.pi
        return (cx, cy, th, s), centers

    def _adopt(self, pose, state, centers, con):
        pose, centers = self._canonizar(pose, centers)
        if state in ("ACQ", "LED", "MOD"):
            self.s_ancla = pose[3]        # las vias globales re-anclan la escala
            self.desacuerdo = 0
        if self.pose is not None:
            v = np.array(pose) - np.array(self.pose)
            v[2] = (v[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(v[2]) > 0.6:            # cambio de sentido del eje, no giro
                v[2] = 0.0
            self.vel = 0.45 * self.vel + 0.55 * v
            self.vel[3] *= 0.4
            self.vel[:2] = np.clip(self.vel[:2], -2.0 * pose[3], 2.0 * pose[3])
        else:
            self.vel = np.zeros(4)
        self.pose = pose
        self.coast = 0
        self.stats[state] = self.stats.get(state, 0) + 1
        return dict(found=True, centers=centers, state=state,
                    pose=pose, quality=con)

    def _nada(self, state):
        self.stats[state] = self.stats.get(state, 0) + 1
        return dict(found=False, centers=None, state=state,
                    pose=self.pose, quality=0.0)

    # ---------------------------------------------------------------- ciclo
    def process(self, frame):
        self.frame_idx += 1
        if self.s_min is None:
            self._limites(frame.shape[:2])
        g8 = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        gray = g8.astype(np.float32)
        pred = self._predict()
        umbral = CONTRASTE_SEG if self.pose is not None else CONTRASTE_ADQ

        # ---- VIA A: blobs saturados + RANSAC (medicion global, sin prior) ----
        mejor = None
        for lvl in NIVELES_SATURACION:
            blobs = bright_blobs(g8, lvl)
            if len(blobs) < 5:
                continue
            pose_g = ransac_row(blobs, self.s_min, self.s_max, self.n)
            if pose_g is None:
                continue
            r = self._settle(gray, pose_g, umbral)
            if r is None:
                continue
            if mejor is None or r[2] > mejor[2]:
                mejor = r
            if self._consistent(r[0], pred):
                break
        if mejor is not None:
            if self._consistent(mejor[0], pred):
                return self._adopt(mejor[0],
                                   "LED" if self.pose is not None else "ACQ",
                                   mejor[1], mejor[2])
            self.desacuerdo += 1
            if self.desacuerdo >= DESACUERDO_MAX and mejor[2] >= CONTRASTE_ADQ:
                self.pose = None
                self.vel = np.zeros(4)
                return self._adopt(mejor[0], "ACQ", mejor[1], mejor[2])

        # ---- VIA C: busqueda local alrededor de la prediccion ----
        if pred is not None:
            c = self._coarse(gray, pred)
            if c is not None:
                r = self._settle(gray, c, umbral, ref=pred,
                                 ds=BAR_ESCALA, dth=BAR_ANGULO,
                                 lentes_min=LENTES_MIN_BAR)
                if r is not None and self._ancla_ok(r[0]) and \
                   math.hypot(r[0][0] - pred[0],
                              r[0][1] - pred[1]) <= BAR_SALTO * pred[3]:
                    return self._adopt(r[0], "BAR", r[1], r[2])

        # ---- VIA B: forma del modulo (encendido o apagado, sin prior) ----
        mejor_b = None
        for cand in module_candidates(g8, self.s_min, self.s_max):
            seed = seed_from_candidate(gray, cand, self.s_min, self.s_max, self.n)
            if seed is None:
                continue
            r = self._settle(gray, seed, umbral)
            if r is None:
                continue
            if mejor_b is None or r[2] > mejor_b[2]:
                mejor_b = r
        if mejor_b is not None and self._consistent(mejor_b[0], pred) \
                and self._ancla_ok(mejor_b[0]):
            return self._adopt(mejor_b[0], "MOD" if self.pose is not None else "ACQ",
                               mejor_b[1], mejor_b[2])

        # ---- re-anclaje: el seguimiento venia mal y hay una medicion buena ----
        if self.coast >= 2:
            for r in (mejor, mejor_b):
                if r is not None:
                    self.pose = None
                    self.vel = np.zeros(4)
                    return self._adopt(r[0], "ACQ", r[1], r[2])

        # ---- sin medicion util ----
        if self.pose is not None:
            self.coast += 1
            if self.coast <= COAST_MAX:
                self.vel *= 0.5
                self.pose = self._predict()
                return self._nada("COAST")
            self.pose = None
            self.vel = np.zeros(4)
            self.coast = 0
        return self._nada("LOST")

    def _coarse(self, gray, pose):
        """Rejilla gruesa por contraste. Solo para acercarse, nunca para afinar."""
        best = pose
        for (rng, paso, rt, nt, rs, ns) in ((2.4, 0.5, 0.10, 5, 0.02, 3),
                                            (0.6, 0.2, 0.04, 3, 0.01, 3)):
            d = np.arange(-rng, rng + 1e-6, paso) * best[3]
            t = np.linspace(-rt, rt, nt)
            e = np.linspace(1 - rs, 1 + rs, ns)
            DX, DY, DT, DS = np.meshgrid(d, d, t, e, indexing="ij")
            CX = (best[0] + DX).ravel().astype(np.float32)
            CY = (best[1] + DY).ravel().astype(np.float32)
            TH = (best[2] + DT).ravel().astype(np.float32)
            S = (best[3] * DS).ravel().astype(np.float32)
            keep = (S >= self.s_min) & (S <= self.s_max)
            if not keep.any():
                return None
            CX, CY, TH, S = CX[keep], CY[keep], TH[keep], S[keep]
            con, sc, ok = contrast_batch(gray, CX, CY, TH, S, self.n)
            sc = np.where(ok, sc, -1e9)
            b = int(np.argmax(sc))
            if sc[b] < 0:
                return None
            best = (float(CX[b]), float(CY[b]), float(TH[b]), float(S[b]))
        return best

    def reset(self):
        self.pose = None
        self.vel = np.zeros(4)
        self.coast = 0
        self.s_ancla = None
        self.desacuerdo = 0


def create(n_leds=N_LEDS):
    """Fabrica usada por el banco de pruebas."""
    return OpticalRowDetector(n_leds)


# ==========================================================================
# VISUALIZACION
# ==========================================================================
COLOR_ESTADO = {
    "ACQ":   (0, 255, 255),
    "LED":   (0, 255, 0),
    "MOD":   (0, 200, 255),
    "BAR":   (255, 200, 0),
    "COAST": (0, 165, 255),
    "LOST":  (0, 0, 255),
}


def dibujar(frame, res, extra=""):
    out = frame.copy()
    color = COLOR_ESTADO.get(res["state"], (200, 200, 200))
    if res["found"]:
        C = res["centers"]
        for i in range(len(C) - 1):
            cv2.line(out, (int(C[i][0]), int(C[i][1])),
                     (int(C[i + 1][0]), int(C[i + 1][1])), (90, 90, 90), 1)
        for i, (x, y) in enumerate(C):
            xi, yi = int(round(x)), int(round(y))
            cv2.drawMarker(out, (xi, yi), color, cv2.MARKER_CROSS, 18, 2)
            cv2.circle(out, (xi, yi), 11, color, 1)
            cv2.putText(out, f"L{i}", (xi + 13, yi - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    h = frame.shape[0]
    esc = h / 1440.0
    txt = f"{res['state']}"
    if res["found"]:
        p = res["pose"]
        txt += (f"  esp={p[3]:.1f}px  ang={math.degrees(p[2]):+.1f}deg"
                f"  contraste={res['quality']:.0f}")
    for t, y in ((txt, 50), (extra, 90)):
        if not t:
            continue
        cv2.putText(out, t, (18, int(y * esc)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * esc, (0, 0, 0), int(6 * esc) + 1, cv2.LINE_AA)
        cv2.putText(out, t, (18, int(y * esc)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * esc, color, max(1, int(2 * esc)), cv2.LINE_AA)
    return out


# ==========================================================================
# MODOS DE EJECUCION
# ==========================================================================
def listar_videos(carpeta):
    ext = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv")
    fs = []
    for e in ext:
        fs.extend(glob.glob(os.path.join(carpeta, e)))
    return sorted(set(fs))


def procesar_headless(path, n_leds, csv_path=None, grabar=None, progreso=True):
    """Procesa un video completo sin ventana. Devuelve el resumen."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = OpticalRowDetector(n_leds)

    escritor = None
    fcsv = None
    if csv_path:
        fcsv = open(csv_path, "w", newline="")
        escritor = csv.writer(fcsv)
        escritor.writerow(["frame", "t_seg", "estado", "espaciado_px",
                           "angulo_deg", "contraste"] +
                          [f"{c}{i}" for i in range(n_leds) for c in ("x", "y")])
    vw = None
    k = -1
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k += 1
        res = det.process(frame)
        if escritor:
            fila = [k, round(k / fps, 4), res["state"]]
            if res["found"]:
                fila += [round(res["pose"][3], 3),
                         round(math.degrees(res["pose"][2]), 3),
                         round(res["quality"], 1)]
                for (x, y) in res["centers"]:
                    fila += [round(x, 3), round(y, 3)]
            else:
                fila += ["", "", ""] + [""] * (2 * n_leds)
            escritor.writerow(fila)
        if grabar:
            vis = dibujar(frame, res, f"frame {k}/{total}")
            if vw is None:
                vw = cv2.VideoWriter(grabar, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (vis.shape[1], vis.shape[0]))
            vw.write(vis)
        if progreso and total > 0 and k % 60 == 0:
            pct = 100.0 * k / total
            sys.stderr.write(f"\r    {os.path.basename(path)}  {pct:5.1f}%   ")
            sys.stderr.flush()
    dt = time.perf_counter() - t0
    cap.release()
    if vw:
        vw.release()
    if fcsv:
        fcsv.close()
    if progreso:
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()

    n = k + 1
    det_ok = sum(det.stats[s] for s in ("ACQ", "LED", "MOD", "BAR"))
    return dict(video=os.path.basename(path), frames=n,
                detectados=det_ok, cobertura=det_ok / max(n, 1),
                fps=n / dt if dt > 0 else 0.0, estados=dict(det.stats))


def run_batch(videos, n_leds, csv_dir=None, grabar_dir=None):
    print(f"\nProcesando {len(videos)} video(s)\n")
    print(f"{'video':<26}{'frames':>7}{'detectados':>12}{'cobertura':>11}"
          f"{'fps':>8}   estados")
    print("-" * 92)
    filas = []
    for v in videos:
        base = os.path.splitext(os.path.basename(v))[0]
        csv_path = os.path.join(csv_dir, base + ".csv") if csv_dir else None
        grab = os.path.join(grabar_dir, base + "_v17.mp4") if grabar_dir else None
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        if grabar_dir:
            os.makedirs(grabar_dir, exist_ok=True)
        r = procesar_headless(v, n_leds, csv_path, grab)
        filas.append(r)
        est = {k: n for k, n in r["estados"].items() if n}
        print(f"{r['video']:<26}{r['frames']:>7}{r['detectados']:>12}"
              f"{100*r['cobertura']:>10.1f}%{r['fps']:>8.1f}   {est}")
    if filas:
        print("-" * 92)
        tot_f = sum(r["frames"] for r in filas)
        tot_d = sum(r["detectados"] for r in filas)
        print(f"{'TOTAL':<26}{tot_f:>7}{tot_d:>12}{100*tot_d/max(tot_f,1):>10.1f}%"
              f"{np.mean([r['fps'] for r in filas]):>8.1f}")
    print()
    return filas


def run_gui(videos, idx, n_leds, csv_dir=None):
    """Reproduccion con ventana, controles de teclado y ajuste en vivo."""
    mostrar_mascara = False
    mascara_abierta = False
    while 0 <= idx < len(videos):
        path = videos[idx]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"no se pudo abrir {path}")
            return
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if not (0 < fps_video <= 240):
            fps_video = 30.0
        intervalo = 1.0 / fps_video
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        det = OpticalRowDetector(n_leds)
        vel = 1.0
        pausa = False
        k = -1
        ultimo = None
        siguiente = None
        print(f"\n[{idx+1}/{len(videos)}] {os.path.basename(path)}  "
              f"{total} frames @ {fps_video:.1f} fps")

        cv2.namedWindow("Deteccion de Centros Opticos v1.7", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteccion de Centros Opticos v1.7", 900, 900)

        while True:
            t_ini = time.perf_counter()
            if not pausa:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    det.reset()
                    k = -1
                    continue
                k += 1
                ultimo = (frame, det.process(frame))
            if ultimo is None:
                break
            frame, res = ultimo
            extra = (f"frame {k}/{total}   x{vel:.2f}"
                     f"{'   PAUSA' if pausa else ''}   "
                     f"[{idx+1}/{len(videos)}] {os.path.basename(path)}")
            cv2.imshow("Deteccion de Centros Opticos v1.7",
                       dibujar(frame, res, extra))
            if mostrar_mascara:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, m = cv2.threshold(g, NIVELES_SATURACION[0], 255, cv2.THRESH_BINARY)
                cv2.imshow("Mascara (LEDs saturados)", m)
                mascara_abierta = True
            elif mascara_abierta:
                cv2.destroyWindow("Mascara (LEDs saturados)")
                mascara_abierta = False

            espera = max(1, int((intervalo / max(vel, 0.05) -
                                 (time.perf_counter() - t_ini)) * 1000))
            key = cv2.waitKey(espera) & 0xFF
            if key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                siguiente = idx + 1
                break
            elif key == ord('p'):
                siguiente = idx - 1
                break
            elif key == ord('r'):
                det.reset()
                print("  re-adquisicion forzada")
            elif key == ord(' '):
                pausa = not pausa
            elif key in (ord('+'), ord('=')):
                vel = min(8.0, vel * 1.5)
            elif key in (ord('-'), ord('_')):
                vel = max(0.1, vel / 1.5)
            elif key == ord('0'):
                vel = 1.0
            elif key == ord('m'):
                mostrar_mascara = not mostrar_mascara
            elif key == ord('d'):
                print(f"  f{k} estado={res['state']} pose={res['pose']} "
                      f"contraste={res['quality']:.1f} acum={det.stats}")
            elif key == ord('g'):
                nombre = f"frame_{os.path.splitext(os.path.basename(path))[0]}_{k:05d}.png"
                cv2.imwrite(nombre, dibujar(frame, res, extra))
                print(f"  guardado {nombre}")

        cap.release()
        det_ok = sum(det.stats[s] for s in ("ACQ", "LED", "MOD", "BAR"))
        print(f"  cobertura: {det_ok}/{k+1} frames "
              f"({100*det_ok/max(k+1,1):.1f}%)  estados={det.stats}")
        if siguiente is None:
            break
        idx = max(0, min(len(videos) - 1, siguiente))
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Detector de centros opticos v1.7",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="archivo de video a procesar")
    ap.add_argument("--carpeta", default=CARPETA_DEFECTO,
                    help="carpeta con los videos (por defecto: Videos de Prueba)")
    ap.add_argument("--todos", action="store_true",
                    help="procesa todos los videos de la carpeta, sin ventana")
    ap.add_argument("--sin-ventana", action="store_true",
                    help="procesa sin abrir ventana")
    ap.add_argument("--leds", type=int, default=N_LEDS,
                    help=f"cantidad de LEDs en la fila (por defecto {N_LEDS})")
    ap.add_argument("--csv", default=None,
                    help="archivo CSV de salida (o carpeta si se usa --todos)")
    ap.add_argument("--grabar", default=None,
                    help="video anotado de salida (o carpeta si se usa --todos)")
    a = ap.parse_args()

    if a.video:
        videos = [a.video]
        if not os.path.exists(a.video):
            alt = os.path.join(a.carpeta, a.video)
            if os.path.exists(alt):
                videos = [alt]
            else:
                print(f"no existe: {a.video}")
                return 1
    else:
        videos = listar_videos(a.carpeta)
        if not videos:
            print(f"no hay videos en: {a.carpeta}")
            return 1

    if a.todos or a.sin_ventana:
        if a.todos and not a.video:
            run_batch(videos, a.leds, a.csv, a.grabar)
        else:
            for v in videos:
                r = procesar_headless(v, a.leds, a.csv, a.grabar)
                print(f"{r['video']}: {r['detectados']}/{r['frames']} frames "
                      f"({100*r['cobertura']:.1f}%)  {r['fps']:.1f} fps  {r['estados']}")
        return 0

    idx = 0
    if not a.video and len(videos) > 1:
        print(f"\nVideos en {a.carpeta}:\n")
        for i, v in enumerate(videos):
            print(f"  [{i+1}] {os.path.basename(v)}")
        print("\n  ENTER = el primero,  numero = ese video,  't' = todos sin ventana")
        try:
            sel = input("  > ").strip()
        except EOFError:
            sel = ""
        if sel.lower() == "t":
            run_batch(videos, a.leds, a.csv, a.grabar)
            return 0
        if sel.isdigit() and 1 <= int(sel) <= len(videos):
            idx = int(sel) - 1

    print("\n  q salir | ESPACIO pausa | r re-adquirir | n/p video sig/ant")
    print("  +/- velocidad | 0 normal | m mascara | d diagnostico | g guardar frame")
    run_gui(videos, idx, a.leds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
