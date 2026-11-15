"""Примеры класса «не цифра» (метка 10).

Откуда берём такие картинки, не скачивая ничего тяжёлого:
  1. БУКВЫ: рисуем их «штриховым шрифтом» (набор отрезков) со случайными искажениями.
     Намеренно НЕ берём буквы, которые на самом деле похожи на цифры: O (ноль), I и l (единица),
     S (пятёрка), Z (двойка), B (восьмёрка), G (шестёрка).
  2. КАРАКУЛИ: случайные ломаные из 5-10 точек.
  3. КЛЯКСЫ: сплошные пятна.

Каждая картинка рисуется той же кистью и проходит через ту же preprocess(), что и рисунки
из окошка, поэтому для сети они выглядят одинаково.
"""
import numpy as np

from preprocess import RADIUS, SIZE, paint, paint_line, preprocess


# Буквы как наборы штрихов в квадрате 1x1 (x вправо, y вниз).
LETTERS = {
    "A": [[(0, 1), (0.5, 0), (1, 1)], [(0.2, 0.65), (0.8, 0.65)]],
    "C": [[(1, 0.15), (0.6, 0), (0.2, 0.1), (0, 0.5), (0.2, 0.9), (0.6, 1), (1, 0.85)]],
    "E": [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.8, 0.5)]],
    "F": [[(1, 0), (0, 0), (0, 1)], [(0, 0.5), (0.8, 0.5)]],
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.5), (1, 0.5)]],
    "K": [[(0, 0), (0, 1)], [(1, 0), (0, 0.55), (1, 1)]],
    "M": [[(0, 1), (0, 0), (0.5, 0.6), (1, 0), (1, 1)]],
    "N": [[(0, 1), (0, 0), (1, 1), (1, 0)]],
    "R": [[(0, 1), (0, 0), (0.9, 0), (1, 0.25), (0.9, 0.5), (0, 0.5)], [(0.45, 0.5), (1, 1)]],
    "T": [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]],
    "U": [[(0, 0), (0, 0.7), (0.25, 1), (0.75, 1), (1, 0.7), (1, 0)]],
    "V": [[(0, 0), (0.5, 1), (1, 0)]],
    "W": [[(0, 0), (0.25, 1), (0.5, 0.4), (0.75, 1), (1, 0)]],
    "X": [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
    "Y": [[(0, 0), (0.5, 0.5), (1, 0)], [(0.5, 0.5), (0.5, 1)]],
}


def _draw(strokes):
    """strokes: список ломаных в пикселях поля -> картинка 280x280."""
    pixels = np.zeros((SIZE, SIZE))
    for stroke in strokes:
        for (x0, y0), (x1, y1) in zip(stroke[:-1], stroke[1:]):
            paint_line(pixels, x0, y0, x1, y1)
    return pixels


def _letter(rng):
    name = rng.choice(sorted(LETTERS))
    width = rng.uniform(0.55, 1.1)                     # буква то уже, то шире
    shear = rng.uniform(-0.25, 0.25)                   # наклон
    angle = rng.uniform(-0.25, 0.25)                   # поворот (радианы)
    size = rng.uniform(140, 240)                       # размер в пикселях
    cos, sin = np.cos(angle), np.sin(angle)
    strokes = []
    for stroke in LETTERS[name]:
        pts = []
        for x, y in stroke:
            x, y = x + rng.normal(0, 0.04), y + rng.normal(0, 0.04)    # «дрожь руки»
            x = (x - 0.5) * width + shear * (y - 0.5)
            y = y - 0.5
            x, y = x * cos - y * sin, x * sin + y * cos
            pts.append((SIZE / 2 + x * size, SIZE / 2 + y * size))
        strokes.append(pts)
    return _draw(strokes)


def _scribble(rng):
    k = int(rng.integers(5, 11))                       # от 5 до 10 случайных точек
    pts = rng.uniform(25, SIZE - 25, size=(k, 2))
    return _draw([[tuple(p) for p in pts]])


def _blob(rng):
    pixels = np.zeros((SIZE, SIZE))
    cx, cy = rng.uniform(80, 200, size=2)
    for _ in range(int(rng.integers(1, 4))):
        paint(pixels, int(cx + rng.normal(0, 8)), int(cy + rng.normal(0, 8)))
    return pixels


def make_unknown(n, seed=0):
    """n картинок «не цифра», форма (n, 784), uint8 (как и картинки MNIST в data.py)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 784), dtype=np.uint8)
    for i in range(n):
        r = rng.random()
        pixels = _letter(rng) if r < 0.5 else _scribble(rng) if r < 0.9 else _blob(rng)
        out[i] = np.round(preprocess(pixels) * 255).astype(np.uint8)
    return out
