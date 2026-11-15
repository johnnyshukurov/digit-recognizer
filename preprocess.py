"""Превращает рисунок с экрана в картинку, похожую на MNIST.

Почему нельзя просто сжать рисунок до 28x28? Цифры в MNIST всегда вписаны в рамку 20x20
и стоят по центру масс внутри поля 28x28. Сеть училась на таких картинках, поэтому
нарисованную цифру нужно привести к тому же виду. Файл не зависит от tkinter - его легко тестировать.
"""
import numpy as np

SIZE = 280        # размер поля для рисования (пиксели)
RADIUS = 10       # радиус кисти

_yy, _xx = np.ogrid[-RADIUS:RADIUS + 1, -RADIUS:RADIUS + 1]
BRUSH = (_xx ** 2 + _yy ** 2) <= RADIUS ** 2      # круглая кисть (маска из True/False)


def paint(pixels, x, y):
    """Ставит одну «кляксу» кистью в точку (x, y). Края поля учитываются."""
    y0, y1 = max(0, y - RADIUS), min(SIZE, y + RADIUS + 1)
    x0, x1 = max(0, x - RADIUS), min(SIZE, x + RADIUS + 1)
    if y0 >= y1 or x0 >= x1:
        return
    mask = BRUSH[y0 - (y - RADIUS):y1 - (y - RADIUS), x0 - (x - RADIUS):x1 - (x - RADIUS)]
    pixels[y0:y1, x0:x1][mask] = 1.0


def paint_line(pixels, x0, y0, x1, y1):
    """Проводит штрих от (x0, y0) до (x1, y1): клякса каждые ~2 пикселя, чтобы не было дыр."""
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) // 2 + 1
    for t in np.linspace(0.0, 1.0, steps + 1):
        paint(pixels, int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))


def resize_area(a, new_h, new_w):
    """Уменьшает/увеличивает картинку, усредняя пиксели (без внешних библиотек).

    Каждому новому пикселю соответствует прямоугольник старых; берём его среднюю яркость.
    Суммы по прямоугольникам считаем быстро через накопленные суммы (integral image).
    """
    h, w = a.shape
    ys = np.linspace(0, h, new_h + 1)
    xs = np.linspace(0, w, new_w + 1)
    y0 = np.floor(ys[:-1]).astype(int)
    y1 = np.maximum(np.ceil(ys[1:]).astype(int), y0 + 1)      # минимум 1 пиксель
    x0 = np.floor(xs[:-1]).astype(int)
    x1 = np.maximum(np.ceil(xs[1:]).astype(int), x0 + 1)
    S = np.zeros((h + 1, w + 1))
    S[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)               # S[i, j] = сумма a[:i, :j]
    total = (S[y1[:, None], x1[None, :]] - S[y0[:, None], x1[None, :]]
             - S[y1[:, None], x0[None, :]] + S[y0[:, None], x0[None, :]])
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return total / area


def preprocess(pixels):
    """pixels: 2D массив, 0 = фон, 1 = штрих. Возвращает (1, 784) или None, если пусто."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2:
        raise ValueError("pixels должен быть двумерным массивом")
    ys, xs = np.where(pixels > 0.1)
    if len(xs) == 0:
        return None                                        # ничего не нарисовано

    a = pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # 1) обрезаем по границам цифры
    h, w = a.shape
    scale = 20.0 / max(h, w)                                   # 2) длинная сторона -> 20 пикселей
    a = resize_area(a, max(1, round(h * scale)), max(1, round(w * scale)))

    canvas = np.zeros((28, 28))                                # 3) кладём в центр поля 28x28
    top, left = (28 - a.shape[0]) // 2, (28 - a.shape[1]) // 2
    canvas[top:top + a.shape[0], left:left + a.shape[1]] = a

    total = canvas.sum()                                       # 4) сдвигаем по центру масс
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    dy = int(np.clip(round(13.5 - cy), -4, 4))
    dx = int(np.clip(round(13.5 - cx), -4, 4))
    canvas = np.roll(canvas, (dy, dx), axis=(0, 1))

    return np.clip(canvas, 0.0, 1.0).reshape(1, 784)
