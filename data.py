"""Загрузка MNIST (рукописные цифры 28x28).

Картинки храним как uint8 (числа 0..255): так в 8 раз меньше памяти, чем float64.
В дробные числа 0..1 их превращает to_input() прямо перед подачей в сеть.
"""
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
NPZ_PATH = DATA_DIR / "mnist.npz"

# Зеркала, одно файл mnist.npz. Если первое не отвечает, пробуем следующее.
URLS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "https://s3.amazonaws.com/img-datasets/mnist.npz",
]


def _read_and_check(path):
    """Читает файл и проверяет, что внутри именно MNIST. Возвращает 4 массива."""
    try:
        with np.load(path) as d:
            for key in ("x_train", "y_train", "x_test", "y_test"):
                if key not in d.files:
                    raise ValueError(f"в файле нет массива '{key}'")
            x_train, y_train = d["x_train"], d["y_train"]
            x_test, y_test = d["x_test"], d["y_test"]
    except (OSError, EOFError) as e:
        raise ValueError(f"файл не читается: {e}")
    except ValueError as e:
        if "pickle" in str(e):                                # так numpy ругается на не-npz файлы
            raise ValueError("файл не похож на mnist.npz (возможно, скачалась страница с ошибкой)")
        raise
    if x_train.shape != (60000, 28, 28) or x_test.shape != (10000, 28, 28):
        raise ValueError(f"неожиданные размеры: {x_train.shape}, {x_test.shape}")
    if len(y_train) != 60000 or len(y_test) != 10000:
        raise ValueError("число меток не совпадает с числом картинок")
    if y_train.min() < 0 or y_train.max() > 9:
        raise ValueError("метки должны быть числами от 0 до 9")
    return x_train, y_train, x_test, y_test


def _download():
    DATA_DIR.mkdir(exist_ok=True)
    tmp = NPZ_PATH.with_name("mnist.part")        # качаем во временный файл,
    errors = []                                   # чтобы не остался битый mnist.npz
    for url in URLS:
        try:
            print(f"Скачиваю MNIST: {url}")
            urllib.request.urlretrieve(url, tmp)
            _read_and_check(tmp)                  # убеждаемся, что файл целый
            tmp.replace(NPZ_PATH)
            print(f"Сохранено: {NPZ_PATH}")
            return
        except Exception as e:                    # сеть, битый файл и т.д.
            errors.append(f"  {url}\n    -> {e}")
            tmp.unlink(missing_ok=True)
    hint = ""
    if any("CERTIFICATE" in e.upper() for e in errors):
        hint = ("\n\nПохоже на проблему с сертификатами (бывает на macOS): запусти в папке Python "
                "файл 'Install Certificates.command' или скачай файл вручную.")
    raise RuntimeError(
        "Не удалось скачать MNIST:\n" + "\n".join(errors) + hint +
        f"\n\nСкачай файл mnist.npz вручную (например, по первой ссылке в браузере)\n"
        f"и положи его сюда: {NPZ_PATH}"
    )


def load_mnist():
    """Возвращает (x_train, y_train), (x_val, y_val), (x_test, y_test).

    x - массивы uint8 формы (N, 784), y - целые метки 0..9.
    train = первые 50 000 картинок, val = последние 10 000 из обучающей части,
    test = отдельные 10 000 картинок, которые сеть не видит до самого конца.
    """
    if not NPZ_PATH.exists():
        _download()
    try:
        x_train, y_train, x_test, y_test = _read_and_check(NPZ_PATH)
    except Exception as e:
        raise RuntimeError(f"Файл {NPZ_PATH} повреждён ({e}). Удали его и запусти снова.")

    x_train = x_train.reshape(-1, 784)            # 28x28 -> вектор из 784 чисел
    x_test = x_test.reshape(-1, 784)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return ((x_train[:50000], y_train[:50000]),
            (x_train[50000:], y_train[50000:]),
            (x_test, y_test))


def to_input(x_uint8):
    """uint8 0..255 -> float64 0..1 (сети удобнее маленькие числа)."""
    return x_uint8.astype(np.float64) / 255.0
