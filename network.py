"""Нейросеть на чистом NumPy: 784 входа -> скрытый слой (ReLU) -> 11 выходов (softmax).

Выходы 0..9 - цифры, выход 10 - «не цифра» (буква, каракуля и т.п.).

Обозначения (n - сколько картинок обрабатываем за раз):
    x  : (n, 784)       картинки, каждая - строка из 784 чисел
    W1 : (784, hidden)  b1 : (hidden,)
    W2 : (hidden, 11)   b2 : (11,)

Прямой проход:   z1 = x @ W1 + b1  ->  a1 = relu(z1)  ->  z2 = a1 @ W2 + b2  ->  probs = softmax(z2)
Результат probs: (n, 11) - вероятность каждого класса (0..9 и «не цифра») для каждой картинки.
"""
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.npz"
N_CLASSES = 11          # 10 цифр + «не цифра»
UNKNOWN = 10            # номер класса «не цифра»
MIN_CONFIDENCE = 0.7    # если сеть уверена меньше чем на 70%, считаем ответ ненадёжным


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    """Превращает 10 произвольных чисел в 10 вероятностей (сумма = 1)."""
    z = z - z.max(axis=1, keepdims=True)   # вычитаем максимум: ответ тот же, но exp не переполнится
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    """Средний штраф: -log(вероятность, которую сеть дала правильной цифре)."""
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def decide(probs, threshold=MIN_CONFIDENCE):
    """Итоговый ответ: цифра 0..9 или UNKNOWN (10).

    Ответ UNKNOWN, если сеть сама выбрала класс «не цифра» ИЛИ если она не уверена
    (максимальная вероятность ниже порога). Это «вариант отказа»: лучше сказать
    «не знаю», чем уверенно ошибиться.
    """
    labels = probs.argmax(axis=1)
    return np.where(probs.max(axis=1) < threshold, UNKNOWN, labels)


class Network:
    def __init__(self, n_hidden=128, n_in=784, n_out=N_CLASSES, seed=0):
        if n_hidden < 1:
            raise ValueError("n_hidden должно быть >= 1")
        rng = np.random.default_rng(seed)
        # He-инициализация: случайные веса с разбросом sqrt(2/вход) хорошо подходят для ReLU
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self._cache = None

    # ---------- прямой проход ----------
    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.W1.shape[0]:
            raise ValueError(f"ожидается массив формы (n, {self.W1.shape[0]}), получено {x.shape}")
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        self._cache = (x, z1, a1)          # запоминаем промежуточные значения для backward
        return softmax(z2)

    # ---------- обратный проход (backprop) ----------
    def backward(self, probs, y):
        """Считает градиенты функции потерь по всем весам. Нужно вызвать после forward."""
        if self._cache is None:
            raise RuntimeError("сначала вызови forward")
        x, z1, a1 = self._cache
        n = x.shape[0]
        y = np.asarray(y)
        if y.shape != (n,) or y.min() < 0 or y.max() >= probs.shape[1]:
            raise ValueError("y должен быть вектором из n целых меток 0..10")

        # Для softmax + cross-entropy градиент по z2 получается очень простым:
        # (предсказанные вероятности) минус (правильный ответ в виде 0/1). Делим на n - это среднее.
        dz2 = probs.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n

        dW2 = a1.T @ dz2                   # (hidden, 11)
        db2 = dz2.sum(axis=0)              # (11,)
        da1 = dz2 @ self.W2.T              # (n, hidden)  - ошибка, пришедшая в скрытый слой
        dz1 = da1 * (z1 > 0)               # производная ReLU: 1 где z1 > 0, иначе 0
        dW1 = x.T @ dz1                    # (784, hidden)
        db1 = dz1.sum(axis=0)              # (hidden,)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---------- шаг градиентного спуска ----------
    def step(self, grads, lr):
        for name, g in grads.items():
            p = getattr(self, name)
            p -= lr * g                    # идём против градиента (меняем массив на месте)

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    # ---------- сохранение / загрузка ----------
    def save(self, path=MODEL_PATH):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Модель {path} не найдена. Сначала обучи её: python train.py")
        with np.load(path) as d:
            missing = {"W1", "b1", "W2", "b2"} - set(d.files)
            if missing:
                raise ValueError(f"в файле модели нет: {sorted(missing)}")
            W1, b1, W2, b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        ok = (W1.ndim == 2 and W2.ndim == 2 and b1.shape == (W1.shape[1],)
              and W2.shape[0] == W1.shape[1] and b2.shape == (W2.shape[1],))
        if ok and W2.shape[1] == 10:
            raise ValueError("Это модель старого формата (10 классов, без «не цифра»). "
                             "Обучи заново: python train.py")
        if not ok or W1.shape[0] != 784 or W2.shape[1] != N_CLASSES:
            raise ValueError("файл модели повреждён: размеры весов не сходятся")
        net = cls(n_hidden=W1.shape[1])
        net.W1, net.b1, net.W2, net.b2 = W1, b1, W2, b2
        return net
