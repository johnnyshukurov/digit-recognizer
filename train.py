"""Обучение сети. Запуск:  python train.py

Полезные варианты:
    python train.py --samples 5000 --epochs 3     # быстрый пробный запуск (секунды)
    python train.py --lr 0.05 --hidden 64         # свои параметры
    python train.py --unknown 0                   # без класса «не цифра»

Сеть учится на цифрах MNIST и на сгенерированных примерах «не цифра» (буквы, каракули,
кляксы; см. unknown.py).
"""
import argparse
import time

import numpy as np

from data import load_mnist, to_input
from network import MIN_CONFIDENCE, MODEL_PATH, UNKNOWN, Network, cross_entropy, decide
from unknown import make_unknown


def probabilities(net, x, chunk=2000):
    """Вероятности для всех картинок. Считаем кусками, чтобы не держать всё в памяти."""
    return np.concatenate([net.forward(to_input(x[i:i + chunk])) for i in range(0, len(x), chunk)])


def evaluate(net, x, y):
    """Возвращает (loss, accuracy)."""
    probs = probabilities(net, x)
    return cross_entropy(probs, y), float((probs.argmax(axis=1) == y).mean())


def build_training_set(x, y, n_unknown, seed=0):
    """Склеивает цифры и сгенерированные «не цифры», перемешивает."""
    x_all = np.concatenate([x, make_unknown(n_unknown, seed=1)])
    y_all = np.concatenate([y, np.full(n_unknown, UNKNOWN, dtype=np.int64)])
    order = np.random.default_rng(seed).permutation(len(x_all))
    return x_all[order], y_all[order]


def train(net, x, y, x_val, y_val, epochs, lr, batch_size, seed=0):
    rng = np.random.default_rng(seed)
    print(f"{'эпоха':>5} | {'train loss':>10} {'train acc':>9} | {'val loss':>8} {'val acc':>8} | время")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(x))                 # новый случайный порядок каждую эпоху
        for start in range(0, len(x), batch_size):
            idx = order[start:start + batch_size]       # одна «порция» (mini-batch)
            probs = net.forward(to_input(x[idx]))
            grads = net.backward(probs, y[idx])
            net.step(grads, lr)

        n = min(len(x), 10000)                          # train считаем по первым 10 000 - быстрее
        tl, ta = evaluate(net, x[:n], y[:n])
        vl, va = evaluate(net, x_val, y_val)
        if not np.isfinite(tl):
            raise SystemExit("loss стал nan/inf: learning rate слишком большой. Попробуй --lr 0.05")
        print(f"{epoch:5d} | {tl:10.4f} {ta:9.4f} | {vl:8.4f} {va:8.4f} | {time.time() - t0:4.1f} c")


def report(net, x_digits, y_digits, x_unknown):
    """Итоговый отчёт на тестовых данных, которых сеть не видела."""
    p_digits = probabilities(net, x_digits)
    acc = float((p_digits.argmax(axis=1) == y_digits).mean())
    print(f"Test, цифры: верно {acc:.4f}")

    labels = decide(p_digits)                           # ответ с порогом уверенности
    accepted = labels != UNKNOWN
    print(f"  с порогом уверенности {MIN_CONFIDENCE}: ответила «не знаю» на {1 - accepted.mean():.4f} цифр, "
          f"из принятых верно {float((labels == y_digits)[accepted].mean()):.4f}")

    if len(x_unknown):
        rejected = float((decide(probabilities(net, x_unknown)) == UNKNOWN).mean())
        print(f"Test, «не цифры» (сгенерированные буквы/каракули): узнано как «не цифра» {rejected:.4f}")
        print("  (это оптимистичная оценка: картинки сделаны тем же генератором, что и при обучении;\n"
              "   настоящая проверка: нарисуй свои буквы в окошке)")


def positive_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("нужно целое число >= 1")
    return v


def non_negative_int(s):
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("нужно целое число >= 0")
    return v


def positive_float(s):
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("нужно число > 0")
    return v


def main():
    p = argparse.ArgumentParser(description="Обучение сети на MNIST + класс «не цифра»")
    p.add_argument("--epochs", type=positive_int, default=10, help="сколько раз пройти по всем данным")
    p.add_argument("--lr", type=positive_float, default=0.1, help="learning rate (размер шага)")
    p.add_argument("--hidden", type=positive_int, default=128, help="нейронов в скрытом слое")
    p.add_argument("--batch-size", type=positive_int, default=64, help="картинок в одной порции")
    p.add_argument("--samples", type=positive_int, default=None,
                   help="брать только первые N обучающих цифр (для быстрых проб)")
    p.add_argument("--unknown", type=non_negative_int, default=None,
                   help="сколько примеров «не цифра» сгенерировать (по умолчанию 10%% от числа цифр)")
    args = p.parse_args()

    try:
        (x, y), (x_val, y_val), (x_test, y_test) = load_mnist()
    except RuntimeError as e:
        raise SystemExit(str(e))
    if args.samples:
        x, y = x[:args.samples], y[:args.samples]
    n_unknown = args.unknown if args.unknown is not None else len(x) // 10

    print(f"Генерирую {n_unknown} примеров «не цифра» (буквы, каракули)...")
    x_train, y_train = build_training_set(x, y, n_unknown)
    n_val_unknown = n_unknown * len(x_val) // max(len(x), 1)       # в val та же доля «не цифр»
    x_val_all = np.concatenate([x_val, make_unknown(n_val_unknown, seed=2)])
    y_val_all = np.concatenate([y_val, np.full(n_val_unknown, UNKNOWN, dtype=np.int64)])
    x_test_unknown = make_unknown(1000 if n_unknown else 0, seed=3)

    print(f"train: {len(x_train)} ({len(x)} цифр + {len(x_train) - len(x)} «не цифр»), "
          f"val: {len(x_val_all)}, test: {len(x_test)} цифр + {len(x_test_unknown)} «не цифр»\n")

    net = Network(n_hidden=args.hidden)
    train(net, x_train, y_train, x_val_all, y_val_all, args.epochs, args.lr, args.batch_size)

    print()
    report(net, x_test, y_test, x_test_unknown)
    if evaluate(net, x_test, y_test)[1] < 0.5:
        print("\nВНИМАНИЕ: сеть почти не обучилась. Обычно помогает меньший --lr (например, 0.05) "
              "или больше --epochs.")
    net.save()
    print(f"\nМодель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
