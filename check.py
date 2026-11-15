"""Самопроверка проекта. Запуск:  python check.py

Ничего не скачивает и не требует обученной модели - работает на случайных данных.
Если что-то сломалось (например, после твоих правок), здесь будет видно, что именно.
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

import app
import data
import train
import unknown
from network import MIN_CONFIDENCE, UNKNOWN, Network, cross_entropy, decide, softmax
from preprocess import SIZE, paint_line, preprocess, resize_area

TESTS = []


def test(f):
    TESTS.append(f)
    return f


# ---------- сеть ----------
@test
def softmax_sums_to_one_and_is_stable():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]))   # большие числа не ломают
    assert np.isfinite(p).all(), "softmax выдал nan/inf"
    assert np.allclose(p.sum(axis=1), 1.0), "вероятности не дают в сумме 1"
    assert np.allclose(p[0], p[1]), "softmax должен давать одно и то же при сдвиге всех чисел"


@test
def forward_shapes_and_validation():
    net = Network(n_hidden=16)
    probs = net.forward(np.random.default_rng(0).random((5, 784)))
    assert probs.shape == (5, 11), f"форма выхода {probs.shape}, ожидалась (5, 11)"
    assert np.allclose(probs.sum(axis=1), 1.0)
    try:
        net.forward(np.zeros((5, 100)))
    except ValueError:
        return
    raise AssertionError("сеть должна отвергать вход неправильной формы")


@test
def backward_matches_numerical_gradient():
    """Главная проверка backprop: сравниваем с градиентом, посчитанным «в лоб»."""
    rng = np.random.default_rng(1)
    net = Network(n_hidden=8, seed=3)
    x, y = rng.normal(size=(6, 784)), rng.integers(0, 11, 6)
    grads = net.backward(net.forward(x), y)
    eps = 1e-5
    for name, analytic in grads.items():
        param = getattr(net, name)
        numeric = np.zeros_like(param)
        for idx in np.ndindex(*param.shape):
            if param.size > 200 and rng.random() > 0.02:     # у больших матриц проверяем выборочно
                numeric[idx] = analytic[idx]
                continue
            old = param[idx]
            param[idx] = old + eps
            loss_plus = cross_entropy(net.forward(x), y)
            param[idx] = old - eps
            loss_minus = cross_entropy(net.forward(x), y)
            param[idx] = old
            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
        err = np.abs(analytic - numeric).max() / (np.abs(analytic).max() + 1e-12)
        assert err < 1e-5, f"градиент {name} не совпал с численным (ошибка {err:.1e})"


@test
def network_can_learn():
    """Сеть должна выучить простой набор данных: если нет - в обучении баг."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(10, 784))
    y = rng.integers(0, 10, 500)
    x = centers[y] + rng.normal(scale=1.0, size=(500, 784))
    net = Network(n_hidden=32)
    first_loss = cross_entropy(net.forward(x), y)
    for _ in range(60):
        net.step(net.backward(net.forward(x), y), lr=0.01)
    last_loss = cross_entropy(net.forward(x), y)
    acc = (net.predict(x) == y).mean()
    assert last_loss < first_loss, "loss не уменьшается"
    assert acc > 0.95, f"сеть не выучила простые данные, accuracy {acc:.2f}"


@test
def save_and_load_roundtrip():
    net = Network(n_hidden=12, seed=5)
    x = np.random.default_rng(2).random((4, 784))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.npz"
        net.save(path)
        loaded = Network.load(path)
        assert np.allclose(net.forward(x), loaded.forward(x)), "после загрузки предсказания изменились"
        np.savez(Path(tmp) / "bad.npz", W1=np.zeros((784, 5)))
        try:
            Network.load(Path(tmp) / "bad.npz")
        except ValueError:
            pass
        else:
            raise AssertionError("битый файл модели должен вызывать понятную ошибку")
        np.savez(Path(tmp) / "old.npz", W1=np.zeros((784, 5)), b1=np.zeros(5),
                 W2=np.zeros((5, 10)), b2=np.zeros(10))              # модель старого формата
        try:
            Network.load(Path(tmp) / "old.npz")
        except ValueError as e:
            assert "старого формата" in str(e)
        else:
            raise AssertionError("модель из 10 классов должна отвергаться с понятным сообщением")
    try:
        Network.load(Path("no_such_model.npz"))
    except FileNotFoundError:
        return
    raise AssertionError("отсутствующий файл модели должен вызывать FileNotFoundError")


# ---------- данные ----------
@test
def data_validation():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.npz"
        np.savez(bad, x_train=np.zeros((10, 28, 28)), y_train=np.zeros(10),
                 x_test=np.zeros((5, 28, 28)), y_test=np.zeros(5))
        try:
            data._read_and_check(bad)
        except ValueError:
            return
    raise AssertionError("файл неправильного размера должен отвергаться")


@test
def to_input_range():
    x = data.to_input(np.array([[0, 255]], dtype=np.uint8))
    assert x.dtype == np.float64 and x.min() == 0.0 and x.max() == 1.0


# ---------- обработка рисунка ----------
def fake_digit(top, left, height, width, thick):
    """Рисуем фигуру в виде буквы «Г»-образной цифры на поле 280x280."""
    img = np.zeros((280, 280))
    img[top:top + thick, left:left + width] = 1.0            # верхняя перекладина
    img[top:top + height, left + width - thick:left + width] = 1.0   # правая вертикаль
    return img


@test
def preprocess_output_format():
    out = preprocess(fake_digit(50, 60, 180, 120, 20))
    assert out.shape == (1, 784), f"форма {out.shape}"
    assert 0.0 <= out.min() and out.max() <= 1.0 and out.max() > 0.5
    img = out.reshape(28, 28)
    ys, xs = np.where(img > 0.1)
    assert ys.max() - ys.min() + 1 <= 20 and xs.max() - xs.min() + 1 <= 20, "цифра не вписана в рамку 20x20"
    cy = (img.sum(axis=1) * np.arange(28)).sum() / img.sum()
    cx = (img.sum(axis=0) * np.arange(28)).sum() / img.sum()
    assert abs(cy - 13.5) < 1.0 and abs(cx - 13.5) < 1.0, f"центр масс ({cy:.1f}, {cx:.1f}) не в центре"


@test
def preprocess_empty_canvas():
    assert preprocess(np.zeros((280, 280))) is None


@test
def preprocess_ignores_size_and_position():
    """Одна и та же фигура, нарисованная большой, маленькой и в углу, даёт почти одну картинку."""
    a = preprocess(fake_digit(40, 40, 200, 140, 24)).ravel()
    b = preprocess(fake_digit(100, 150, 100, 70, 12)).ravel()
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cos > 0.85, f"картинки слишком разные (сходство {cos:.2f})"


@test
def resize_area_keeps_average():
    a = np.arange(100, dtype=float).reshape(10, 10)
    assert resize_area(a, 5, 5).shape == (5, 5)
    assert abs(resize_area(a, 5, 5).mean() - a.mean()) < 1e-9
    assert resize_area(np.ones((3, 3)), 20, 20).min() == 1.0   # увеличение тоже работает


# ---------- класс «не цифра» ----------
@test
def decide_rejects_unsure_and_unknown():
    probs = np.array([[0.9] + [0.1 / 10] * 10,                 # уверенная цифра 0
                      [0.4, 0.35] + [0.25 / 9] * 9,            # не уверена: ниже порога
                      [0.01] * 10 + [0.9]])                    # выбрала класс «не цифра»
    labels = decide(probs, threshold=MIN_CONFIDENCE)
    assert list(labels) == [0, UNKNOWN, UNKNOWN], f"получено {list(labels)}"


@test
def paint_line_draws_inside_and_survives_edges():
    p = np.zeros((SIZE, SIZE))
    paint_line(p, 20, 20, 200, 20)
    assert p[20, 100] == 1.0 and p[100, 100] == 0.0, "штрих нарисован не там"
    paint_line(p, -100, -100, 500, 500)                         # выход за границы не должен падать
    assert p.shape == (SIZE, SIZE)


@test
def unknown_generator_output():
    a = unknown.make_unknown(60, seed=1)
    assert a.shape == (60, 784) and a.dtype == np.uint8
    assert (a.reshape(60, -1).max(axis=1) > 100).all(), "в каждой картинке должен быть штрих"
    assert (a != a[0]).any(axis=1).mean() > 0.9, "картинки не должны быть одинаковыми"
    assert np.array_equal(a, unknown.make_unknown(60, seed=1)), "один seed - одни и те же картинки"
    assert not np.array_equal(a, unknown.make_unknown(60, seed=2)), "разные seed - разные картинки"


@test
def training_set_assembly():
    x = np.zeros((30, 784), dtype=np.uint8)
    y = np.arange(30) % 10
    xs, ys = train.build_training_set(x, y, n_unknown=10)
    assert xs.shape == (40, 784) and ys.shape == (40,)
    assert (ys == UNKNOWN).sum() == 10 and (ys < 10).sum() == 30


@test
def unknown_class_is_learnable_pipeline():
    """Сквозная проверка «обвязки»: цифры (синтетические) + сгенерированные «не цифры»
    -> обучение -> сеть отличает одно от другого. Это проверка кода, а не качества на MNIST."""
    rng = np.random.default_rng(0)
    protos = (rng.random((10, 784)) > 0.8).astype(float)         # 10 разных «цифр»
    n = 600
    y = rng.integers(0, 10, n)
    x_dig = np.clip(protos[y] * 200 + rng.normal(0, 25, (n, 784)), 0, 255).astype(np.uint8)
    x_unk = unknown.make_unknown(100, seed=7)
    xs = np.concatenate([x_dig, x_unk])
    ys = np.concatenate([y, np.full(100, UNKNOWN)])
    net = Network(n_hidden=32)
    rng2 = np.random.default_rng(1)
    for _ in range(20):                                           # короткое обучение без печати
        order = rng2.permutation(len(xs))
        for s in range(0, len(xs), 32):
            idx = order[s:s + 32]
            net.step(net.backward(net.forward(data.to_input(xs[idx])), ys[idx]), lr=0.05)
    test_unk = unknown.make_unknown(100, seed=8)
    recall = (net.predict(data.to_input(test_unk)) == UNKNOWN).mean()
    digits_ok = (net.predict(data.to_input(x_dig)) == y).mean()
    assert recall > 0.9, f"«не цифры» узнаются плохо: {recall:.2f}"
    assert digits_ok > 0.9, f"цифры узнаются плохо: {digits_ok:.2f}"


# ---------- окошко (только если есть tkinter и экран) ----------
@test
def window_smoke_test():
    if app.tk is None:
        return "пропущено: tkinter не установлен (sudo apt install python3-tk)"
    try:
        root = app.tk.Tk()
    except app.tk.TclError:
        return "пропущено: нет графического экрана"
    root.withdraw()
    try:
        w = app.App(root, Network(n_hidden=16))
        w.predict()                                   # пустое поле: ничего не происходит, ошибки нет
        assert w.answer is None
        w.on_press(type("E", (), {"x": 80, "y": 40})())
        for y in range(40, 240, 10):
            w.on_move(type("E", (), {"x": 140, "y": y})())
        assert w.pixels.sum() > 0, "штрих не попал в массив pixels"
        w.draw_line(-50, -50, 400, 400)               # выход за края поля не должен падать
        w.predict()
        assert w.answer in range(11), f"странный ответ в окошке: {w.answer!r}"
        assert len(w.rows) == 11
        assert abs(sum(w.values) - 1.0) < 1e-6, "вероятности в окошке должны давать 1"
        # самый вероятный - синий, остальные - светло-синие (ровно два цвета)
        fills = [bar.itemcget(fill, "fill") for _, bar, fill, _ in w.rows]
        assert fills[w.answer] == app.BLUE
        assert all(f == app.LIGHT for k, f in enumerate(fills) if k != w.answer)
        w.clear()
        assert w.answer is None and w.pixels.sum() == 0 and sum(w.values) == 0
        assert all(bar.itemcget(fill, "fill") == app.LIGHT for _, bar, fill, _ in w.rows)
    finally:
        root.destroy()


def main():
    failed = 0
    for t in TESTS:
        try:
            note = t()
            print(f"  OK    {t.__name__}" + (f"  ({note})" if note else ""))
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    print()
    print("Датасет:", "скачан" if data.NPZ_PATH.exists() else "ещё не скачан (скачается при python train.py)")
    from network import MODEL_PATH
    print("Модель: ", "есть" if MODEL_PATH.exists() else "ещё не обучена (python train.py)")
    print()
    if failed:
        print(f"Не прошло проверок: {failed}")
        sys.exit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
