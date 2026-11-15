"""Окошко: рисуешь цифру мышкой, нажимаешь «Определить» - справа видно, что думает сеть.

Запуск:  python app.py      (сначала нужна обученная модель: python train.py)
Клавиши: Enter - определить, Esc - очистить.

Окошко ничего не «решает» за тебя: оно показывает вероятности всех 11 классов
(цифры 0-9 и «не цифра»), самый вероятный выделен синим. Выводы делаешь ты.
"""
import sys

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk
except ImportError:                      # tkinter ставится отдельно: sudo apt install python3-tk
    tk = ttk = tkfont = None

import numpy as np

from network import N_CLASSES, Network
from preprocess import RADIUS, SIZE, paint_line, preprocess

# ---------- оформление: всего два цвета ----------
BLUE = "#2563eb"        # самый вероятный вариант
LIGHT = "#bfdbfe"       # всё остальное (полоски, кнопка «Очистить»)
BG = "white"
TEXT = "#222222"

CELL = 5                # размер клетки в превью 28x28 (пиксели)
BAR_W, BAR_H = 140, 10  # размер полоски вероятности
NAMES = [str(k) for k in range(10)] + ["не цифра"]


class App:
    def __init__(self, root, net):
        self.root, self.net = root, net
        self.pixels = np.zeros((SIZE, SIZE))          # нарисованное: 0 - фон, 1 - штрих
        self.last = None                              # где была мышь в прошлый раз
        self.answer = None                            # самый вероятный класс (0..10) или None
        self.values = [0.0] * N_CLASSES               # последние вероятности

        root.title("Распознавание цифр")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._make_style()
        self._build_left()
        self._build_right()
        root.bind("<Return>", lambda event: self.predict())
        root.bind("<Escape>", lambda event: self.clear())

    # ---------- построение окна ----------
    def _make_style(self):
        family = tkfont.nametofont("TkDefaultFont").actual("family")     # родной шрифт системы
        self.font = tkfont.Font(root=self.root, family=family, size=11)
        self.font_bold = tkfont.Font(root=self.root, family=family, size=11, weight="bold")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")        # в этой теме цвета кнопок работают на всех системах
        except tk.TclError:
            pass
        for name, bg, fg in (("TButton", LIGHT, TEXT), ("Primary.TButton", BLUE, "white")):
            style.configure(name, font=self.font, padding=(14, 8), background=bg, foreground=fg,
                            borderwidth=0, focusthickness=0, focuscolor=bg)
            style.map(name, background=[("active", bg)])

    def _build_left(self):
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="n")
        self.canvas = tk.Canvas(left, width=SIZE, height=SIZE, bg="black", cursor="crosshair",
                                highlightthickness=0, borderwidth=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)

        buttons = tk.Frame(left, bg=BG)
        buttons.pack(fill="x", pady=(12, 0))
        buttons.columnconfigure((0, 1), weight=1, uniform="b")
        ttk.Button(buttons, text="Определить", style="Primary.TButton",
                   command=self.predict).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(buttons, text="Очистить",
                   command=self.clear).grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _build_right(self):
        right = tk.Frame(self.root, bg=BG)
        right.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="n")

        tk.Label(right, text="Что видит сеть (28×28)", font=self.font_bold, fg=TEXT,
                 bg=BG).pack(anchor="w")
        self.preview = tk.Canvas(right, width=28 * CELL, height=28 * CELL, bg="black",
                                 highlightthickness=0, borderwidth=0)
        self.preview.pack(anchor="w", pady=(8, 14))

        rows = tk.Frame(right, bg=BG)
        rows.pack(anchor="w")
        self.rows = []                                # (подпись, полоска, прямоугольник, проценты)
        for k, name in enumerate(NAMES):
            label = tk.Label(rows, text=name, font=self.font, fg=TEXT, bg=BG, width=8, anchor="w")
            label.grid(row=k, column=0, pady=2, sticky="w")
            bar = tk.Canvas(rows, width=BAR_W, height=BAR_H, bg=BG,
                            highlightthickness=0, borderwidth=0)
            bar.grid(row=k, column=1, padx=10)
            fill = bar.create_rectangle(0, 0, 0, BAR_H, width=0, fill=LIGHT)
            percent = tk.Label(rows, text="0.0%", font=self.font, fg=TEXT, bg=BG, width=6, anchor="e")
            percent.grid(row=k, column=2, sticky="e")
            self.rows.append((label, bar, fill, percent))

    # ---------- рисование ----------
    def on_press(self, event):
        self.last = None
        self.draw_line(event.x, event.y, event.x, event.y)

    def on_move(self, event):
        self.draw_line(event.x, event.y, event.x, event.y)

    def draw_line(self, x0, y0, x1, y1):
        """Рисует штрих от предыдущей точки до текущей (и на экране, и в массиве pixels)."""
        if self.last is not None:
            x0, y0 = self.last
        paint_line(self.pixels, x0, y0, x1, y1)
        if (x0, y0) == (x1, y1):                               # одиночный клик - рисуем точку
            self.canvas.create_oval(x1 - RADIUS, y1 - RADIUS, x1 + RADIUS, y1 + RADIUS,
                                    fill="white", outline="white")
        else:
            self.canvas.create_line(x0, y0, x1, y1, fill="white", width=2 * RADIUS, capstyle="round")
        self.last = (x1, y1)

    def clear(self):
        self.pixels[:] = 0.0
        self.last = None
        self.answer = None
        self.canvas.delete("all")
        self.preview.delete("all")
        self.show_values(np.zeros(N_CLASSES), winner=None)

    # ---------- предсказание ----------
    def predict(self):
        x = preprocess(self.pixels)
        if x is None:                                         # ничего не нарисовано
            return
        p = self.net.forward(x)[0]                            # 11 вероятностей
        self.answer = int(p.argmax())
        self.show_values(p, winner=self.answer)
        self.show_preview(x.reshape(28, 28))

    def show_values(self, p, winner):
        """Самый вероятный класс - синий и жирный, остальные - светло-синие."""
        self.values = [float(v) for v in p]
        for k, (label, bar, fill, percent) in enumerate(self.rows):
            font = self.font_bold if k == winner else self.font
            bar.coords(fill, 0, 0, BAR_W * float(p[k]), BAR_H)
            bar.itemconfig(fill, fill=BLUE if k == winner else LIGHT)
            label.config(font=font)
            percent.config(text=f"{p[k] * 100:.1f}%", font=font)

    def show_preview(self, img):
        self.preview.delete("all")
        for i in range(28):
            for j in range(28):
                g = int(img[i, j] * 255)
                if g > 0:
                    self.preview.create_rectangle(j * CELL, i * CELL, (j + 1) * CELL, (i + 1) * CELL,
                                                  fill=f"#{g:02x}{g:02x}{g:02x}", outline="")


def main():
    if tk is None:
        sys.exit("Не найден tkinter. Установи: sudo apt install python3-tk "
                 "(на Windows/macOS он входит в установщик Python с python.org)")
    try:
        net = Network.load()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    root = tk.Tk()
    App(root, net)
    root.mainloop()


if __name__ == "__main__":
    main()