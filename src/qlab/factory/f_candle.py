"""Candle Geometry: форма бара, объективно, без названий паттернов."""
import numpy as np
import pandas as pd

from qlab.factory import family, safe_div

CAT, CODE = "Candle Geometry", "CG"
D = ("open", "high", "low", "close")


def top(x): return np.maximum(x.o, x.c)
def bot(x): return np.minimum(x.o, x.c)


SIMPLE = [
    ("cg_body_range", "(c − o) / (h − l)", "[−1,1]", "сила и знак тела относительно размаха", lambda x: safe_div(x.c - x.o, x.rng)),
    ("cg_upper_range", "(h − max(o,c)) / (h − l)", "доля [0,1]", "длину верхней тени относительно размаха", lambda x: safe_div(x.h - top(x), x.rng)),
    ("cg_lower_range", "(min(o,c) − l) / (h − l)", "доля [0,1]", "длину нижней тени относительно размаха", lambda x: safe_div(bot(x) - x.l, x.rng)),
    ("cg_clv", "(2c − h − l) / (h − l)", "[−1,1]", "где закрылся бар внутри своего размаха", lambda x: safe_div(2 * x.c - x.h - x.l, x.rng)),
    ("cg_body_atr", "(c − o) / a", "ATR", "тело бара в ATR", lambda x: (x.c - x.o) / x.a),
    ("cg_range_atr", "(h − l) / a", "ATR", "размах бара в ATR", lambda x: x.rng / x.a),
    ("cg_upper_atr", "(h − max(o,c)) / a", "ATR", "верхнюю тень в ATR", lambda x: (x.h - top(x)) / x.a),
    ("cg_lower_atr", "(min(o,c) − l) / a", "ATR", "нижнюю тень в ATR", lambda x: (bot(x) - x.l) / x.a),
    ("cg_body2_atr", "(c_t − o_(t−1)) / a", "ATR", "тело двух последних баров вместе", lambda x: (x.c - x.o.shift()) / x.a),
    ("cg_body3_atr", "(c_t − o_(t−2)) / a", "ATR", "тело трёх последних баров вместе", lambda x: (x.c - x.o.shift(2)) / x.a),
    ("cg_overlap_prev", "пересечение [l,h] с [l,h]_(t−1) / (h − l)", "доля [0,1]", "насколько бар перекрывает предыдущий",
     lambda x: safe_div((np.minimum(x.h, x.h.shift()) - np.maximum(x.l, x.l.shift())).clip(lower=0), x.rng)),
]
for name, formula, unit, what, fn in SIMPLE:
    family(CAT, CODE, name, None, formula=formula, kind="непрерывный", unit=unit, deps=D + ("atr",), cost="O(n)",
           mechanism="Форма бара — след баланса покупателей и продавцов внутри интервала",
           measures=what, why="Объективная замена свечных паттернов", limits="Один бар — мало информации; зависит от границ ТФ", fn=fn)
FLAGS = [
    ("cg_dir", "sign(c − o)", "−1 / 0 / +1", "направление бара", lambda x: np.sign(x.c - x.o)),
    ("cg_inside", "h ≤ h_(t−1) и l ≥ l_(t−1)", "0/1", "бар внутри предыдущего", lambda x: ((x.h <= x.h.shift()) & (x.l >= x.l.shift())).astype(float)),
    ("cg_outside", "h > h_(t−1) и l < l_(t−1)", "0/1", "бар поглощает размах предыдущего", lambda x: ((x.h > x.h.shift()) & (x.l < x.l.shift())).astype(float)),
    ("cg_engulf", "outside и sign(c−o) ≠ sign(c−o)_(t−1)", "0/1", "внешний бар со сменой направления",
     lambda x: ((x.h > x.h.shift()) & (x.l < x.l.shift()) & (np.sign(x.c - x.o) != np.sign(x.c - x.o).shift())).astype(float)),
    ("cg_small_body", "|c − o| / (h − l) < 0.1", "0/1", "почти нет тела (нерешительность)", lambda x: (safe_div((x.c - x.o).abs(), x.rng) < 0.1).astype(float)),
    ("cg_full_body", "|c − o| / (h − l) > 0.9", "0/1", "почти нет теней (односторонний бар)", lambda x: (safe_div((x.c - x.o).abs(), x.rng) > 0.9).astype(float)),
]
for name, formula, unit, what, fn in FLAGS:
    family(CAT, CODE, name, None, formula=formula, kind="категориальный" if name == "cg_dir" else "бинарный", unit=unit,
           deps=D, cost="O(n)", mechanism="Дискретная форма бара", measures=what, why="Простые объективные состояния бара",
           limits="Пороги 0.1 и 0.9 условны", fn=fn)
for base, lag in (("cg_body_atr", 1), ("cg_body_atr", 2), ("cg_clv", 1), ("cg_clv", 2), ("cg_range_atr", 1)):
    fn = dict((n, f) for n, _, _, _, f in SIMPLE)[base]
    family(CAT, CODE, f"{base}_lag{lag}", None, formula=f"{base} бара t−{lag}", kind="непрерывный", unit="как у базового",
           deps=D + ("atr",), cost="O(n)", mechanism="Последовательность форм баров", measures=f"{base} {lag} бар(а) назад",
           why="Короткая память формы", limits="Чем больше лагов, тем больше избыточность",
           fn=lambda x, fn=fn, lag=lag: fn(x).shift(lag))
