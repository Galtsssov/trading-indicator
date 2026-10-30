"""Compression и Expansion: сжатие и расширение диапазона и волатильности."""
import numpy as np
import pandas as pd

from qlab.factory import bars_since, family, grid, rolling_pctl, safe_div, streak

CP = ("Compression", "CP")


def _bbw(x):
    m, s = x.c.rolling(20, min_periods=20).mean(), x.c.rolling(20, min_periods=20).std()
    return 4 * s / m


family(*CP, "cp_bb_width_pctl", None, formula="процентиль (4·std_20 c / SMA_20 c) за 30 суток", kind="непрерывный", unit="доля [0,1]",
       deps=("close",), cost="O(n log W)", mechanism="Сжатие полос Боллинджера — затишье перед расширением",
       measures="ширина полос относительно месяца", why="Классическая мера сжатия", limits="—",
       fn=lambda x: rolling_pctl(_bbw(x), 30 * x.bpd))
family(*CP, "cp_range_ratio_{W}", grid(W=(6, 24, 96)), formula="(max_W h − min_W l) / (max_4W h − min_4W l)", kind="непрерывный",
       unit="доля [0,1]", deps=("high", "low"), cost="O(n)", mechanism="Короткий диапазон внутри длинного — сжатие",
       measures="ширина {W}-барного диапазона относительно в 4 раза более длинного", why="Сжатие без ATR", limits="—",
       fn=lambda x, W: safe_div(x.hmax(W) - x.lmin(W), x.hmax(4 * W) - x.lmin(4 * W)))
family(*CP, "cp_narrowest_{k}", grid(k=(4, 7)), formula="1, если h − l — минимум за последние k баров (NR-k)", kind="бинарный",
       unit="0/1", deps=("high", "low"), cost="O(n)", mechanism="Самый узкий бар серии — пауза", measures="NR{k}",
       why="Объективное определение паузы", limits="—",
       fn=lambda x, k: (x.rng <= x.rng.rolling(k, min_periods=k).min()).astype(float))
family(*CP, "cp_days_since_low_vol", None, formula="(баров с последнего atr_pctl < 0.1) / баров в сутках", kind="непрерывный",
       unit="сутки", deps=("atr_pctl",), cost="O(n)", mechanism="Давность последнего сжатия", measures="сколько суток назад было сжатие",
       why="Фаза цикла волатильности", limits="NaN до первого сжатия", fn=lambda x: bars_since(x.col("atr_pctl") < 0.1) / x.bpd)
family(*CP, "cp_squeeze", None, formula="1, если полосы Боллинджера(20, 2σ) внутри каналов Кельтнера(20, 1.5·ATR)", kind="бинарный",
       unit="0/1", deps=("close", "atr"), cost="O(n)", mechanism="Волатильность закрытий меньше среднего размаха — сжатие",
       measures="сжатие по Боллинджеру–Кельтнеру", why="Распространённое объективное определение", limits="—",
       fn=lambda x: (2 * x.c.rolling(20, min_periods=20).std() < 1.5 * x.a).astype(float))

EX = ("Expansion", "EX")


def _atr(x, n):
    tr = np.maximum(x.rng, np.maximum((x.h - x.c.shift()).abs(), (x.l - x.c.shift()).abs()))
    return x.memo(("atr", n), lambda: tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean())


family(*EX, "ex_atr_ratio_{S}_{L}", [{"S": 5, "L": 50}, {"S": 14, "L": 100}], formula="ATR({S}) / ATR({L})", kind="непрерывный",
       unit="отношение", deps=("high", "low", "close"), cost="O(n)", mechanism="Короткий ATR выше длинного — расширение",
       measures="отношение ATR {S} и {L}", why="Фаза расширения", limits="—", fn=lambda x, S, L: _atr(x, S) / _atr(x, L))
family(*EX, "ex_atr_change_{W}", grid(W=(6, 24)), formula="ln(ATR_t / ATR_(t−W))", kind="непрерывный", unit="ln-отношение",
       deps=("atr",), cost="O(n)", mechanism="Скорость роста волатильности", measures="изменение ATR за {W} баров", why="Разгон",
       limits="—", fn=lambda x, W: np.log(x.col("atr") / x.col("atr").shift(W)))
family(*EX, "ex_expanding_streak", None, formula="число баров подряд с (h−l) > (h−l)_(t−1)", kind="счётчик", unit="бары",
       deps=("high", "low"), cost="O(n)", mechanism="Последовательное расширение", measures="серия расширяющихся баров",
       why="Ускорение активности", limits="—", fn=lambda x: streak(x.rng > x.rng.shift()))
family(*EX, "ex_days_since_high_vol", None, formula="(баров с последнего atr_pctl > 0.9) / баров в сутках", kind="непрерывный",
       unit="сутки", deps=("atr_pctl",), cost="O(n)", mechanism="Давность последнего шока волатильности",
       measures="сколько суток назад было расширение", why="Затухание после шока", limits="NaN до первого события",
       fn=lambda x: bars_since(x.col("atr_pctl") > 0.9) / x.bpd)
