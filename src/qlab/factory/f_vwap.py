"""VWAP: средневзвешенная по объёму цена — ориентир исполнения крупных заявок."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, safe_div

CAT, CODE = "VWAP", "VW"


def _tp(x):
    return (x.h + x.l + x.c) / 3


def _anchored(x, key):
    def make():
        tpv, v = _tp(x) * x.v, x.v
        g = key(x)
        vw = tpv.groupby(g).cumsum() / v.groupby(g).cumsum()
        ex2 = (_tp(x) ** 2 * x.v).groupby(g).cumsum() / v.groupby(g).cumsum()
        return pd.DataFrame({"vwap": vw, "sd": np.sqrt((ex2 - vw ** 2).clip(lower=0))})
    return x.memo(("vwap", key.__name__), make)


def day_key(x): return x.day
def week_key(x): return x.week


family(CAT, CODE, "vw_day_dist", None, formula="(c − VWAP_дня) / a, VWAP с 00:00 UTC", kind="непрерывный", unit="ATR",
       deps=("high", "low", "close", "volume", "atr"), cost="O(n)", mechanism="Исполнители по VWAP возвращают цену к нему",
       measures="удаление от дневного VWAP в ATR", why="Справедливая цена дня для крупных заявок",
       limits="В начале суток VWAP построен по малому числу баров", fn=lambda x: (x.c - _anchored(x, day_key)["vwap"]) / x.a)
family(CAT, CODE, "vw_week_dist", None, formula="(c − VWAP_недели) / a, неделя с понедельника 00:00 UTC", kind="непрерывный",
       unit="ATR", deps=("high", "low", "close", "volume", "atr"), cost="O(n)", mechanism="Справедливая цена недели",
       measures="удаление от недельного VWAP", why="Более медленный якорь", limits="Граница недели условна",
       fn=lambda x: (x.c - _anchored(x, week_key)["vwap"]) / x.a)
family(CAT, CODE, "vw_day_band_z", None, formula="(c − VWAP_дня) / sqrt(Σv·(tp − VWAP)² / Σv)", kind="непрерывный",
       unit="σ объёма", deps=("high", "low", "close", "volume"), cost="O(n)", mechanism="Насколько цена вне зоны, где прошёл объём дня",
       measures="удаление от VWAP в объёмно-взвешенных стандартных отклонениях", why="Нормирует на разброс цены дня",
       limits="Неустойчиво в первые бары дня",
       fn=lambda x: safe_div(x.c - _anchored(x, day_key)["vwap"], _anchored(x, day_key)["sd"]))
family(CAT, CODE, "vw_roll_dist_{W}", grid(W=(24, 96, 288)), formula="(c − Σ_W(tp·v)/Σ_W v) / a", kind="непрерывный",
       unit="ATR", deps=("high", "low", "close", "volume", "atr"), cost="O(n)", mechanism="Скользящая справедливая цена без привязки к суткам",
       measures="удаление от скользящего VWAP за {W} баров", why="Без разрыва в полночь", limits="—",
       fn=lambda x, W: (x.c - safe_div((_tp(x) * x.v).rolling(W, min_periods=W).sum(), x.v.rolling(W, min_periods=W).sum())) / x.a)
family(CAT, CODE, "vw_day_slope_{W}", grid(W=(6, 24)), formula="(VWAP_дня,t − VWAP_дня,t−W) / a", kind="непрерывный",
       unit="ATR", deps=("high", "low", "close", "volume", "atr"), cost="O(n)", mechanism="Куда смещается справедливая цена",
       measures="наклон дневного VWAP за {W} баров", why="Направление «средней сделки»", limits="Скачок в полночь",
       fn=lambda x, W: (_anchored(x, day_key)["vwap"] - _anchored(x, day_key)["vwap"].shift(W)) / x.a)
family(CAT, CODE, "vw_above_frac_{W}", grid(W=(24, 96)), formula="mean_W [c > VWAP_дня]", kind="непрерывный",
       unit="доля [0,1]", deps=("close", "volume"), cost="O(n)", mechanism="Устойчивое принятие цены выше/ниже справедливой",
       measures="доля баров выше дневного VWAP за {W} баров", why="Принятие против разового выхода", limits="—",
       fn=lambda x, W: (x.c > _anchored(x, day_key)["vwap"]).astype(float).rolling(W, min_periods=W).mean())
family(CAT, CODE, "vw_crosses_24", None, formula="число смен знака (c − VWAP_дня) за 24 бара", kind="счётчик",
       unit="раз", deps=("close", "volume"), cost="O(n)", mechanism="Частые пересечения — баланс, редкие — тренд",
       measures="сколько раз цена пересекала VWAP за 24 бара", why="Объективная мера «баланса»", limits="—",
       fn=lambda x: np.sign(x.c - _anchored(x, day_key)["vwap"]).diff().ne(0).astype(float).rolling(24, min_periods=24).sum())
