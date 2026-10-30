"""Gap, Auction, Distance Metrics: разрывы, принятие цены и расстояния до объективных уровней."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, safe_div

GP = ("Gap", "GP")
family(*GP, "gp_open_gap", None, formula="(o_t − c_(t−1)) / a", kind="непрерывный", unit="ATR", deps=("open", "close", "atr"),
       cost="O(n)", mechanism="Разрыв между барами (на 24/7 рынке — признак сбоя или скачка)", measures="разрыв открытия бара",
       why="Выявляет скачки и артефакты", limits="Почти всегда ≈ 0", fn=lambda x: (x.o - x.c.shift()) / x.a)
family(*GP, "gp_max_gap_288", None, formula="max_288 |o − c_(t−1)| / a", kind="непрерывный", unit="ATR", deps=("open", "close", "atr"),
       cost="O(n)", mechanism="Недавний разрыв — нестабильность", measures="наибольший разрыв за 288 баров", why="Память о скачке", limits="—",
       fn=lambda x: (x.o - x.c.shift()).abs().rolling(288, min_periods=288).max() / x.a)


def _period_anchor(x, freq):
    key = x.day if freq == "D" else x.week
    return x.memo(("anchor", freq), lambda: pd.DataFrame({"open": x.o.groupby(key).transform("first"), "key": key}))


def _prev_day(x):
    def make():
        d = pd.DataFrame({"day": x.day, "h": x.h, "l": x.l, "c": x.c})
        daily = d.groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).shift(1)
        return daily.reindex(x.day.to_numpy()).set_axis(x.b.index)
    return x.memo("prev_day", make)


family(*GP, "gp_prev_day_close_dist", None, formula="(c − закрытие прошлых суток UTC) / a", kind="непрерывный", unit="ATR",
       deps=("close", "atr", "ts"), cost="O(n)", mechanism="Якорение к вчерашнему закрытию", measures="удаление от вчерашнего закрытия",
       why="Объективный якорь дня", limits="Сутки по UTC", fn=lambda x: (x.c - _prev_day(x)["c"]) / x.a)
family(*GP, "gp_week_open_dist", None, formula="(c − открытие недели) / a", kind="непрерывный", unit="ATR",
       deps=("open", "close", "atr", "ts"), cost="O(n)", mechanism="Якорение к открытию недели", measures="удаление от открытия недели",
       why="Недельная фаза", limits="Граница недели условна", fn=lambda x: (x.c - _period_anchor(x, "W")["open"]) / x.a)

AU = ("Auction", "AU")
family(*AU, "au_acceptance_{W}", grid(W=(24, 96)), formula="mean_W [l_i ≤ c_t ≤ h_i], i = t−W+1..t", kind="непрерывный",
       unit="доля [0,1]", deps=("high", "low", "close"), cost="O(n·W)", mechanism="Время, проведённое у цены = её принятие (теория аукциона)",
       measures="доля последних {W} баров, чей диапазон содержит текущую цену", why="Принятие против отвержения цены",
       limits="Не взвешено объёмом",
       fn=lambda x, W: pd.Series(np.mean([((x.l.shift(k) <= x.c) & (x.c <= x.h.shift(k))).to_numpy() for k in range(W)], axis=0),
                                 index=x.b.index).where(pd.Series(np.arange(len(x.b)) >= W - 1, index=x.b.index)))
family(*AU, "au_prev_day_pos", None, formula="(c − min вчера) / (max вчера − min вчера)", kind="непрерывный", unit="доля (может выходить за [0,1])",
       deps=("high", "low", "close", "ts"), cost="O(n)", mechanism="Цена внутри или вне вчерашнего баланса", measures="положение в диапазоне прошлых суток",
       why="Вне вчерашнего диапазона — поиск нового баланса", limits="Сутки по UTC",
       fn=lambda x: safe_div(x.c - _prev_day(x)["l"], _prev_day(x)["h"] - _prev_day(x)["l"]))
family(*AU, "au_prev_day_high_dist", None, formula="(max вчера − c) / a", kind="непрерывный", unit="ATR", deps=("high", "close", "atr", "ts"),
       cost="O(n)", mechanism="Вчерашний максимум — видимый уровень", measures="расстояние до вчерашнего максимума", why="Объективный уровень",
       limits="—", fn=lambda x: (_prev_day(x)["h"] - x.c) / x.a)
family(*AU, "au_prev_day_low_dist", None, formula="(c − min вчера) / a", kind="непрерывный", unit="ATR", deps=("low", "close", "atr", "ts"),
       cost="O(n)", mechanism="Вчерашний минимум — видимый уровень", measures="расстояние до вчерашнего минимума", why="Объективный уровень",
       limits="—", fn=lambda x: (x.c - _prev_day(x)["l"]) / x.a)


def _poc(x):
    def make():
        vmax = x.v.groupby(x.day).cummax()
        price = x.c.where(x.v >= vmax)
        return price.groupby(x.day).ffill()
    return x.memo("poc", make)


family(*AU, "au_poc_dist", None, formula="(c − закрытие самого объёмного бара суток) / a", kind="непрерывный", unit="ATR",
       deps=("close", "volume", "atr", "ts"), cost="O(n)", mechanism="Цена максимального объёма — центр принятия дня",
       measures="удаление от цены самого объёмного бара текущих суток", why="Грубый POC без профиля объёма",
       limits="Цена бара, а не уровень профиля", fn=lambda x: (x.c - _poc(x)) / x.a)


def _ib_state(x):
    first = (x.ts - x.day) < pd.Timedelta(hours=1)
    hi = x.h.where(first).groupby(x.day).cummax().groupby(x.day).ffill()
    lo = x.l.where(first).groupby(x.day).cummin().groupby(x.day).ffill()
    st = pd.Series(np.select([x.c > hi, x.c < lo], [1.0, -1.0], 0.0), index=x.b.index)
    return st.where(~first & hi.notna())


family(*AU, "au_initial_balance_state", None, formula="+1, если c выше диапазона первого часа суток UTC; −1 — ниже; 0 — внутри",
       kind="категориальный", unit="−1 / 0 / +1", deps=("high", "low", "close", "ts"), cost="O(n)",
       mechanism="Выход из начального баланса — признак трендового дня (теория аукциона)", measures="состояние относительно первого часа суток",
       why="Тип дня", limits="Первый час по UTC условен; NaN в первый час", fn=_ib_state)
family(*AU, "au_value_width", None, formula="sqrt(Σv(tp − VWAP)² / Σv) с начала суток / a", kind="непрерывный", unit="ATR",
       deps=("high", "low", "close", "volume", "atr"), cost="O(n)", mechanism="Ширина зоны, где прошёл объём дня",
       measures="разброс цены, взвешенный объёмом, с начала суток", why="Узкая зона стоимости — баланс", limits="—",
       fn=lambda x: (lambda tp: np.sqrt(((tp ** 2 * x.v).groupby(x.day).cumsum() / x.v.groupby(x.day).cumsum()
                                         - ((tp * x.v).groupby(x.day).cumsum() / x.v.groupby(x.day).cumsum()) ** 2).clip(lower=0)) / x.a)((x.h + x.l + x.c) / 3))

DS = ("Distance Metrics", "DS")


def _round(x, div):
    step = 10 ** (np.floor(np.log10(x.c)) - 1) / div
    m = np.mod(x.c, step)
    return np.minimum(m, step - m) / x.a


family(*DS, "ds_round_major", None, formula="расстояние до кратного 10^(⌊log10 c⌋−1) / a (ETH 3000 → шаг 100)", kind="непрерывный",
       unit="ATR", deps=("close", "atr"), cost="O(n)", mechanism="Скопление ордеров у круглых чисел (Osler)",
       measures="расстояние до ближайшего крупного круглого уровня", why="Объективный уровень без разметки", limits="Шаг меняется при смене порядка цены",
       fn=lambda x: _round(x, 1))
family(*DS, "ds_round_minor", None, formula="расстояние до кратного 10^(⌊log10 c⌋−1)/2 / a (ETH 3000 → шаг 50)", kind="непрерывный",
       unit="ATR", deps=("close", "atr"), cost="O(n)", mechanism="Скопление ордеров у полукруглых чисел",
       measures="расстояние до ближайшего полукруглого уровня", why="Более частая сетка уровней", limits="—", fn=lambda x: _round(x, 2))
for D in (7, 30):
    family(*DS, f"ds_high_{D}d", None, formula=f"(max h за {D} суток − c) / a", kind="непрерывный", unit="ATR", deps=("high", "close", "atr"),
           cost="O(n)", mechanism="Дальние экстремумы — уровни свинг-участников", measures=f"расстояние до максимума {D} суток",
           why="Контекст старшего масштаба", limits="—", fn=lambda x, D=D: (x.hmax(D * x.bpd) - x.c) / x.a)
    family(*DS, f"ds_low_{D}d", None, formula=f"(c − min l за {D} суток) / a", kind="непрерывный", unit="ATR", deps=("low", "close", "atr"),
           cost="O(n)", mechanism="Дальние экстремумы — уровни свинг-участников", measures=f"расстояние до минимума {D} суток",
           why="Контекст старшего масштаба", limits="—", fn=lambda x, D=D: (x.c - x.lmin(D * x.bpd)) / x.a)
family(*DS, "ds_drawdown_from_max", None, formula="ln(c / max c с начала данных)", kind="непрерывный", unit="доля (≤ 0)", deps=("close",),
       cost="O(n)", mechanism="Глубина просадки от исторического максимума данных", measures="просадка от максимума всей доступной истории",
       why="Долгосрочный контекст", limits="Зависит от начала данных", fn=lambda x: np.log(x.c / x.c.cummax()))
