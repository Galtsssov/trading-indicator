"""Price Structure: где цена находится относительно недавнего диапазона и как он устроен."""
import numpy as np
import pandas as pd

from qlab.factory import WIN, family, grid, rolling_argext_age, safe_div

W = grid(W=WIN)
CAT, CODE = "Price Structure", "PS"
DEPS = ("high", "low", "close", "atr")

family(CAT, CODE, "ps_pos_range_{W}", W, formula="(c − min_W l) / (max_W h − min_W l)",
       kind="непрерывный", unit="доля [0,1]", deps=DEPS, cost="O(n)",
       mechanism="Участники ориентируются на края недавнего диапазона (стопы, тейки, лимитки)",
       measures="положение закрытия внутри диапазона последних {W} баров",
       why="Отличает покупку у дна диапазона от покупки у вершины при одинаковой доходности",
       limits="Не различает узкий и широкий диапазон; при пробое равно 0 или 1",
       fn=lambda x, W: safe_div(x.c - x.lmin(W), x.hmax(W) - x.lmin(W)))
family(CAT, CODE, "ps_dist_high_{W}", W, formula="(max_W h − c) / a", kind="непрерывный", unit="ATR",
       deps=DEPS, cost="O(n)", mechanism="Скопление ордеров у экстремумов окна",
       measures="расстояние от закрытия до максимума {W} баров в ATR",
       why="Даёт масштаб: сколько волатильности до ближайшего очевидного уровня сверху",
       limits="Окно фиксировано; уровни старше {W} баров не видны",
       fn=lambda x, W: (x.hmax(W) - x.c) / x.a)
family(CAT, CODE, "ps_dist_low_{W}", W, formula="(c − min_W l) / a", kind="непрерывный", unit="ATR",
       deps=DEPS, cost="O(n)", mechanism="Скопление ордеров у экстремумов окна",
       measures="расстояние от закрытия до минимума {W} баров в ATR",
       why="Масштаб до ближайшего очевидного уровня снизу", limits="Окно фиксировано",
       fn=lambda x, W: (x.c - x.lmin(W)) / x.a)
family(CAT, CODE, "ps_age_high_{W}", W, formula="(баров с момента max_W h) / W", kind="непрерывный",
       unit="доля [0,1]", deps=("high",), cost="O(n·W), W={W}",
       mechanism="Свежий максимум — импульс; старый — уровень, на котором успели накопиться ордера",
       measures="давность максимума окна {W} баров", why="Различает свежий импульс и давний уровень",
       limits="0 и при свежем пробое, и при постоянном росте",
       fn=lambda x, W: rolling_argext_age(x.h, W, True) / W)
family(CAT, CODE, "ps_age_low_{W}", W, formula="(баров с момента min_W l) / W", kind="непрерывный",
       unit="доля [0,1]", deps=("low",), cost="O(n·W), W={W}",
       mechanism="Свежий минимум — импульс; старый — уровень", measures="давность минимума окна {W} баров",
       why="Различает свежий импульс и давний уровень", limits="Как у ps_age_high",
       fn=lambda x, W: rolling_argext_age(x.l, W, False) / W)
family(CAT, CODE, "ps_range_{W}", W, formula="(max_W h − min_W l) / a", kind="непрерывный", unit="ATR",
       deps=DEPS, cost="O(n)", mechanism="Широкий диапазон — тренд или шок, узкий — баланс",
       measures="ширина диапазона {W} баров в ATR", why="Масштаб структуры относительно текущей волатильности",
       limits="Зависит от ATR предыдущего бара", fn=lambda x, W: (x.hmax(W) - x.lmin(W)) / x.a)
family(CAT, CODE, "ps_hh_minus_ll_{W}", W,
       formula="mean_W[h_t > h_(t−1)] − mean_W[l_t < l_(t−1)]", kind="непрерывный", unit="доля [−1,1]",
       deps=("high", "low"), cost="O(n)", mechanism="Последовательность более высоких максимумов — объективный тренд",
       measures="перевес баров с более высоким максимумом над барами с более низким минимумом за {W} баров",
       why="Объективная замена понятия «структура тренда»", limits="Не учитывает размер шагов",
       fn=lambda x, W: ((x.h > x.h.shift()).astype(float) - (x.l < x.l.shift()).astype(float))
       .rolling(W, min_periods=W).mean())
family(CAT, CODE, "ps_break_state_{W}", grid(W=WIN[1:]),
       formula="+1, если c > max_W h (без текущего бара); −1, если c < min_W l; иначе 0",
       kind="категориальный", unit="−1 / 0 / +1", deps=DEPS, cost="O(n)",
       mechanism="Закрытие за экстремумом окна активирует стопы и пробойные ордера",
       measures="закрылся ли бар за пределами диапазона прошлых {W} баров",
       why="Объективное определение пробоя", limits="Не различает глубину пробоя",
       fn=lambda x, W: pd.Series(np.select([x.c > x.hmax(W).shift(), x.c < x.lmin(W).shift()], [1.0, -1.0], 0.0),
                                 index=x.b.index).where(x.hmax(W).shift().notna()))
