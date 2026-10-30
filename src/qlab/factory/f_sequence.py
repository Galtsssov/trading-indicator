"""Sequence и Cluster Features: порядок событий и дискретные коды состояний (без обучения — пороги фиксированы)."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, streak

SQ = ("Sequence", "SQ")
family(*SQ, "sq_up_streak", None, formula="число закрытий подряд выше предыдущего", kind="счётчик", unit="бары", deps=("close",),
       cost="O(n)", mechanism="Серии — перекупленность или устойчивый поток", measures="длина текущей серии роста", why="Порядок, а не сумма",
       limits="—", fn=lambda x: streak(x.r > 0))
family(*SQ, "sq_down_streak", None, formula="число закрытий подряд ниже предыдущего", kind="счётчик", unit="бары", deps=("close",),
       cost="O(n)", mechanism="Серии — перепроданность или устойчивый поток", measures="длина текущей серии падения", why="Порядок, а не сумма",
       limits="—", fn=lambda x: streak(x.r < 0))
family(*SQ, "sq_pattern3", None, formula="код 0–7 из знаков трёх последних доходностей (рост = 1): 4·s_(t−2) + 2·s_(t−1) + s_t",
       kind="категориальный", unit="0–7", deps=("close",), cost="O(n)", mechanism="Короткие узоры направления",
       measures="узор направлений последних трёх баров", why="Дискретная последовательность", limits="Нулевая доходность считается падением",
       fn=lambda x: (4 * (x.r.shift(2) > 0) + 2 * (x.r.shift(1) > 0) + (x.r > 0)).astype(float).where(x.r.shift(2).notna()))
family(*SQ, "sq_max_run_{W}", grid(W=(24, 96)), formula="max_W(max(серия роста, серия падения))", kind="счётчик", unit="бары",
       deps=("close",), cost="O(n)", mechanism="Самая длинная однонаправленная серия окна", measures="длиннейшая серия за {W}",
       why="Наличие трендового эпизода", limits="—",
       fn=lambda x, W: np.maximum(streak(x.r > 0), streak(x.r < 0)).rolling(W, min_periods=W).max())
family(*SQ, "sq_higher_high_streak", None, formula="число баров подряд с h_t > h_(t−1)", kind="счётчик", unit="бары", deps=("high",),
       cost="O(n)", mechanism="Последовательные обновления максимумов", measures="серия более высоких максимумов", why="Объективная структура",
       limits="—", fn=lambda x: streak(x.h > x.h.shift()))
family(*SQ, "sq_lower_low_streak", None, formula="число баров подряд с l_t < l_(t−1)", kind="счётчик", unit="бары", deps=("low",),
       cost="O(n)", mechanism="Последовательные обновления минимумов", measures="серия более низких минимумов", why="Объективная структура",
       limits="—", fn=lambda x: streak(x.l < x.l.shift()))

CL = ("Cluster Features", "CF")
family(*CL, "cf_bar_shape", None, formula="3·bin(clv: < −1/3, иначе < 1/3, иначе) + bin((h−l)/a: < 0.8, < 1.5, иначе) → 0–8",
       kind="категориальный", unit="0–8", deps=("high", "low", "close", "atr"), cost="O(n)",
       mechanism="Тип бара: где закрылся × насколько большой", measures="дискретный код формы бара",
       why="Компактное состояние для поиска комбинаций", limits="Пороги фиксированы, не обучены",
       fn=lambda x: pd.Series(3 * np.digitize(((2 * x.c - x.h - x.l) / (x.h - x.l)).fillna(0), [-1 / 3, 1 / 3])
                              + np.digitize((x.rng / x.a).fillna(1), [0.8, 1.5]), index=x.b.index, dtype=float)
       .where(x.a.notna()))
family(*CL, "cf_vol_state", None, formula="2·[std_6 r > std_288 r] + [std_288 r > медианы std_288 за 30 суток] → 0–3",
       kind="категориальный", unit="0–3", deps=("close",), cost="O(n log W)", mechanism="Короткая и длинная фаза волатильности вместе",
       measures="состояние волатильности из двух бит", why="Компактный режим", limits="Грубо",
       fn=lambda x: (2 * (x.rv(6) > x.rv(288)) + (x.rv(288) > x.rv(288).rolling(30 * x.bpd, min_periods=x.bpd).median())).astype(float)
       .where(x.rv(288).notna()))
family(*CL, "cf_flow_state", None, formula="4·[доходность_6 > 0] + 2·[taker_share_6 > 0] + [ΔOI_6 > 0] → 0–7",
       kind="категориальный", unit="0–7", deps=("close", "taker_buy_volume", "volume", "oi"), cost="O(n)",
       mechanism="Цена, агрессия и позиции вместе: например, рост + покупки + рост OI = новые лонги",
       measures="состояние потока из трёх бит", why="Объективная версия идеи NIF как состояние, а не правило",
       limits="NaN без OI",
       fn=lambda x: (4 * (x.ret(6) > 0) + 2 * ((x.col("taker_buy_volume").rolling(6).sum() / x.v.rolling(6).sum()) > 0.5)
                     + (np.log(x.col("oi")).diff(6) > 0)).astype(float).where(x.col("oi").notna() & x.ret(6).notna()))
