"""Volume: сколько и как торговали."""
import numpy as np
import pandas as pd

from qlab.factory import WIN, family, grid, rolling_z, safe_div

CAT, CODE = "Volume", "VO"
family(CAT, CODE, "vo_rel_{W}", grid(W=WIN), formula="ln( v_t / mean(v_(t−1..t−W)) )", kind="непрерывный",
       unit="ln-отношение", deps=("volume",), cost="O(n)", mechanism="Всплеск участия: новость, каскад, приход крупного игрока",
       measures="объём бара относительно предыдущих {W} баров", why="Базовый индикатор аномального участия",
       limits="Смешивает время суток с аномалией",
       fn=lambda x, W: np.log(safe_div(x.v, x.v.shift().rolling(W, min_periods=W).mean())))
family(CAT, CODE, "vo_z_{W}", grid(W=(24, 288)), formula="z_W( ln v )", kind="непрерывный", unit="σ", deps=("volume",),
       cost="O(n)", mechanism="Аномальность объёма в единицах его разброса", measures="z-оценка логарифма объёма за {W} баров",
       why="Учитывает изменчивость объёма", limits="Логнормальность — приближение",
       fn=lambda x, W: rolling_z(np.log(x.v.where(x.v > 0)), W))
family(CAT, CODE, "vo_trend", None, formula="ln( mean_24 v / mean_288 v )", kind="непрерывный", unit="ln-отношение",
       deps=("volume",), cost="O(n)", mechanism="Нарастание участия перед движением", measures="тренд объёма: 24 бара против 288",
       why="Медленная составляющая активности", limits="Сезонность суток",
       fn=lambda x: np.log(safe_div(x.v.rolling(24, min_periods=24).mean(), x.v.rolling(288, min_periods=288).mean())))


def _tod_rel(x):
    lags = pd.DataFrame({k: x.v.shift(k * x.bpd) for k in range(1, 21)})
    return np.log(safe_div(x.v, lags.median(axis=1, skipna=True).where(lags.notna().sum(axis=1) >= 10)))


family(CAT, CODE, "vo_time_of_day_rel", None, formula="ln( v_t / median(v в то же время суток за 20 прошлых дней) )",
       kind="непрерывный", unit="ln-отношение", deps=("volume",), cost="O(20n)",
       mechanism="Аномалия объёма с поправкой на внутридневную сезонность",
       measures="объём относительно обычного для этого времени суток", why="Чистая аномалия без эффекта часа",
       limits="Требует 20 дней истории", fn=_tod_rel)
family(CAT, CODE, "vo_up_share_{W}", grid(W=(24, 96, 288)), formula="Σ_W v[r>0] / Σ_W v", kind="непрерывный",
       unit="доля [0,1]", deps=("volume", "close"), cost="O(n)", mechanism="Объём на росте против объёма на падении",
       measures="доля объёма растущих баров за {W} баров", why="Грубая оценка накопления/распределения",
       limits="Направление бара ≠ направление сделок",
       fn=lambda x, W: safe_div((x.v * (x.r > 0)).rolling(W, min_periods=W).sum(), x.v.rolling(W, min_periods=W).sum()))
family(CAT, CODE, "vo_absr_corr_{W}", grid(W=(96, 288)), formula="corr_W( |r|, ln v )", kind="непрерывный",
       unit="[−1,1]", deps=("volume", "close"), cost="O(n)", mechanism="Связь объёма и движения (Karpoff, 1987)",
       measures="насколько движения сопровождаются объёмом за {W} баров",
       why="Движение без объёма и движение на объёме — разные режимы", limits="Шумно",
       fn=lambda x, W: x.r.abs().rolling(W, min_periods=W).corr(np.log(x.v.where(x.v > 0))))
family(CAT, CODE, "vo_concentration_{W}", grid(W=(24, 288)), formula="max_W v / Σ_W v", kind="непрерывный",
       unit="доля [0,1]", deps=("volume",), cost="O(n)", mechanism="Один бар собрал объём окна — кульминация",
       measures="доля объёма окна {W} баров в самом объёмном баре", why="Кульминация против равномерной активности",
       limits="Не говорит, где был этот бар",
       fn=lambda x, W: safe_div(x.v.rolling(W, min_periods=W).max(), x.v.rolling(W, min_periods=W).sum()))
