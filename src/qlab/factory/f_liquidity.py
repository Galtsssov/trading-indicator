"""Liquidity: насколько легко двигать цену — оценки по OHLCV (полного стакана нет)."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, safe_div

CAT, CODE = "Liquidity", "LQ"

family(CAT, CODE, "lq_amihud_{W}", grid(W=(24, 288)), formula="log10( mean_W(|r| / quote_volume) )",
       kind="непрерывный", unit="log10(доля/USDT)", deps=("close", "quote_volume"), cost="O(n)",
       mechanism="Неликвидность (Amihud, 2002): сколько цены сдвигает каждый доллар объёма",
       measures="ценовой импакт на единицу объёма за {W} баров",
       why="В тонком рынке свипы и импульсы ведут себя иначе",
       limits="Смешивает спрос на ликвидность и информацию; зависит от уровня цены",
       fn=lambda x, W: np.log10(safe_div(x.r.abs(), x.col("quote_volume")).rolling(W, min_periods=W).mean()))
family(CAT, CODE, "lq_depth_proxy_{W}", grid(W=(6, 24, 96)),
       formula="ln[(Σ_W v / Σ_W (h−l)) / (Σ_8640 v / Σ_8640 (h−l))]", kind="непрерывный", unit="ln-отношение",
       deps=("volume", "high", "low"), cost="O(n)", mechanism="Много объёма на малый размах = глубокий рынок",
       measures="объём на единицу размаха за {W} баров относительно месяца",
       why="Прокси глубины стакана без стакана", limits="Объём не равен ликвидности стакана",
       fn=lambda x, W: np.log(safe_div(x.v.rolling(W).sum(), x.rng.rolling(W).sum())
                              / safe_div(x.v.rolling(8640, min_periods=2000).sum(),
                                         x.rng.rolling(8640, min_periods=2000).sum())))
family(CAT, CODE, "lq_trades_rel", None, formula="ln( trades_t / mean_288(trades_(t−1..t−288)) )",
       kind="непрерывный", unit="ln-отношение", deps=("trades",), cost="O(n)",
       mechanism="Всплеск числа сделок — приход участников", measures="число сделок в баре относительно суток",
       why="Активность независимо от размера сделок", limits="Дробление ордеров алгоритмами раздувает счёт",
       fn=lambda x: np.log(safe_div(x.col("trades"), x.col("trades").shift().rolling(288, min_periods=288).mean())))
family(CAT, CODE, "lq_trade_size_rel", None, formula="ln( (qv/trades)_t / mean_288(qv/trades) )",
       kind="непрерывный", unit="ln-отношение", deps=("quote_volume", "trades"), cost="O(n)",
       mechanism="Крупные средние сделки — присутствие крупных участников",
       measures="средний размер сделки в баре относительно суток",
       why="Отличает приход крупных от наплыва мелких", limits="Среднее чувствительно к единичным блокам",
       fn=lambda x: np.log(safe_div(safe_div(x.col("quote_volume"), x.col("trades")),
                                    safe_div(x.col("quote_volume"), x.col("trades")).shift()
                                    .rolling(288, min_periods=288).mean())))


def _roll_spread(x, W):
    dc = x.c.diff()
    cov = dc.rolling(W, min_periods=W).cov(dc.shift())
    return 2 * np.sqrt((-cov).clip(lower=0)) / x.a


family(CAT, CODE, "lq_roll_spread_{W}", grid(W=(24, 96, 288)), formula="2·sqrt(max(0, −cov_W(Δc_t, Δc_(t−1)))) / a",
       kind="непрерывный", unit="ATR", deps=("close", "atr"), cost="O(n)",
       mechanism="Отскок цены между bid и ask даёт отрицательную автоковариацию (Roll, 1984)",
       measures="оценка эффективного спреда за {W} баров в ATR",
       why="Высокий спред — дорогое исполнение и шумные сигналы",
       limits="На барах 5m+ оценка грубая; ноль при положительной автоковариации", fn=_roll_spread)


def _cs_spread(x):
    hl = np.log(x.h / x.l) ** 2
    beta = hl + hl.shift()
    gamma = np.log(np.maximum(x.h, x.h.shift()) / np.minimum(x.l, x.l.shift())) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)
    return s.rolling(24, min_periods=24).mean() / x.a_pct


family(CAT, CODE, "lq_cs_spread", None, formula="mean_24(Corwin–Schultz по паре баров) / (a/c)", kind="непрерывный",
       unit="ATR", deps=("high", "low", "atr"), cost="O(n)",
       mechanism="Размах пары баров содержит и волатильность, и спред (Corwin–Schultz, 2012)",
       measures="оценка спреда по максимумам и минимумам", why="Независимая от Roll оценка стоимости исполнения",
       limits="Отрицательные оценки обнуляются; смещена в периоды трендов", fn=_cs_spread)


def _ar_spread(x):
    mid = (np.log(x.h) + np.log(x.l)) / 2
    lc = np.log(x.c)
    s2 = 4 * (lc.shift() - mid.shift()) * (lc.shift() - mid)   # бар t−1 и середина бара t: всё известно в t
    return np.sqrt(s2.rolling(24, min_periods=24).mean().clip(lower=0)) / x.a_pct


family(CAT, CODE, "lq_ar_spread", None, formula="sqrt(max(0, mean_24 4(c_(t−1)−η_(t−1))(c_(t−1)−η_t))) / (a/c), η = (ln h + ln l)/2",
       kind="непрерывный", unit="ATR", deps=("high", "low", "close", "atr"), cost="O(n)",
       mechanism="Закрытие относительно середины соседних баров выдаёт спред (Abdi–Ranaldo, 2017)",
       measures="оценка спреда по закрытию и середине диапазона",
       why="Третья независимая оценка спреда", limits="Шумная на коротких окнах", fn=_ar_spread)
family(CAT, CODE, "lq_zero_ret_frac_{W}", grid(W=(24, 288)), formula="mean_W[r = 0]", kind="непрерывный",
       unit="доля [0,1]", deps=("close",), cost="O(n)", mechanism="Бары без изменения цены — неликвидность (Lesmond)",
       measures="доля баров без изменения закрытия за {W} баров", why="Простой индикатор «застоя» рынка",
       limits="На ликвидных парах почти всегда 0", fn=lambda x, W: (x.r == 0).astype(float).rolling(W, min_periods=W).mean())
