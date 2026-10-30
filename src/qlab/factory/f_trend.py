"""Mean Reversion, Momentum, Acceleration, Curvature: отклонение от «нормы», импульс и его производные."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, safe_div

MR = ("Mean Reversion", "MR")
family(*MR, "mr_z_sma_{W}", grid(W=(6, 24, 96, 288)), formula="(c − SMA_W(c)) / std_W(c)", kind="непрерывный", unit="σ",
       deps=("close",), cost="O(n)", mechanism="Временное отклонение от средней возвращается (предоставление ликвидности)",
       measures="отклонение цены от средней {W} баров", why="Классическая мера перекупленности",
       limits="В тренде остаётся экстремальной долго",
       fn=lambda x, W: safe_div(x.c - x.c.rolling(W, min_periods=W).mean(), x.c.rolling(W, min_periods=W).std()))
family(*MR, "mr_dist_ema_{W}", grid(W=(12, 48, 200)), formula="(c − EMA_W(c)) / a", kind="непрерывный", unit="ATR",
       deps=("close", "atr"), cost="O(n)", mechanism="Отклонение от экспоненциальной средней", measures="удаление от EMA{W} в ATR",
       why="Отклонение в единицах волатильности", limits="EMA помнит бесконечно (с затуханием)",
       fn=lambda x, W: (x.c - x.c.ewm(span=W, adjust=False, min_periods=W).mean()) / x.a)


def _rsi(x, n):
    up = x.c.diff().clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-x.c.diff().clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 * safe_div(up, up + dn)            # = 100 − 100/(1 + up/dn), но корректно при dn = 0


family(*MR, "mr_rsi_{W}", grid(W=(6, 14, 48)), formula="RSI Уайлдера(W) = 100 · EMA(рост) / (EMA(рост) + EMA(падение))",
       kind="непрерывный", unit="0–100", deps=("close",), cost="O(n)", mechanism="Соотношение ростов и падений",
       measures="RSI с периодом {W}", why="Стандартная мера перекупленности", limits="Дублирует нормированную доходность", fn=lambda x, W: _rsi(x, W))
family(*MR, "mr_stoch_{W}", grid(W=(14, 96)), formula="(c − min_W l) / (max_W h − min_W l)", kind="непрерывный", unit="доля [0,1]",
       deps=("high", "low", "close"), cost="O(n)", mechanism="Положение в диапазоне (стохастик %K)",
       measures="стохастик за {W} баров", why="Нужен для проверки дублей с ps_pos_range", limits="То же, что ps_pos_range",
       fn=lambda x, W: safe_div(x.c - x.l.rolling(W).min(), x.h.rolling(W).max() - x.l.rolling(W).min()))
family(*MR, "mr_cci_20", None, formula="(tp − SMA_20 tp) / (0.015 · mean_20 |tp − SMA_20 tp|)", kind="непрерывный", unit="безразмерно",
       deps=("high", "low", "close"), cost="O(n·W)", mechanism="Отклонение типичной цены от средней", measures="CCI(20)",
       why="Устойчивая к выбросам нормировка", limits="Дублирует z-оценку",
       fn=lambda x: (lambda tp: safe_div(tp - tp.rolling(20).mean(), 0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean()))((x.h + x.l + x.c) / 3))


def _half_life(x, W=288):
    dev = x.c - x.c.rolling(W, min_periods=W).mean()
    b = safe_div(x.c.diff().rolling(W, min_periods=W).cov(dev.shift()), dev.shift().rolling(W, min_periods=W).var())
    return (-np.log(2) / np.log1p(b.where(b < 0))).clip(upper=10 * W) / W


family(*MR, "mr_half_life_288", None, formula="−ln 2 / ln(1 + b) / 288, b — наклон Δc на отклонение от SMA_288",
       kind="непрерывный", unit="доля окна", deps=("close",), cost="O(n)", mechanism="Скорость возврата к средней (процесс Орнштейна–Уленбека)",
       measures="период полураспада отклонения в долях суток", why="Быстрый возврат — флэт, нет возврата — тренд",
       limits="NaN, если возврата нет (b ≥ 0)", fn=_half_life)

MO = ("Momentum", "MO")
family(*MO, "mo_ret_{W}", grid(W=(1, 3, 6, 12, 24, 96, 288)), formula="ln(c / c_(t−W)) / (a / c_(t−1))", kind="непрерывный",
       unit="ATR", deps=("close", "atr"), cost="O(n)", mechanism="Импульс: краткосрочный разворот, долгосрочное продолжение",
       measures="доходность за {W} баров в ATR", why="Базовая мера движения в единицах волатильности", limits="—",
       fn=lambda x, W: x.ret(W) / x.a_pct)
family(*MO, "mo_up_frac_{W}", grid(W=(6, 24, 96)), formula="mean_W[r > 0]", kind="непрерывный", unit="доля [0,1]", deps=("close",),
       cost="O(n)", mechanism="Устойчивость направления", measures="доля растущих баров за {W}", why="Непараметрический моментум", limits="—",
       fn=lambda x, W: (x.r > 0).astype(float).rolling(W, min_periods=W).mean())
family(*MO, "mo_macd_hist", None, formula="(EMA12 − EMA26 − EMA9(EMA12 − EMA26)) / a", kind="непрерывный", unit="ATR",
       deps=("close", "atr"), cost="O(n)", mechanism="Ускорение средней", measures="гистограмма MACD в ATR", why="Стандартный импульс",
       limits="Запаздывает",
       fn=lambda x: (lambda m: (m - m.ewm(span=9, adjust=False).mean()) / x.a)(x.c.ewm(span=12, adjust=False).mean() - x.c.ewm(span=26, adjust=False).mean()))
family(*MO, "mo_tsmom_{D}d", grid(D=(1, 7)), formula="sign(ln(c / c_(t − D суток)))", kind="категориальный", unit="−1 / 0 / +1",
       deps=("close",), cost="O(n)", mechanism="Моментум временных рядов (Moskowitz–Ooi–Pedersen; в крипте — Liu–Tsyvinski)",
       measures="знак доходности за {D} сут.", why="Самый устойчивый из известных эффектов на дневных горизонтах", limits="Медленный",
       fn=lambda x, D: np.sign(x.ret(D * x.bpd)))
family(*MO, "mo_consistency_{W}", grid(W=(24, 96)), formula="|Σ_W sign r| / W", kind="непрерывный", unit="доля [0,1]", deps=("close",),
       cost="O(n)", mechanism="Однонаправленность пути", measures="насколько бары окна смотрят в одну сторону", why="Тренд против пилы",
       limits="—", fn=lambda x, W: np.sign(x.r).rolling(W, min_periods=W).sum().abs() / W)

AC = ("Acceleration", "AC")
family(*AC, "ac_ret_{W}", grid(W=(3, 6, 24)), formula="(ln(c/c_(t−W)) − ln(c_(t−W)/c_(t−2W))) / (a/c)", kind="непрерывный", unit="ATR",
       deps=("close", "atr"), cost="O(n)", mechanism="Ускорение или замедление импульса", measures="изменение доходности {W} баров к предыдущим {W}",
       why="Выдыхающийся импульс против разгоняющегося", limits="Вторая разность шумна",
       fn=lambda x, W: (x.ret(W) - x.ret(W).shift(W)) / x.a_pct)
family(*AC, "ac_ema_slope_chg", None, formula="(ΔEMA20_t − ΔEMA20_(t−6)) / a", kind="непрерывный", unit="ATR", deps=("close", "atr"),
       cost="O(n)", mechanism="Изменение наклона средней", measures="ускорение сглаженной цены", why="Менее шумно, чем разности цен",
       limits="Запаздывает", fn=lambda x: (lambda e: (e.diff() - e.diff().shift(6)) / x.a)(x.c.ewm(span=20, adjust=False).mean()))
family(*AC, "ac_vol", None, formula="ln(std_6(r)_t / std_6(r)_(t−6))", kind="непрерывный", unit="ln-отношение", deps=("close",),
       cost="O(n)", mechanism="Разгон волатильности", measures="изменение короткой волатильности", why="Начало расширения", limits="Шумно",
       fn=lambda x: np.log(safe_div(x.rv(6), x.rv(6).shift(6))))

CV = ("Curvature", "CV")


def _poly_coef(x, W, deg_idx):
    t = np.arange(W, dtype=float) - (W - 1)
    X = np.column_stack([np.ones(W), t, t ** 2])
    w = np.linalg.pinv(X)[deg_idx]                          # коэффициент = w · (последние W значений)
    y = np.log(x.c.to_numpy())
    out = np.full(len(y), np.nan)
    if len(y) >= W:
        out[W - 1:] = np.convolve(y, w[::-1], mode="valid")
    return pd.Series(out, index=x.b.index)


family(*CV, "cv_quad_{W}", grid(W=(12, 48)), formula="коэффициент при t² квадратичной регрессии ln c на последних W барах · W² / std_W(r)",
       kind="непрерывный", unit="σ", deps=("close",), cost="O(n·W)", mechanism="Выгнутый путь (разгон) против вогнутого (замедление)",
       measures="кривизна траектории цены за {W} баров", why="Форма движения, которой нет в доходности",
       limits="Чувствительна к последним барам", fn=lambda x, W: _poly_coef(x, W, 2) * W ** 2 / x.rv(W))
family(*CV, "cv_slope_{W}", grid(W=(12, 48)), formula="наклон той же квадратичной регрессии в текущей точке · W / std_W(r)",
       kind="непрерывный", unit="σ", deps=("close",), cost="O(n·W)", mechanism="Мгновенная скорость сглаженной траектории",
       measures="наклон траектории сейчас", why="Скорость без запаздывания средней", limits="Шумна на концах",
       fn=lambda x, W: _poly_coef(x, W, 1) * W / x.rv(W))
family(*CV, "cv_ema_second_diff", None, formula="(EMA20_t − 2·EMA20_(t−3) + EMA20_(t−6)) / a", kind="непрерывный", unit="ATR",
       deps=("close", "atr"), cost="O(n)", mechanism="Кривизна сглаженной цены", measures="вторая разность EMA20",
       why="Сглаженная кривизна", limits="Запаздывает",
       fn=lambda x: (lambda e: (e - 2 * e.shift(3) + e.shift(6)) / x.a)(x.c.ewm(span=20, adjust=False).mean()))
