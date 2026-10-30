"""Regime и Fractal: тренд или флэт, персистентность, «шероховатость» пути цены."""
import numpy as np
import pandas as pd

from qlab.factory import WIN, family, grid, rolling_pctl, safe_div

CAT, CODE = "Regime", "RG"
CODES = {"regime_trend": ("range", "trend"), "regime_vol": ("low_vol", "high_vol"),
         "regime_btc": ("btc_weak", "btc_strong"), "regime_risk": ("risk_off", "risk_on")}
for col, (zero, one) in CODES.items():
    family(CAT, CODE, f"rg_{col.removeprefix('regime_')}", None, formula=f"1 = {one}, 0 = {zero} (см. features/regime.py)",
           kind="бинарный", unit="0/1", deps=(col,), cost="O(n)", mechanism="Свойства рынка меняются между режимами",
           measures=f"режим {col.removeprefix('regime_')}", why="Разбивка любых эффектов по режимам",
           limits="Пороги заданы заранее и грубые",
           fn=lambda x, col=col, one=one: pd.Series(np.where(x.b[col].isna(), np.nan, (x.b[col] == one).astype(float)),
                                                     index=x.b.index))


def _er(x, W):
    return x.memo(("er", W), lambda: safe_div((x.c - x.c.shift(W)).abs(), x.c.diff().abs().rolling(W, min_periods=W).sum()))


family(CAT, CODE, "rg_efficiency_{W}", grid(W=WIN), formula="|c_t − c_(t−W)| / Σ_W |Δc|", kind="непрерывный",
       unit="доля [0,1]", deps=("close",), cost="O(n)", mechanism="Прямой путь — тренд, извилистый — флэт (Kaufman)",
       measures="эффективность движения за {W} баров", why="Объективная мера трендовости", limits="Не знает направления",
       fn=lambda x, W: _er(x, W))
family(CAT, CODE, "rg_efficiency_pctl", None, formula="процентиль rg_efficiency_288 за 90 суток", kind="непрерывный",
       unit="доля [0,1]", deps=("close",), cost="O(n log W)", mechanism="Относительная трендовость", measures="трендовость суток относительно квартала",
       why="Сравнимо во времени", limits="Медленная", fn=lambda x: rolling_pctl(_er(x, 288), 90 * x.bpd))


def _vr(x, q, W=288):
    rq = np.log(x.c / x.c.shift(q))
    return safe_div(rq.rolling(W, min_periods=W).var(), q * x.r.rolling(W, min_periods=W).var())


family(CAT, CODE, "rg_variance_ratio_{q}", grid(q=(2, 4, 8)), formula="var_288(r_q) / (q · var_288(r)), r_q = ln(c_t/c_(t−q))",
       kind="непрерывный", unit="отношение", deps=("close",), cost="O(n)",
       mechanism="VR > 1 — моментум, VR < 1 — возврат к среднему (Lo–MacKinlay, 1988)",
       measures="variance ratio горизонта {q} баров за последние 288", why="Прямая мера персистентности",
       limits="Перекрывающиеся доходности смещают оценку", fn=lambda x, q: _vr(x, q))


def _tstat(x, W):
    t = pd.Series(np.arange(len(x.b), dtype=float), index=x.b.index)
    rho = t.rolling(W, min_periods=W).corr(np.log(x.c))
    return rho * np.sqrt(W - 2) / np.sqrt((1 - rho ** 2).clip(lower=1e-9))


family(CAT, CODE, "rg_trend_tstat_{W}", grid(W=(24, 96, 288)), formula="t-статистика наклона регрессии ln c на время за W",
       kind="непрерывный", unit="t", deps=("close",), cost="O(n)", mechanism="Статистически значимый тренд",
       measures="насколько линейный тренд за {W} баров отличим от шума", why="Тренд с учётом разброса",
       limits="Остатки автокоррелированы — t завышена", fn=_tstat)


def _adx(x, n=14):
    up, dn = x.h.diff(), -x.l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = np.maximum(x.rng, np.maximum((x.h - x.c.shift()).abs(), (x.l - x.c.shift()).abs()))
    sm = lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()  # noqa: E731
    pdi, mdi = sm(pdm) / sm(tr), sm(mdm) / sm(tr)
    return sm(safe_div((pdi - mdi).abs(), pdi + mdi))


family(CAT, CODE, "rg_adx_14", None, formula="ADX(14) Уайлдера: сглаженное |+DI − −DI| / (+DI + −DI)", kind="непрерывный",
       unit="доля [0,1]", deps=("high", "low", "close"), cost="O(n)", mechanism="Сила направленного движения",
       measures="сила тренда без направления", why="Классическая объективная мера", limits="Запаздывает", fn=_adx)

F, FC = "Fractal", "FR"


def _katz(x, W):
    L = x.c.diff().abs().rolling(W - 1, min_periods=W - 1).sum()
    d = x.c.rolling(W, min_periods=W).max() - x.c.rolling(W, min_periods=W).min()
    n = W - 1
    return np.log10(n) / (np.log10(n) + np.log10(safe_div(d, L)))


family(F, FC, "fr_katz_{W}", grid(W=(24, 96)), formula="log10(n) / (log10(n) + log10(d/L)), L — длина пути, d — размах закрытий, n = W−1",
       kind="непрерывный", unit="размерность 1–2", deps=("close",), cost="O(n)",
       mechanism="Чем извилистее путь, тем ближе размерность к 2 (Katz, 1988)", measures="фрактальная размерность пути за {W} баров",
       why="Шероховатость движения независимо от масштаба", limits="Оценка Katz смещена", fn=_katz)


def _swings(x, W):
    hi = (x.h.shift(2) > x.h.shift(3)) & (x.h.shift(2) > x.h.shift(4)) & (x.h.shift(2) > x.h.shift(1)) & (x.h.shift(2) > x.h)
    lo = (x.l.shift(2) < x.l.shift(3)) & (x.l.shift(2) < x.l.shift(4)) & (x.l.shift(2) < x.l.shift(1)) & (x.l.shift(2) < x.l)
    return (hi.astype(float) + lo.astype(float)).rolling(W, min_periods=W).sum() / W


family(F, FC, "fr_swing_density_{W}", grid(W=(96, 288)), formula="(число подтверждённых 2+2-баровых фракталов за W) / W",
       kind="непрерывный", unit="доля", deps=("high", "low"), cost="O(n)", mechanism="Частота локальных разворотов",
       measures="плотность локальных экстремумов за {W} баров (с задержкой подтверждения 2 бара)",
       why="Объективная «рваность» рынка", limits="Подтверждение запаздывает на 2 бара", fn=_swings)
family(F, FC, "fr_scaling_{S}_{L}", [{"S": 6, "L": 24}, {"S": 24, "L": 288}], formula="ln(R_L / R_S) / ln(L/S), R_W — средний размах окна W",
       kind="непрерывный", unit="показатель", deps=("high", "low"), cost="O(n)",
       mechanism="Рост размаха с масштабом: 0.5 — случайное блуждание, > 0.5 — тренд (аналог Хёрста)",
       measures="показатель масштабирования размаха между {S} и {L} барами", why="Персистентность без регрессий",
       limits="Грубая оценка",
       fn=lambda x, S, L: np.log(safe_div((x.hmax(L) - x.lmin(L)), (x.hmax(S) - x.lmin(S)).rolling(L, min_periods=L).mean())) / np.log(L / S))
