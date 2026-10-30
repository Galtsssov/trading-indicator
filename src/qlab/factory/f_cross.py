"""Cross Asset (BTC / ETH / SOL Context) и Relative Strength.

Для каждой монеты-контекста X считаются одни и те же признаки. Если X — это сама монета, признак = NaN
(колонки одинаковы для всех монет, чтобы гипотезы и анализ работали единообразно).
"""
import numpy as np
import pandas as pd

from qlab.factory import SYMBOLS, family, grid, nan_series, safe_div

CAT = {"BTCUSDT": "BTC Context", "ETHUSDT": "ETH Context", "SOLUSDT": "SOL Context"}


def _x(x, sym, field="close"):
    o = x.other(sym)
    return None if o is None else o[field].astype(float)


def _xr(x, sym):
    c = _x(x, sym)
    return None if c is None else x.memo(("xr", sym), lambda: np.log(c).diff())


def _wrap(fn):
    def inner(x, X, **p):
        return nan_series(x) if x.other(X) is None else fn(x, X, **p)
    return inner


for X in SYMBOLS:
    xs = X.removesuffix("USDT").lower()
    cat, code = CAT[X], "C" + xs[0].upper()
    deps = ("close", f"close({X})")
    family(cat, code, f"cx_{xs}_ret_{{W}}", grid(W=(1, 6, 24, 288), X=(X,)), formula=f"ln(c_t / c_(t−W)) монеты {X}",
           kind="непрерывный", unit="доля", deps=deps, cost="O(n)", mechanism="Рыночный фактор: альты следуют за BTC, монеты — друг за другом",
           measures=f"доходность {X} за {{W}} баров", why="Контекст: движется ли рынок, а не только монета",
           limits="Для самой монеты не определён (NaN)",
           fn=_wrap(lambda x, X, W: np.log(_x(x, X) / _x(x, X).shift(W))))
    family(cat, code, f"cx_{xs}_spread_{{W}}", grid(W=(6, 24, 288), X=(X,)), formula=f"ln(c/c_(t−W)) − ln(cX/cX_(t−W)), X = {X}",
           kind="непрерывный", unit="доля", deps=deps, cost="O(n)", mechanism="Идиосинкратическое движение монеты против рынка",
           measures=f"доходность монеты минус доходность {X} за {{W}} баров", why="Отделяет собственное движение от общего",
           limits="Без учёта беты", fn=_wrap(lambda x, X, W: x.ret(W) - np.log(_x(x, X) / _x(x, X).shift(W))))
    family(cat, code, f"cx_{xs}_beta_{{W}}", grid(W=(96, 288), X=(X,)), formula=f"cov_W(r, r_X) / var_W(r_X), X = {X}",
           kind="непрерывный", unit="коэффициент", deps=deps, cost="O(n)", mechanism="Чувствительность монеты к рыночному фактору",
           measures=f"бета к {X} за {{W}} баров", why="Нужна для остатка доходности", limits="Нестабильна на коротких окнах",
           fn=_wrap(lambda x, X, W: safe_div(x.r.rolling(W, min_periods=W).cov(_xr(x, X)), _xr(x, X).rolling(W, min_periods=W).var())))
    family(cat, code, f"cx_{xs}_corr_{{W}}", grid(W=(96, 288), X=(X,)), formula=f"corr_W(r, r_X), X = {X}",
           kind="непрерывный", unit="[−1,1]", deps=deps, cost="O(n)", mechanism="Декорреляция = у монеты своя история",
           measures=f"корреляция с {X} за {{W}} баров", why="Режим: рыночный или собственный", limits="Шумна",
           fn=_wrap(lambda x, X, W: x.r.rolling(W, min_periods=W).corr(_xr(x, X))))
    family(cat, code, f"cx_{xs}_resid_3", [{"X": X}], formula=f"Σ_3 (r − β_288·r_X) / std_288(r), X = {X}",
           kind="непрерывный", unit="σ", deps=deps, cost="O(n)", mechanism="Отклонение монеты от рыночной модели (стат-арбитраж)",
           measures=f"остаток доходности за 3 бара после учёта беты к {X}", why="Чистое собственное движение",
           limits="Бета запаздывает",
           fn=_wrap(lambda x, X: (x.r - safe_div(x.r.rolling(288, min_periods=288).cov(_xr(x, X)),
                                               _xr(x, X).rolling(288, min_periods=288).var()) * _xr(x, X))
                    .rolling(3).sum() / x.rv(288)))
    family(cat, code, f"cx_{xs}_vol_ratio", [{"X": X}], formula=f"std_96(r) / std_96(r_X), X = {X}", kind="непрерывный",
           unit="отношение", deps=deps, cost="O(n)", mechanism="Монета волатильнее рынка обычного — своя история",
           measures=f"относительная волатильность к {X}", why="Нормирует волатильность на рынок", limits="—",
           fn=_wrap(lambda x, X: x.rv(96) / _xr(x, X).rolling(96, min_periods=96).std()))
    family(cat, code, f"cx_{xs}_rv_96", [{"X": X}], formula=f"std_96(r_X), X = {X}", kind="непрерывный", unit="доля",
           deps=(f"close({X})",), cost="O(n)", mechanism="Режим волатильности рынка", measures=f"волатильность {X}",
           why="Контекст риска рынка", limits="—", fn=_wrap(lambda x, X: _xr(x, X).rolling(96, min_periods=96).std()))
    family(cat, code, f"cx_{xs}_er_96", [{"X": X}], formula=f"|cX_t − cX_(t−96)| / Σ_96 |ΔcX|, X = {X}",
           kind="непрерывный", unit="доля [0,1]", deps=(f"close({X})",), cost="O(n)",
           mechanism="Трендовость рынка", measures=f"эффективность движения {X} за 96 баров",
           why="Тренд или флэт на рынке в целом", limits="—",
           fn=_wrap(lambda x, X: safe_div((_x(x, X) - _x(x, X).shift(96)).abs(), _x(x, X).diff().abs().rolling(96).sum())))
    family(cat, code, f"cx_{xs}_vol_rel", [{"X": X}], formula=f"ln( vX_t / mean_24(vX_(t−1..t−24)) ), X = {X}",
           kind="непрерывный", unit="ln-отношение", deps=(f"volume({X})",), cost="O(n)",
           mechanism="Всплеск участия на рынке-лидере", measures=f"всплеск объёма {X}", why="Рыночное событие против локального",
           limits="—", fn=_wrap(lambda x, X: np.log(safe_div(_x(x, X, "volume"), _x(x, X, "volume").shift().rolling(24).mean()))))


def _basket(x, W):
    rets = [x.ret(W)] + [np.log(_x(x, s) / _x(x, s).shift(W)) for s in SYMBOLS if x.other(s) is not None]
    return pd.concat(rets, axis=1)


family("Relative Strength", "RS", "rs_rank_{W}", grid(W=(6, 24, 288)),
       formula="ранг доходности монеты за W среди BTC, ETH, SOL, нормированный в [0,1]", kind="непрерывный",
       unit="доля [0,1]", deps=("close", "close(BTC,ETH,SOL)"), cost="O(n)",
       mechanism="Деньги перетекают в сильнейшую монету", measures="место монеты по доходности за {W} баров",
       why="Относительная сила без привязки к уровню рынка", limits="Всего 3 монеты — грубая шкала",
       fn=lambda x, W: (_basket(x, W).rank(axis=1, pct=True).iloc[:, 0] - 1 / 3) * 1.5)
family("Relative Strength", "RS", "rs_vs_basket_{W}", grid(W=(6, 24, 288)), formula="ln(c/c_(t−W)) − среднее по корзине BTC/ETH/SOL",
       kind="непрерывный", unit="доля", deps=("close", "close(BTC,ETH,SOL)"), cost="O(n)",
       mechanism="Собственная сила против рынка", measures="перевес доходности монеты над корзиной за {W} баров",
       why="Учитывает величину, а не только место", limits="Корзина из 3 монет",
       fn=lambda x, W: x.ret(W) - _basket(x, W).mean(axis=1))


def _ratio_trend(x):
    b = _x(x, "BTCUSDT")
    if b is None:
        return nan_series(x)
    lr = np.log(x.c / b)
    return (lr - lr.rolling(288, min_periods=288).mean()) / (x.rv(288) * np.sqrt(288))


family("Relative Strength", "RS", "rs_btc_ratio_trend", None, formula="(ln(c/c_BTC) − mean_288 ln(c/c_BTC)) / (std_288(r)·√288)",
       kind="непрерывный", unit="σ суточного движения", deps=("close", "close(BTC)"), cost="O(n)",
       mechanism="Тренд отношения монеты к BTC", measures="отклонение пары монета/BTC от средней за 288 баров",
       why="Ротация капитала между BTC и альтами", limits="Для BTC не определён", fn=_ratio_trend)
