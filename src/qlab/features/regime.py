"""Режимы рынка (архитектура, модуль 7). Каждый режим — категория, известная на закрытии бара.

- regime_trend: trend / range — efficiency ratio за сутки выше/ниже своей медианы за 90 дней;
- regime_vol:   high_vol / low_vol — процентиль ATR за 30 дней ≥ / < 0.5;
- regime_btc:   btc_strong / btc_weak — доходность BTC за сутки > / ≤ 0;
- regime_risk:  risk_on / risk_off — BTC выше / ниже своей средней за 20 дней.
Пороги заданы заранее и не подбираются.
"""
import numpy as np
import pandas as pd

from qlab.features import Context, feature


@feature("regime", requires=("close", "atr_pctl", "btc_close"),
         provides=("regime_trend", "regime_vol", "regime_btc", "regime_risk"),
         defaults={"er_window_days": 1, "er_pctl_days": 90, "btc_window_days": 1, "risk_ma_days": 20})
def regime(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    bpd, p = ctx.bars_per_day, ctx.params
    n = int(p["er_window_days"] * bpd)
    er = (df["close"] - df["close"].shift(n)).abs() / df["close"].diff().abs().rolling(n).sum()
    w = int(p["er_pctl_days"] * bpd)
    er_pctl = er.rolling(w, min_periods=w // 4).rank(pct=True)
    btc_ret = df["btc_close"] / df["btc_close"].shift(int(p["btc_window_days"] * bpd)) - 1
    m = int(p["risk_ma_days"] * bpd)
    btc_ma = df["btc_close"].rolling(m, min_periods=m).mean()

    def label(cond: pd.Series, known: pd.Series, yes: str, no: str) -> pd.Series:
        return pd.Series(np.where(known, np.where(cond, yes, no), None), index=df.index, dtype=object)

    return pd.DataFrame({
        "regime_trend": label(er_pctl >= 0.5, er_pctl.notna(), "trend", "range"),
        "regime_vol": label(df["atr_pctl"] >= 0.5, df["atr_pctl"].notna(), "high_vol", "low_vol"),
        "regime_btc": label(btc_ret > 0, btc_ret.notna(), "btc_strong", "btc_weak"),
        "regime_risk": label(df["btc_close"] > btc_ma, btc_ma.notna(), "risk_on", "risk_off"),
    })
