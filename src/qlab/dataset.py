"""Dataset Builder: единая таблица баров ТФ с честным временем доступности каждого значения.

Правило: строка бара с меткой `ts` (открытие) известна в `available_at` = ts + ТФ (закрытие).
Внешние значения (OI, funding) присоединяются as-of по `available_at`: в строку попадает только то,
что было опубликовано не позже закрытия бара. Своё время доступности каждого внешнего значения
хранится рядом (`oi_available_at`, `funding_available_at`).
"""
from __future__ import annotations

import pandas as pd

from qlab.config import TF_MINUTES

BAR_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "taker_buy_volume": "sum", "trades": "sum",
}


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1m → ТФ. Бар неполный, если в нём меньше минут, чем положено (пропуски в источнике)."""
    minutes = TF_MINUTES[tf]
    step = pd.Timedelta(minutes=minutes)
    g = m1.groupby(m1["ts"].dt.floor(step), sort=True)
    bars = g.agg(BAR_AGG)
    bars["n_minutes"] = g.size()
    bars = bars.reset_index()
    bars["available_at"] = bars["ts"] + step
    bars["complete"] = bars["n_minutes"] == minutes
    return bars


def _asof(bars: pd.DataFrame, ext: pd.DataFrame, value_cols: list[str], prefix: str,
          lag: pd.Timedelta, max_staleness: pd.Timedelta | None) -> pd.DataFrame:
    ext = ext[["ts", *value_cols]].copy()
    ext[f"{prefix}_available_at"] = ext["ts"] + lag
    ext = ext.rename(columns={"ts": f"{prefix}_ts"}).sort_values(f"{prefix}_available_at")
    out = pd.merge_asof(bars.sort_values("available_at"), ext, left_on="available_at",
                        right_on=f"{prefix}_available_at", direction="backward")
    if max_staleness is not None:
        stale = (out["available_at"] - out[f"{prefix}_available_at"]) > max_staleness
        out.loc[stale, value_cols] = float("nan")
    return out


def build(m1: pd.DataFrame, tf: str, *, oi: pd.DataFrame | None, funding: pd.DataFrame | None,
          btc_m1: pd.DataFrame | None, oi_lag_minutes: float, oi_max_staleness_minutes: float) -> pd.DataFrame:
    bars = resample(m1, tf)
    if oi is not None and not oi.empty:
        bars = _asof(bars, oi, ["oi"], "oi", pd.Timedelta(minutes=oi_lag_minutes),
                     pd.Timedelta(minutes=oi_max_staleness_minutes))
    else:
        bars["oi"] = float("nan")
    if funding is not None and not funding.empty:
        # ставка известна в момент расчёта (calc_time)
        bars = _asof(bars, funding, ["funding_rate"], "funding", pd.Timedelta(0), None)
    else:
        bars["funding_rate"] = float("nan")
    if btc_m1 is not None:
        btc = resample(btc_m1, tf)[["ts", "close", "complete"]]
        btc = btc.rename(columns={"close": "btc_close", "complete": "btc_complete"})
        bars = bars.merge(btc, on="ts", how="left")  # бар BTC с тем же ts известен в тот же момент
    bars = bars.sort_values("ts").reset_index(drop=True)
    bars.attrs["tf"] = tf
    bars.attrs["tf_minutes"] = TF_MINUTES[tf]
    return bars
