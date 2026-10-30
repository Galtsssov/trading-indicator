"""Outcome Engine: что произошло после события. О гипотезах ничего не знает.

Вход — открытие первой минуты после `available_at` события (сигнал известен только на закрытии бара).
1R = r_atr · ATR_ref события; стоп = 1R против направления. Путь цены — минутные свечи.

Для каждого события:
- mfe_r / mae_r — максимальное благоприятное / неблагоприятное движение за горизонт (в R);
- t_hit_{k}r, t_stop — минута первого касания +kR и −1R (NaN, если не было);
- res_{k}r — исход «цель kR / стоп 1R / выход по времени» в R до издержек;
  если цель и стоп в одной минуте — считается, что первым сработал стоп (консервативно);
- cost_r — издержки круга в R; res_{k}r_net = res_{k}r − cost_r.
События без полного непрерывного минутного пути до конца горизонта исключаются с причиной.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTE_NS = 60 * 10**9


@dataclass(frozen=True)
class OutcomeParams:
    r_atr: float = 1.0
    targets: tuple[float, ...] = (1.0, 2.0, 3.0)
    horizon_minutes: int = 120
    cost_pct_round_trip: float = 0.0014


def _first_hit(mask: np.ndarray) -> np.ndarray:
    """Индекс первой True по строке (минуты от входа, начиная с 0); NaN — не было."""
    first = mask.argmax(axis=1).astype(float)
    first[~mask.any(axis=1)] = np.nan
    return first


def _key(series: pd.Series) -> np.ndarray:
    return series.dt.as_unit("ns").astype("int64").to_numpy() // MINUTE_NS


def compute(events: pd.DataFrame, m1: pd.DataFrame, p: OutcomeParams) -> pd.DataFrame:
    """events: direction (+1/−1), atr_ref, available_at. m1: ts, open, high, low, close (отсортировано)."""
    out = events.copy()
    n, h = len(events), p.horizon_minutes
    out["exclude_reason"] = pd.Series([None] * n, dtype=object)
    cols = ["entry_price", "r_pct", "cost_r", "mfe_r", "mae_r", "t_stop"]
    for k in p.targets:
        cols += [f"t_hit_{k:g}r", f"res_{k:g}r", f"res_{k:g}r_net"]
    for c in cols:
        out[c] = np.nan
    if n == 0:
        return out

    mkey = _key(m1["ts"])
    pos = np.searchsorted(mkey, _key(events["available_at"]))
    in_range = pos + h <= len(mkey)
    ok = in_range.copy()
    ok[in_range] = mkey[pos[in_range]] == _key(events["available_at"])[in_range]
    out.loc[~ok, "exclude_reason"] = "нет минутных данных на входе или за горизонтом"
    cont = np.zeros(n, dtype=bool)
    cont[ok] = (mkey[pos[ok] + h - 1] - mkey[pos[ok]]) == h - 1
    out.loc[ok & ~cont, "exclude_reason"] = "разрыв минутных данных внутри горизонта"
    atr_ok = np.isfinite(events["atr_ref"].to_numpy()) & (events["atr_ref"].to_numpy() > 0)
    out.loc[cont & ~atr_ok, "exclude_reason"] = "нет ATR"
    good = np.flatnonzero(cont & atr_ok)
    if len(good) == 0:
        return out

    o, hi, lo, cl = (m1[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    atr_ref = events["atr_ref"].to_numpy(dtype=float)
    direction = events["direction"].to_numpy().astype(float)
    for start in range(0, len(good), CHUNK):
        g = good[start:start + CHUNK]
        _fill(out, g, pos[g], o, hi, lo, cl, p.r_atr * atr_ref[g], direction[g], p)
    return out


CHUNK = 4000  # событий за раз: ограничивает память матрицы пути (CHUNK × horizon)


def _fill(out: pd.DataFrame, g: np.ndarray, start: np.ndarray, o: np.ndarray, hi: np.ndarray,
          lo: np.ndarray, cl: np.ndarray, r_abs: np.ndarray, d: np.ndarray, p: OutcomeParams) -> None:
    idx = start[:, None] + np.arange(p.horizon_minutes)[None, :]
    entry = o[start]
    long = (d > 0)[:, None]
    fav = np.where(long, hi[idx] - entry[:, None], entry[:, None] - lo[idx]) / r_abs[:, None]
    adv = np.where(long, entry[:, None] - lo[idx], hi[idx] - entry[:, None]) / r_abs[:, None]
    last = d * (cl[idx[:, -1]] - entry) / r_abs

    t_stop = _first_hit(adv >= 1.0)
    r_pct = r_abs / entry
    cost_r = p.cost_pct_round_trip / r_pct
    rows = out.index[g]
    out.loc[rows, "entry_price"] = entry
    out.loc[rows, "r_pct"] = r_pct
    out.loc[rows, "cost_r"] = cost_r
    out.loc[rows, "mfe_r"] = fav.max(axis=1)
    out.loc[rows, "mae_r"] = adv.max(axis=1)
    out.loc[rows, "t_stop"] = t_stop
    stop_t = np.where(np.isnan(t_stop), np.inf, t_stop)
    for k in p.targets:
        t_hit = _first_hit(fav >= k)
        hit_t = np.where(np.isnan(t_hit), np.inf, t_hit)
        res = np.where(hit_t < stop_t, k, np.where(np.isfinite(stop_t), -1.0, np.clip(last, -1.0, k)))
        out.loc[rows, f"t_hit_{k:g}r"] = t_hit
        out.loc[rows, f"res_{k:g}r"] = res
        out.loc[rows, f"res_{k:g}r_net"] = res - cost_r
