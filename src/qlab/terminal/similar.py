"""Похожие ситуации: статистика того, как развивалось движение после похожих событий в истории.

Это статистика, а не сигнал. База строится один раз из архива (`qlab-terminal history`):
для каждого события сохраняются контекст в момент события и путь цены на H баров вперёд
(в ATR, со знаком направления события: > 0 — движение продолжилось в сторону события).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from qlab import events

log = logging.getLogger(__name__)
EVENT_TYPES = ("liquidity_sweep", "breakout", "atr_expansion", "vwap_touch", "range_break")
EVENT_NAMES = {"liquidity_sweep": "Снятие ликвидности", "breakout": "Пробой диапазона", "atr_expansion": "Импульсный бар",
               "vwap_touch": "Касание VWAP", "range_break": "Выход из азиатского диапазона"}
HORIZON = 24
CONTEXT = ("atr_pctl", "vwap_dist_atr", "vol_spike", "btc_ret", "hour_sin", "hour_cos", "trend_code", "vol_code", "oi_change")


def context(bars: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Контекст события на закрытии его бара (все значения известны в этот момент)."""
    b = bars.iloc[rows]
    hour = b["ts"].dt.hour + b["ts"].dt.minute / 60
    return pd.DataFrame({
        "atr_pctl": b["atr_pctl"].to_numpy(), "vwap_dist_atr": b["vwap_dist_atr"].to_numpy(),
        "vol_spike": np.log(b["vol_spike"].clip(lower=1e-6)).to_numpy(), "btc_ret": b["btc_ret"].to_numpy(),
        "hour_sin": np.sin(2 * np.pi * hour / 24).to_numpy(), "hour_cos": np.cos(2 * np.pi * hour / 24).to_numpy(),
        "trend_code": (b["regime_trend"] == "trend").astype(float).where(b["regime_trend"].notna()).to_numpy(),
        "vol_code": (b["regime_vol"] == "high_vol").astype(float).where(b["regime_vol"].notna()).to_numpy(),
        "oi_change": b["oi_change"].to_numpy(),
    })


def forward(bars: pd.DataFrame, rows: np.ndarray, direction: np.ndarray, atr: np.ndarray, h: int = HORIZON) -> pd.DataFrame:
    """Путь после события: закрытия, лучшее (MFE) и худшее (MAE) движение, достижение порогов."""
    c, hi, lo, vw = (bars[k].to_numpy(float) for k in ("close", "high", "low", "vwap"))
    n = len(bars)
    ok = rows + h < n
    rows, direction, atr = rows[ok], direction[ok], atr[ok]
    idx = rows[:, None] + np.arange(1, h + 1)[None, :]
    entry = c[rows][:, None]
    d = direction[:, None].astype(float)
    a = atr[:, None]
    path = d * (c[idx] - entry) / a
    fav = np.where(d > 0, hi[idx] - entry, entry - lo[idx]) / a
    adv = np.where(d > 0, entry - lo[idx], hi[idx] - entry) / a
    cum_fav, cum_adv = np.maximum.accumulate(fav, axis=1), np.maximum.accumulate(adv, axis=1)

    def first(mask):
        f = mask.argmax(axis=1).astype(float)
        f[~mask.any(axis=1)] = np.inf
        return f

    t_adv1 = first(cum_adv >= 1)
    ext = np.where(direction > 0, hi[rows], lo[rows])[:, None]
    out = pd.DataFrame(path.astype(np.float32), columns=[f"p{k}" for k in range(1, h + 1)])
    out["row"] = rows
    out["mfe"] = cum_fav[:, -1]
    out["mae"] = cum_adv[:, -1]
    out["t_mfe"] = fav.argmax(axis=1) + 1
    out["hit1"] = first(cum_fav >= 1) < t_adv1          # +1 ATR раньше −1 ATR (в одном баре — считаем откат первым)
    out["hit2"] = first(cum_fav >= 2) < t_adv1
    side0 = np.sign(c[rows] - vw[rows])[:, None]          # по какую сторону VWAP был вход
    out["touch_vwap"] = (((lo[idx] <= vw[idx]) & (hi[idx] >= vw[idx])) | (np.sign(c[idx] - vw[idx]) != side0)).any(axis=1)
    out["new_extreme"] = np.where(d > 0, hi[idx] > ext, lo[idx] < ext).any(axis=1)
    return out


def build_history(bars: pd.DataFrame, symbol: str, tf: str, h: int = HORIZON) -> pd.DataFrame:
    parts = []
    for et in EVENT_TYPES:
        ev = events.detect(et, bars)
        ev = ev[np.isfinite(ev["atr_ref"]) & (ev["atr_ref"] > 0)]
        if ev.empty:
            continue
        rows = ev["bar"].to_numpy()
        fw = forward(bars, rows, ev["dir"].to_numpy(), ev["atr_ref"].to_numpy(), h)
        keep = ev.set_index("bar").loc[fw["row"]]
        ctx = context(bars, fw["row"].to_numpy())
        parts.append(pd.concat([pd.DataFrame({"type": et, "dir": keep["dir"].to_numpy(), "ts": keep["ts"].to_numpy()}),
                                ctx, fw.drop(columns="row")], axis=1))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "symbol", symbol)
    out.insert(1, "tf", tf)
    return out


class History:
    """База событий одной пары монета × ТФ."""

    def __init__(self, folder: Path, symbol: str, tf: str):
        p = folder / f"{symbol}_{tf}.parquet"
        self.df = pd.read_parquet(p) if p.exists() else None
        self.path = p

    @property
    def available(self) -> bool:
        return self.df is not None and len(self.df) > 0

    def query(self, etype: str, direction: int, ctx: dict, before: pd.Timestamp, k: int = 150, tf_minutes: int = 5) -> dict:
        if not self.available:
            return {"available": False}
        h = self.df
        pool = h[(h["type"] == etype) & (h["dir"] == direction) & (h["ts"] < before - pd.Timedelta(minutes=HORIZON * tf_minutes))]
        if len(pool) < 30:
            return {"available": True, "n": len(pool), "too_few": True}
        X = pool[list(CONTEXT)].to_numpy(float)
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), med, X)
        sd = X.std(axis=0)
        sd[sd == 0] = 1
        q = np.array([ctx.get(c, np.nan) for c in CONTEXT], dtype=float)
        q = np.where(np.isnan(q), med, q)
        dist = np.sqrt((((X - q) / sd) ** 2).sum(axis=1))
        near = pool.iloc[np.argsort(dist)[:k]]
        return {"available": True, "n_pool": len(pool), "similar": summarize(near, tf_minutes),
                "all": summarize(pool, tf_minutes), "type": etype, "dir": direction}


def summarize(g: pd.DataFrame, tf_minutes: int) -> dict:
    P = g[[f"p{k}" for k in range(1, HORIZON + 1)]].to_numpy(float)
    last = P[:, -1]
    return {
        "n": int(len(g)),
        "continued": float((last > 0).mean()),
        "mean_path": np.round(P.mean(0), 3).tolist(),
        "q25_path": np.round(np.percentile(P, 25, axis=0), 3).tolist(),
        "q75_path": np.round(np.percentile(P, 75, axis=0), 3).tolist(),
        "mfe_median": float(g["mfe"].median()), "mfe_p90": float(g["mfe"].quantile(0.9)),
        "mae_median": float(g["mae"].median()), "mae_p90": float(g["mae"].quantile(0.9)),
        "t_mfe_minutes": float(g["t_mfe"].median() * tf_minutes),
        "p_hit1": float(g["hit1"].mean()), "p_hit2": float(g["hit2"].mean()),
        "p_touch_vwap": float(g["touch_vwap"].mean()), "p_new_extreme": float(g["new_extreme"].mean()),
        "horizon_minutes": HORIZON * tf_minutes,
    }
