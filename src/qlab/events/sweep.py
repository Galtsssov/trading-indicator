"""Liquidity Sweep High/Low без заглядывания в будущее (этап 1: A1, A10–A12).

Свип вверх на баре t:
- уровень L = максимум high за бары [t−lookback, t−1] (только прошлое; правые бары не нужны);
- этому максимуму не меньше `min_age` баров (это уровень, а не предыдущая свеча);
- high_t ≥ L + depth_atr · ATR_{t−1}   (вынос за уровень);
- close_t < L                           (закрытие обратно под уровнем).
Свип вниз — зеркально. Событие известно на закрытии бара t (`available_at`).
Направление реверсии: свип вверх → −1 (шорт), свип вниз → +1 (лонг).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class SweepParams:
    lookback: int = 24
    min_age: int = 3
    depth_atr: float = 0.10
    cooldown: int = 12


def _level_and_age(x: np.ndarray, lookback: int, use_max: bool) -> tuple[np.ndarray, np.ndarray]:
    """Для каждого t: экстремум x[t−lookback : t] и сколько баров назад он был (1 = прошлый бар)."""
    n = len(x)
    level = np.full(n, np.nan)
    age = np.full(n, -1, dtype=np.int64)
    if n <= lookback:
        return level, age
    win = sliding_window_view(x[:-1], lookback)          # окно i покрывает x[i : i+lookback] → для t = i+lookback
    pos = np.nanargmax(win, axis=1) if use_max else np.nanargmin(win, axis=1)
    level[lookback:] = win[np.arange(len(win)), pos]
    age[lookback:] = lookback - pos
    return level, age


def _cooldown(idx: np.ndarray, bars: int) -> np.ndarray:
    keep, last = [], -10**12
    for i in idx:
        if i - last > bars:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)


def detect(df: pd.DataFrame, p: SweepParams) -> pd.DataFrame:
    """События свипов. Требует колонки high, low, close, atr, complete, ts, available_at."""
    high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    atr_prev = df["atr"].shift(1).to_numpy(dtype=float)
    complete = df["complete"].to_numpy(dtype=bool)
    frames = []
    for side, use_max, direction in (("high", True, -1), ("low", False, 1)):
        level, age = _level_and_age(high if use_max else low, p.lookback, use_max)
        if use_max:
            depth = (high - level) / atr_prev
            back = close < level
        else:
            depth = (level - low) / atr_prev
            back = close > level
        hit = (age >= p.min_age) & (depth >= p.depth_atr) & back & complete & np.isfinite(atr_prev)
        idx = _cooldown(np.flatnonzero(hit), p.cooldown)
        frames.append(pd.DataFrame({
            "bar": idx,
            "side": side,
            "direction": direction,
            "level": level[idx],
            "extreme": (high if use_max else low)[idx],
            "depth_atr": depth[idx],
            "level_age": age[idx],
            "atr_ref": atr_prev[idx],
        }))
    ev = pd.concat(frames, ignore_index=True).sort_values("bar").reset_index(drop=True)
    for c in ("ts", "available_at"):
        ev[c] = df[c].iloc[ev["bar"].to_numpy()].reset_index(drop=True)
    return ev
