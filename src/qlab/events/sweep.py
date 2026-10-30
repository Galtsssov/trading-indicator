"""Liquidity Sweep: вынос за экстремум прошлых баров и закрытие обратно (этап 1: A1, A10–A12).

dir = +1 — сняли максимум (вынос вверх), −1 — сняли минимум. Если на одном баре сняты обе стороны —
событие пропускается (направление не определено).
Уровень — экстремум баров [t−lookback, t−1], не моложе min_age баров; вынос ≥ depth_atr·ATR_{t−1}.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from qlab.events import event, frame


def level_and_age(x: np.ndarray, lookback: int, use_max: bool) -> tuple[np.ndarray, np.ndarray]:
    """Для каждого t: экстремум x[t−lookback : t] и сколько баров назад он был (1 = прошлый бар)."""
    n = len(x)
    level = np.full(n, np.nan)
    age = np.full(n, -1, dtype=np.int64)
    if n <= lookback:
        return level, age
    win = sliding_window_view(x[:-1], lookback)
    pos = np.argmax(win, axis=1) if use_max else np.argmin(win, axis=1)
    level[lookback:] = win[np.arange(len(win)), pos]
    age[lookback:] = lookback - pos
    return level, age


@event("liquidity_sweep", requires=("high", "low", "close", "atr"),
       defaults={"lookback": 24, "min_age": 3, "depth_atr": 0.10},
       description="вынос за экстремум N баров ≥ depth·ATR и закрытие обратно")
def liquidity_sweep(bars: pd.DataFrame, p) -> pd.DataFrame:
    high, low, close = (bars[c].to_numpy(float) for c in ("high", "low", "close"))
    atr_prev = bars["atr"].shift(1).to_numpy(float)
    lvl_hi, age_hi = level_and_age(high, p["lookback"], True)
    lvl_lo, age_lo = level_and_age(low, p["lookback"], False)
    with np.errstate(invalid="ignore"):
        up = (age_hi >= p["min_age"]) & ((high - lvl_hi) / atr_prev >= p["depth_atr"]) & (close < lvl_hi)
        dn = (age_lo >= p["min_age"]) & ((lvl_lo - low) / atr_prev >= p["depth_atr"]) & (close > lvl_lo)
    direction = np.where(up & ~dn, 1, np.where(dn & ~up, -1, 0))
    level = np.where(direction > 0, lvl_hi, lvl_lo)
    return frame(bars, direction != 0, direction, level=level)
