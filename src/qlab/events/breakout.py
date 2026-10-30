"""Breakout: закрытие за экстремумом N прошлых баров (в отличие от свипа — цена осталась за уровнем)."""
import numpy as np
import pandas as pd

from qlab.events import event, frame


@event("breakout", requires=("high", "low", "close"), defaults={"lookback": 24})
def breakout(bars: pd.DataFrame, p) -> pd.DataFrame:
    n = p["lookback"]
    hi = bars["high"].shift(1).rolling(n).max()
    lo = bars["low"].shift(1).rolling(n).min()
    up, dn = bars["close"] > hi, bars["close"] < lo
    return frame(bars, up | dn, np.where(up, 1, -1))
