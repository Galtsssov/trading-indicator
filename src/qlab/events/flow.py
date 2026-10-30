"""События позиционирования: необычные изменения funding и всплески OI (этап 1: B1–B5)."""
import numpy as np
import pandas as pd

from qlab.events import bars_per_day, event, frame


@event("funding_spike", requires=("funding_rate", "funding_ts"), defaults={"z": 3.0, "window": 90},
       description="новая ставка funding отличается от прошлой на ≥ z стандартных отклонений изменений "
                   "за N расчётов; dir — знак изменения")
def funding_spike(bars: pd.DataFrame, p) -> pd.DataFrame:
    new = (bars["funding_ts"] != bars["funding_ts"].shift(1)) & bars["funding_rate"].notna()
    rates = bars["funding_rate"][new]
    change = rates.diff()
    sd = change.shift(1).rolling(p["window"], min_periods=p["window"] // 3).std()
    z = (change / sd).reindex(bars.index)
    return frame(bars, (z.abs() >= p["z"]).to_numpy(), np.sign(z))


@event("oi_spike", requires=("oi_change", "close"), defaults={"quantile": 0.99, "window_days": 30, "dir_bars": 3},
       description="|ΔOI| выше своего 99-го процентиля за 30 дней; dir — знак движения цены за те же бары")
def oi_spike(bars: pd.DataFrame, p) -> pd.DataFrame:
    x = bars["oi_change"].abs()
    w = p["window_days"] * bars_per_day(bars)
    thr = x.shift(1).rolling(w, min_periods=w // 4).quantile(p["quantile"])
    return frame(bars, x > thr, np.sign(bars["close"] - bars["close"].shift(p["dir_bars"])))
