"""Общий детектор аномалий: значение признака выходит за свой скользящий процентиль.

Отвечает на вопрос «что статистически чаще происходит после экстремальных значений признака?».
side: high — выше q-процентиля; low — ниже (1−q); abs — |x| выше q-процентиля |x|.
dir_mode: sign — знак самого признака; bar — знак свечи; up / down — фиксированное направление.
Порог считается по прошлым значениям (окно window_days, без текущего бара).
"""
import numpy as np
import pandas as pd

from qlab.events import bars_per_day, event, frame, sign_of_bar


@event("anomaly", requires=("close", "open"),
       defaults={"feature": "vol_spike", "side": "high", "quantile": 0.99, "window_days": 30, "dir_mode": "bar"})
def anomaly(bars: pd.DataFrame, p) -> pd.DataFrame:
    x = bars[p["feature"]].astype(float)
    w = int(p["window_days"] * bars_per_day(bars))
    q = p["quantile"]
    if p["side"] == "abs":
        x = x.abs()
    hist = x.shift(1).rolling(w, min_periods=w // 4)
    mask = x < hist.quantile(1 - q) if p["side"] == "low" else x > hist.quantile(q)
    raw = bars[p["feature"]].astype(float)
    direction = {"sign": np.sign(raw).to_numpy(), "bar": sign_of_bar(bars),
                 "up": np.ones(len(bars)), "down": -np.ones(len(bars))}[p["dir_mode"]]
    return frame(bars, mask.to_numpy(), direction, value=raw.to_numpy())
