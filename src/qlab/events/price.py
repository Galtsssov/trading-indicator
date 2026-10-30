"""Ценовые события: касание VWAP после удаления, экстремальная доходность за N баров."""
import numpy as np
import pandas as pd

from qlab.events import event, frame


@event("vwap_touch", requires=("high", "low", "vwap", "vwap_dist_atr"),
       defaults={"min_dist_atr": 1.5, "lookback": 12},
       description="бар коснулся VWAP, а за последние N баров цена отходила от него ≥ k ATR; "
                   "dir — направление подхода к VWAP")
def vwap_touch(bars: pd.DataFrame, p) -> pd.DataFrame:
    prev = bars["vwap_dist_atr"].shift(1)
    far = prev.abs().rolling(p["lookback"]).max() >= p["min_dist_atr"]
    touch = (bars["low"] <= bars["vwap"]) & (bars["high"] >= bars["vwap"])
    return frame(bars, touch & far, -np.sign(prev))


@event("return_extreme", requires=("close", "atr"), defaults={"lookback": 3, "threshold_atr": 2.0},
       description="|close − close[−N]| ≥ k ATR; dir — знак движения")
def return_extreme(bars: pd.DataFrame, p) -> pd.DataFrame:
    move = (bars["close"] - bars["close"].shift(p["lookback"])) / bars["atr"].shift(1)
    return frame(bars, move.abs() >= p["threshold_atr"], np.sign(move))
