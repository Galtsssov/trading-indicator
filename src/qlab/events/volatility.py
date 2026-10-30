"""События волатильности (этап 1: D1–D3).

- compression:   процентиль ATR опустился ниже low (вход в сжатие); dir — знак движения за N баров;
- expansion:     процентиль ATR поднялся выше high (выход в расширение); dir — знак движения за N баров;
- atr_expansion: размах бара ≥ k·ATR_{t−1} (импульсный бар); dir — знак свечи.
"""
import numpy as np
import pandas as pd

from qlab.events import event, frame, sign_of_bar


def _trend_sign(bars: pd.DataFrame, n: int) -> np.ndarray:
    return np.sign(bars["close"] - bars["close"].shift(n)).to_numpy()


@event("compression", requires=("atr_pctl", "close"), defaults={"low": 0.10, "dir_bars": 12})
def compression(bars: pd.DataFrame, p) -> pd.DataFrame:
    x = bars["atr_pctl"]
    return frame(bars, (x < p["low"]) & (x.shift(1) >= p["low"]), _trend_sign(bars, p["dir_bars"]))


@event("expansion", requires=("atr_pctl", "close"), defaults={"high": 0.90, "dir_bars": 3})
def expansion(bars: pd.DataFrame, p) -> pd.DataFrame:
    x = bars["atr_pctl"]
    return frame(bars, (x > p["high"]) & (x.shift(1) <= p["high"]), _trend_sign(bars, p["dir_bars"]))


@event("atr_expansion", requires=("open", "high", "low", "close", "atr"), defaults={"k": 2.5})
def atr_expansion(bars: pd.DataFrame, p) -> pd.DataFrame:
    rng = (bars["high"] - bars["low"]) / bars["atr"].shift(1)
    return frame(bars, rng >= p["k"], sign_of_bar(bars))
