"""VWAP суток UTC и расстояние до него в ATR (этап 1: G12)."""
import pandas as pd

from qlab.features import Context, feature


@feature("vwap", requires=("ts", "high", "low", "close", "volume", "atr"), provides=("vwap", "vwap_dist_atr"),
         description="VWAP с обнулением в 00:00 UTC; (close − VWAP) / ATR")
def vwap(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    day = df["ts"].dt.floor("D")
    tp_vol = (df["high"] + df["low"] + df["close"]) / 3 * df["volume"]
    value = tp_vol.groupby(day).cumsum() / df["volume"].groupby(day).cumsum()
    return pd.DataFrame({"vwap": value, "vwap_dist_atr": (df["close"] - value) / df["atr"]})
