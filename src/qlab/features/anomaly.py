"""Признаки формы бара и расхождений — сырьё для поиска аномалий (этап 1: I1, G1, G2, C1, J15).

Всё в единицах ATR предыдущего бара, чтобы значения были сравнимы между монетами и периодами.
"""
import numpy as np
import pandas as pd

from qlab.features import Context, feature


@feature("anomaly", requires=("open", "high", "low", "close", "atr", "volume", "taker_buy_volume", "btc_close"),
         provides=("wick_up_atr", "wick_dn_atr", "body_atr", "range_atr", "taker_imb", "btc_gap"),
         defaults={"gap_window": 3},
         description="тени, тело и размах бара в ATR; дисбаланс агрессоров; расхождение с BTC в ATR")
def anomaly(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    atr_prev = df["atr"].shift(1)
    top = np.maximum(df["open"], df["close"])
    bottom = np.minimum(df["open"], df["close"])
    n = int(ctx.params["gap_window"])
    gap = np.log(df["close"] / df["close"].shift(n)) - np.log(df["btc_close"] / df["btc_close"].shift(n))
    return pd.DataFrame({
        "wick_up_atr": (df["high"] - top) / atr_prev,
        "wick_dn_atr": (bottom - df["low"]) / atr_prev,
        "body_atr": (df["close"] - df["open"]) / atr_prev,
        "range_atr": (df["high"] - df["low"]) / atr_prev,
        "taker_imb": df["taker_buy_volume"] / df["volume"].replace(0, np.nan) - 0.5,
        "btc_gap": gap / (atr_prev / df["close"]),
    })
