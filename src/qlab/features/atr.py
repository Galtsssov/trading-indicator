"""ATR (Уайлдер) и его процентиль за скользящее окно — режим волатильности (этап 1: D1)."""
import numpy as np
import pandas as pd

from qlab.features import Context, feature


@feature("atr", requires=("high", "low", "close"), provides=("atr", "atr_pct", "atr_pctl"),
         defaults={"period": 14, "pctl_window_days": 30},
         description="ATR в цене и в % от close; процентиль ATR за N дней (0..1)")
def atr(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    n = int(ctx.params["period"])
    value = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    window = int(ctx.params["pctl_window_days"]) * ctx.bars_per_day
    return pd.DataFrame({
        "atr": value,
        "atr_pct": value / df["close"],
        "atr_pctl": value.rolling(window, min_periods=window // 4).rank(pct=True),
    })
