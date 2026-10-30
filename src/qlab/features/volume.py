"""Всплеск объёма: объём бара к среднему за предыдущие N баров (этап 1: H1, H2)."""
import pandas as pd

from qlab.features import Context, feature


@feature("volume", requires=("volume",), provides=("vol_spike",), defaults={"window": 48},
         description="volume / mean(volume за N предыдущих баров)")
def volume(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    n = int(ctx.params["window"])
    base = df["volume"].shift(1).rolling(n, min_periods=n).mean()
    return pd.DataFrame({"vol_spike": df["volume"] / base})
