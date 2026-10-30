"""Контекст BTC: доходность и направление BTC за N баров, закончившихся на текущем баре (этап 1: J1)."""
import numpy as np
import pandas as pd

from qlab.features import Context, feature


@feature("btc", requires=("btc_close",), provides=("btc_ret", "btc_dir"), defaults={"window": 3},
         description="лог-доходность BTC за N баров и её знак (−1/0/+1)")
def btc(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    n = int(ctx.params["window"])
    ret = np.log(df["btc_close"] / df["btc_close"].shift(n))
    return pd.DataFrame({"btc_ret": ret, "btc_dir": np.sign(ret)})
