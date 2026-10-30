"""Изменение открытого интереса за N баров по значениям, известным на момент закрытия бара (этап 1: B4, B5).

Колонка `oi` уже присоединена as-of с задержкой публикации (dataset.py), поэтому сдвиг по барам причинен.
"""
import pandas as pd

from qlab.features import Context, feature


@feature("oi", requires=("oi",), provides=("oi_change",), defaults={"window": 3},
         description="OI_t / OI_{t−N} − 1 (доля)")
def oi(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    n = int(ctx.params["window"])
    return pd.DataFrame({"oi_change": df["oi"] / df["oi"].shift(n) - 1})
