"""Сессия по часу UTC (этап 1: E1–E6). Без учёта перехода на летнее время — ограничение MVP."""
import numpy as np
import pandas as pd

from qlab.features import Context, feature

SESSIONS = ((0, "asia"), (7, "europe"), (13, "us"), (20, "late"))


@feature("session", requires=("ts",), provides=("hour", "session", "weekend"),
         description="час UTC; asia 00–07, europe 07–13, us 13–20, late 20–24; выходные")
def session(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    hour = df["ts"].dt.hour
    starts = np.array([s for s, _ in SESSIONS])
    names = np.array([n for _, n in SESSIONS])
    return pd.DataFrame({
        "hour": hour,
        "session": names[np.searchsorted(starts, hour.to_numpy(), side="right") - 1],
        "weekend": df["ts"].dt.dayofweek >= 5,
    })
