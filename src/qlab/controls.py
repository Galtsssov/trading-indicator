"""Контрольная выборка: случайные бары с тем же часом UTC и той же децилью процентиля ATR, что у события.

Без контроля «эффект» события легко оказывается сезонностью волатильности.
Направление сделки копируется у события, поэтому контроль отвечает на вопрос:
«та же сделка в похожий момент без свипа дала бы что?»
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _key(hour: pd.Series, atr_pctl: pd.Series) -> pd.Series:
    decile = np.minimum((atr_pctl.to_numpy() * 10).astype("float"), 9)
    return hour.astype("int64") * 10 + pd.Series(np.floor(decile), index=hour.index)


def matched(bars: pd.DataFrame, events: pd.DataFrame, seed: int) -> pd.DataFrame:
    """По одному контрольному бару на событие (с возвращением). Требует hour, atr_pctl, atr, complete."""
    rng = np.random.default_rng(seed)
    pool = bars.assign(atr_ref=bars["atr"].shift(1))
    pool = pool[pool["complete"] & pool["atr_pctl"].notna() & pool["atr_ref"].notna()]
    pool = pool.drop(index=events["bar"], errors="ignore")
    pool_key = _key(pool["hour"], pool["atr_pctl"])
    groups = {k: idx.to_numpy() for k, idx in pool.groupby(pool_key).groups.items()}

    ev = events[events["atr_pctl"].notna()]
    ev_key = _key(ev["hour"], ev["atr_pctl"])
    chosen, src = [], []
    for i, k in zip(ev.index, ev_key):
        cand = groups.get(k)
        if cand is not None and len(cand):
            chosen.append(cand[rng.integers(len(cand))])
            src.append(i)
    ctrl = pool.loc[chosen, ["ts", "available_at", "atr_ref"]].reset_index(names="bar")
    ctrl["direction"] = events.loc[src, "direction"].to_numpy()
    return ctrl
