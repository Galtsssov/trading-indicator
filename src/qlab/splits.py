"""Разбивка истории: training / validation / holdout.

Holdout закрыт: `window("holdout")` без `allow_holdout=True` бросает HoldoutLocked.
Разрешение выдаёт только реестр (один раз на гипотезу, после пройденной валидации).
"""
from __future__ import annotations

import pandas as pd

from qlab.config import Config

SPLITS = ("training", "validation", "holdout")


class HoldoutLocked(PermissionError):
    pass


def window(cfg: Config, name: str, *, allow_holdout: bool = False,
           clean_after: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if name not in SPLITS:
        raise ValueError(f"неизвестный период {name!r}; допустимы {SPLITS}")
    if name == "holdout" and not allow_holdout:
        raise HoldoutLocked("holdout закрыт: только `qlab test <id> --split holdout --confirm-holdout`")
    start, end = (pd.Timestamp(x, tz="UTC") for x in cfg.raw["splits"][name])
    if clean_after:
        start = max(start, pd.Timestamp(clean_after, tz="UTC"))
    return start, end


def select(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
           horizon: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    """Строки, чьё событие и весь горизонт исхода лежат внутри [start, end) — без перетекания в следующий период."""
    return df[(df["ts"] >= start) & (df["available_at"] + horizon <= end)]
