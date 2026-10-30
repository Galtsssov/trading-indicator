"""Baseline Library: простые модели, которые гипотеза обязана обыграть.

- random_entry     — та же сделка в случайный момент того же часа и той же децили волатильности;
- mean_reversion   — против движения ≥ k ATR за N баров;
- momentum         — по движению ≥ k ATR за N баров;
- buy_every_sweep  — лонг после каждого свипа (любой стороны);
- sell_every_sweep — шорт после каждого свипа.
Все эталоны используют ту же модель исхода и тот же период, что и гипотеза.
"""
from __future__ import annotations

import pandas as pd

from qlab import controls, events

NAMES = ("random_entry", "mean_reversion", "momentum", "buy_every_sweep", "sell_every_sweep")


def build(name: str, bars: pd.DataFrame, trades: pd.DataFrame, params: dict, seed: int,
          pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Сделки эталона: bar, ts, available_at, atr_ref, direction.
    bars — вся история (индекс 0..n−1); pool — бары периода, из которых выбирается случайный вход."""
    if name == "random_entry":
        return controls.matched(bars if pool is None else pool, trades, seed)
    if name in ("mean_reversion", "momentum"):
        ev = events.detect("return_extreme", bars, {"lookback": params["return_lookback"],
                                                    "threshold_atr": params["return_threshold_atr"]})
        return ev.assign(direction=ev["dir"] * (-1 if name == "mean_reversion" else 1))
    if name in ("buy_every_sweep", "sell_every_sweep"):
        ev = events.detect("liquidity_sweep", bars)
        return ev.assign(direction=1 if name == "buy_every_sweep" else -1)
    raise ValueError(f"неизвестный эталон {name!r}; есть {NAMES}")
