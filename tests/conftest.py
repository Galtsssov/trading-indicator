"""Синтетические данные ТОЛЬКО для тестов кода. Результаты исследования считаются лишь на реальных данных."""
import numpy as np
import pandas as pd
import pytest


def random_walk_m1(n: int, seed: int, start: str = "2024-01-01", price: float = 100.0,
                   vol: float = 0.0008) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, (2, n))) * close
    volume = rng.lognormal(3, 0.5, n)
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq="1min", tz="UTC").as_unit("ns"),
        "open": open_, "high": np.maximum(open_, close) + wick[0], "low": np.minimum(open_, close) - wick[1],
        "close": close, "volume": volume, "quote_volume": volume * close,
        "taker_buy_volume": volume * rng.uniform(0.3, 0.7, n), "trades": rng.integers(10, 100, n),
    })


def random_oi(m1: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = m1["ts"][m1["ts"].dt.minute % 5 == 0].reset_index(drop=True)
    return pd.DataFrame({"ts": ts, "oi": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))})


def random_funding(m1: pd.DataFrame) -> pd.DataFrame:
    ts = m1["ts"][(m1["ts"].dt.hour % 8 == 0) & (m1["ts"].dt.minute == 0)].reset_index(drop=True)
    return pd.DataFrame({"ts": ts, "funding_rate": 0.0001})


@pytest.fixture
def m1():
    return random_walk_m1(20 * 1440, seed=1)
