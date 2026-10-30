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


def write_clean(data_dir, symbols, days: int, start: str = "2024-01-01") -> None:
    """Синтетические Parquet в структуре data/clean — для сквозных тестов кода."""
    for i, s in enumerate(symbols):
        m = random_walk_m1(days * 1440, seed=100 + i, start=start)
        for ds, df in (("klines_1m", m), ("metrics", random_oi(m, 200 + i)), ("funding", random_funding(m))):
            p = data_dir / "clean" / ds
            p.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p / f"{s}.parquet")


def make_config(tmp_path):
    """Конфиг платформы с короткими периодами под синтетику и маленьким числом повторов."""
    from qlab.config import Config
    cfg = Config.load("configs/platform.toml")
    cfg.raw["data"]["dir"] = str(tmp_path / "data")
    cfg.raw["splits"] = {"training": ["2024-01-01", "2024-02-01"], "validation": ["2024-02-01", "2024-03-01"],
                         "holdout": ["2024-03-01", "2100-01-01"]}
    cfg.raw["stats"].update(bootstrap=100, permutations=100)
    cfg.raw["discovery"].update(bootstrap=100, min_events=30, min_days=10)
    return cfg
