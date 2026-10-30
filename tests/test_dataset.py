import numpy as np
import pandas as pd

from qlab import dataset


def test_resample_ohlc_and_completeness(m1):
    part = m1.drop(index=[7])  # одна пропущенная минута во втором 5m-баре
    bars = dataset.resample(part, "5m")
    first = m1.iloc[:5]
    assert bars["open"].iloc[0] == first["open"].iloc[0]
    assert bars["high"].iloc[0] == first["high"].max()
    assert bars["low"].iloc[0] == first["low"].min()
    assert bars["close"].iloc[0] == first["close"].iloc[-1]
    assert bars["complete"].iloc[0] and not bars["complete"].iloc[1]
    assert (bars["available_at"] - bars["ts"] == pd.Timedelta(minutes=5)).all()


def test_oi_respects_publication_lag(m1):
    ts = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10"], utc=True).as_unit("ns")
    oi = pd.DataFrame({"ts": ts, "oi": [1.0, 2.0, 3.0]})
    bars = dataset.build(m1.iloc[:30], "5m", oi=oi, funding=None, btc_m1=None,
                         oi_lag_minutes=5, oi_max_staleness_minutes=30)
    # бар 00:00–00:05 закрыт в 00:05; снимок 00:00 опубликован в 00:05 → виден; снимок 00:05 — ещё нет
    assert bars["oi"].iloc[0] == 1.0
    assert bars["oi"].iloc[1] == 2.0
    assert (bars["oi_available_at"].dropna() <= bars.loc[bars["oi_available_at"].notna(), "available_at"]).all()


def test_stale_oi_becomes_unknown(m1):
    oi = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 00:00"], utc=True).as_unit("ns"), "oi": [1.0]})
    bars = dataset.build(m1.iloc[:120], "5m", oi=oi, funding=None, btc_m1=None,
                         oi_lag_minutes=0, oi_max_staleness_minutes=30)
    assert np.isnan(bars["oi"].iloc[-1])
