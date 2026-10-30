"""Тест причинности: признаки и ВСЕ события библиотеки до момента t не меняются, если обрезать будущее."""
import numpy as np
import pandas as pd
import pytest

from qlab import dataset, discovery, events, features
from qlab.config import Config

from conftest import random_funding, random_oi, random_walk_m1


@pytest.fixture(scope="module")
def bars():
    m1 = random_walk_m1(40 * 1440, seed=3)
    btc = random_walk_m1(40 * 1440, seed=4)
    b = dataset.build(m1, "5m", oi=random_oi(m1, 5), funding=random_funding(m1), btc_m1=btc,
                      oi_lag_minutes=5, oi_max_staleness_minutes=30)
    return features.compute(b, 5)


@pytest.mark.parametrize("cut_frac", [0.31, 0.87])
def test_features_are_causal(bars, cut_frac):
    raw = bars[[c for c in bars.columns if c in dataset.BAR_AGG or c in
                ("ts", "available_at", "n_minutes", "complete", "oi", "oi_ts", "oi_available_at",
                 "funding_rate", "funding_ts", "funding_available_at", "btc_close", "btc_complete")]]
    cut = int(len(raw) * cut_frac)
    truncated = features.compute(raw.iloc[:cut].copy(), 5)
    for f in features.load_all().values():
        for col in f.provides:
            a, b = bars[col].iloc[:cut], truncated[col]
            if a.dtype.kind in "fc":
                np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, equal_nan=True,
                                           err_msg=f"{f.name}.{col} подглядывает в будущее")
            else:
                assert (a.to_numpy() == b.to_numpy()).all(), f"{f.name}.{col} подглядывает в будущее"


SPECS = [(label, etype, params) for label, etype, params in discovery.specs(Config.load("configs/platform.toml"))]


@pytest.mark.parametrize("label,etype,params", SPECS, ids=[s[0] for s in SPECS])
def test_events_are_causal(bars, label, etype, params):
    ev = events.detect(etype, bars, params)
    cut = int(len(bars) * 0.6)
    ev_cut = events.detect(etype, bars.iloc[:cut].copy(), params)
    pd.testing.assert_frame_equal(ev[ev["bar"] < cut].reset_index(drop=True), ev_cut, check_dtype=False)
