"""Тест причинности: значения признаков и событий до момента t не меняются, если обрезать будущее."""
import numpy as np
import pandas as pd
import pytest

from qlab import dataset, features
from qlab.events import sweep

from conftest import random_oi, random_walk_m1


@pytest.fixture(scope="module")
def full():
    m1 = random_walk_m1(20 * 1440, seed=3)
    btc = random_walk_m1(20 * 1440, seed=4)
    bars = dataset.build(m1, "5m", oi=random_oi(m1, 5), funding=None, btc_m1=btc,
                         oi_lag_minutes=5, oi_max_staleness_minutes=30)
    return m1, btc, bars


@pytest.mark.parametrize("cut_frac", [0.31, 0.58, 0.87])
def test_features_are_causal(full, cut_frac):
    m1, btc, bars = full
    feats = features.compute(bars, 5)
    cut = int(len(bars) * cut_frac)
    truncated = features.compute(bars.iloc[:cut].copy(), 5)
    for f in features.load_all().values():
        for col in f.provides:
            a, b = feats[col].iloc[:cut], truncated[col]
            if a.dtype.kind in "fc":
                np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, equal_nan=True,
                                           err_msg=f"{f.name}.{col} подглядывает в будущее")
            else:
                assert (a.to_numpy() == b.to_numpy()).all(), f"{f.name}.{col} подглядывает в будущее"


def test_sweeps_are_causal(full):
    _, _, bars = full
    feats = features.compute(bars, 5, names=["atr"])
    p = sweep.SweepParams()
    ev = sweep.detect(feats, p)
    cut = int(len(feats) * 0.6)
    ev_cut = sweep.detect(feats.iloc[:cut].copy(), p)
    pd.testing.assert_frame_equal(ev[ev["bar"] < cut].reset_index(drop=True), ev_cut)
    assert len(ev_cut) > 10
