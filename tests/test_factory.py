"""Feature Factory: каждый признак проверяется автоматически, формулы — на данных с известным ответом."""
import numpy as np
import pandas as pd
import pytest

from qlab import dataset, factory, features
from qlab.factory import analysis, docs

from conftest import random_funding, random_oi, random_walk_m1

SYMS = ("ETHUSDT", "SOLUSDT", "BTCUSDT")


def _bars(m1, btc, seed):
    oi = random_oi(m1, seed)
    rng = np.random.default_rng(seed)
    oi = oi.assign(ls_ratio_accounts=1 + rng.random(len(oi)), taker_ls_vol_ratio=0.5 + rng.random(len(oi)))
    fund = random_funding(m1).assign(funding_rate=lambda d: rng.normal(1e-4, 5e-5, len(d)))
    b = dataset.build(m1, "5m", oi=oi, funding=fund, btc_m1=btc, oi_lag_minutes=5, oi_max_staleness_minutes=30)
    return features.compute(b, 5)


@pytest.fixture(scope="module")
def world():
    m = {s: random_walk_m1(45 * 1440, seed=10 + i) for i, s in enumerate(SYMS)}
    bars = {s: _bars(m[s], m["BTCUSDT"], i) for i, s in enumerate(SYMS)}
    others = {s: bars[s].set_index("ts")[["close", "volume"]] for s in SYMS}
    return bars, others


def test_registry_shape_and_docs():
    reg = factory.load_all()
    assert 200 <= len(reg) <= 500
    cats = {s.category for s in reg.values()}
    assert len(cats) >= 30
    assert len({s.id for s in reg.values()}) == len(reg)
    for s in reg.values():
        for field in ("formula", "unit", "cost", "mechanism", "measures", "why", "limits"):
            assert getattr(s, field).strip(), f"{s.name}: пустое поле {field}"
        assert s.kind in factory.KINDS and s.deps


def test_all_features_are_causal_and_finite(world):
    bars, others = world
    b = bars["ETHUSDT"]
    full = factory.compute(b, "ETHUSDT", 5, others)
    assert not np.isinf(full.to_numpy()).any()
    for cut in (int(len(b) * 0.55), int(len(b) * 0.9)):
        part = factory.compute(b.iloc[:cut].copy(), "ETHUSDT", 5,
                               {s: o[o.index < b["ts"].iloc[cut]] for s, o in others.items()})
        bad = [c for c in full.columns if not np.allclose(full[c].iloc[:cut].to_numpy(np.float64),
                                                          part[c].to_numpy(np.float64), rtol=1e-4, atol=1e-6,
                                                          equal_nan=True)]
        assert not bad, f"подглядывают в будущее: {bad}"
    tail = full.iloc[len(full) // 2:]
    self_ctx = [c for c in full.columns if c.startswith("cx_eth_")]
    empty = [c for c in full.columns if c not in self_ctx and tail[c].isna().mean() > 0.5]
    assert not empty, f"почти пустые признаки: {empty}"
    assert full[self_ctx].isna().all().all()          # контекст монеты для самой себя — NaN по определению


def ctx_from(close, high=None, low=None, open_=None, volume=None, others=None, start="2024-01-01"):
    n = len(close)
    c = pd.Series(close, dtype=float)
    b = pd.DataFrame({"ts": pd.date_range(start, periods=n, freq="5min", tz="UTC"),
                      "open": c.shift().fillna(c.iloc[0]) if open_ is None else open_,
                      "high": c if high is None else high, "low": c if low is None else low, "close": c,
                      "volume": 1.0 if volume is None else volume, "atr": 1.0})
    return factory.Ctx(b, "ETHUSDT", 5, others)


def f(name, x):
    return factory.load_all()[name].fn(x)


def test_formulas_on_known_inputs():
    up = ctx_from(np.arange(1, 401, dtype=float) + 100)
    assert f("mr_rsi_14", up).iloc[-1] == pytest.approx(100)
    assert f("en_permutation_24", up).iloc[-1] == pytest.approx(0, abs=1e-12)   # монотонный путь — один узор
    assert f("fr_katz_24", up).iloc[-1] == pytest.approx(1, abs=1e-9)           # прямая — размерность 1
    assert f("rg_efficiency_24", up).iloc[-1] == pytest.approx(1)
    assert f("sq_up_streak", up).iloc[-1] == 399
    assert f("ps_pos_range_24", up).iloc[-1] == pytest.approx(1)
    assert f("rg_trend_tstat_24", up).iloc[-1] > 100
    zig = ctx_from(100 + np.tile([0.0, 1.0], 200))
    assert f("st_autocorr1_24", zig).iloc[-1] == pytest.approx(-1)
    assert f("lq_roll_spread_24", zig).iloc[-1] > 0                               # отскок bid/ask → спред > 0
    assert f("lq_roll_spread_24", up).iloc[-1] == 0                               # ровный тренд → 0
    assert f("rg_variance_ratio_2", zig).iloc[-1] < 0.1                           # сильный возврат
    i = np.arange(400)
    convex = ctx_from(100 * np.exp(1e-5 * i ** 2 + np.tile([0, 1e-4], 200)))      # выпукло в логарифме цены
    concave = ctx_from(100 * np.exp(-1e-5 * i ** 2 + np.tile([0, 1e-4], 200)))
    assert f("cv_quad_48", convex).iloc[-1] > 0 > f("cv_quad_48", concave).iloc[-1]
    flat = ctx_from(np.full(400, 100.0), volume=np.arange(1, 401, dtype=float))
    assert f("vw_day_dist", flat).iloc[-1] == pytest.approx(0)
    assert f("ds_round_major", ctx_from(np.full(30, 3050.0))).iloc[-1] == pytest.approx(50)  # шаг 100 для 3050


def test_cross_asset_with_identical_series():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, 400)))
    ts = pd.date_range("2024-01-01", periods=400, freq="5min", tz="UTC")
    others = {"BTCUSDT": pd.DataFrame({"close": close, "volume": 1.0}, index=ts)}
    x = ctx_from(close, others=others)
    assert f("cx_btc_beta_288", x).iloc[-1] == pytest.approx(1)
    assert f("cx_btc_corr_96", x).iloc[-1] == pytest.approx(1)
    assert f("cx_btc_spread_24", x).iloc[-1] == pytest.approx(0, abs=1e-12)
    assert np.isnan(f("cx_sol_ret_1", x).iloc[-1])                                # нет данных SOL → NaN


def test_session_clock_with_dst():
    x = ctx_from(np.full(3, 100.0), start="2024-07-01 13:25")                     # бар закрывается в 09:30 Нью-Йорка
    assert f("tm_min_since_ny_open", x).iloc[0] == 0
    assert f("tm_us_dst", x).iloc[0] == 1
    w = ctx_from(np.full(3, 100.0), start="2024-01-08 14:25")                     # зимой 09:30 = 14:30 UTC
    assert f("tm_min_since_ny_open", w).iloc[0] == 0 and f("tm_us_dst", w).iloc[0] == 0


def test_dedupe_and_clusters():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    frame = pd.DataFrame({"a": a, "a_copy": 3 * a + 1, "a_noisy": a + rng.normal(0, 0.3, 2000),
                          "b": rng.normal(size=2000), "const": 1.0})
    rho, degenerate = analysis.spearman(frame)
    assert degenerate == ["const"]
    kept, dropped = analysis.dedupe(rho, ["a", "a_copy", "a_noisy", "b"], 0.99)
    assert kept == ["a", "a_noisy", "b"] and dropped.iloc[0]["twin"] == "a"
    labels, _ = analysis.clusters(rho, 0.7)
    assert labels["a"] == labels["a_copy"] == labels["a_noisy"] != labels["b"]


def test_catalog_is_up_to_date():
    from pathlib import Path
    assert Path("docs/FEATURES.md").read_text(encoding="utf-8") == docs.render(), \
        "каталог устарел: выполните `python -m qlab catalog`"
