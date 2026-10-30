import numpy as np
import pandas as pd

from qlab.events import sweep


def bars_from(highs, lows, closes, atr=1.0):
    n = len(highs)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC").as_unit("ns"),
        "high": highs, "low": lows, "close": closes, "atr": atr, "complete": True,
    }).assign(available_at=lambda d: d["ts"] + pd.Timedelta(minutes=5))


def base(n=40):
    highs = np.full(n, 100.0)
    highs[10] = 105.0  # уровень
    return highs, np.full(n, 95.0), np.full(n, 99.0)


P = sweep.SweepParams(lookback=24, min_age=3, depth_atr=0.10, cooldown=12)


def test_wick_sweep_with_close_back_detected():
    h, l, c = base()
    h[28], c[28] = 105.5, 104.0
    ev = sweep.detect(bars_from(h, l, c), P)
    hi = ev[ev["side"] == "high"]
    assert list(hi["bar"]) == [28]
    assert hi["direction"].iloc[0] == -1 and hi["level"].iloc[0] == 105.0
    assert np.isclose(hi["depth_atr"].iloc[0], 0.5)


def test_close_beyond_level_is_not_a_sweep():
    h, l, c = base()
    h[28], c[28] = 105.5, 105.2
    assert (sweep.detect(bars_from(h, l, c), P)["side"] != "high").all()


def test_too_shallow_or_too_young_level_ignored():
    h, l, c = base()
    h[28], c[28] = 105.05, 104.0           # вынос 0.05 ATR < 0.10
    assert (sweep.detect(bars_from(h, l, c), P)["side"] != "high").all()
    h, l, c = base()
    h[26], c[26] = 106.0, 105.5        # бар 26 — пробой с закрытием выше, не свип
    h[28], c[28] = 106.5, 105.0  # максимум за 2 бара до свипа — моложе min_age=3
    assert (sweep.detect(bars_from(h, l, c), P)["side"] != "high").all()


def test_cooldown_suppresses_repeats():
    h, l, c = base(60)
    for i, hi in ((30, 105.5), (35, 106.0), (50, 107.0)):  # 35 — в пределах cooldown после 30
        h[i], c[i] = hi, 104.0
    ev = sweep.detect(bars_from(h, l, c), P)
    assert list(ev[ev["side"] == "high"]["bar"]) == [30, 50]
