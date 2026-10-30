import numpy as np
import pandas as pd

from qlab import events


def bars_from(highs, lows, closes, atr=1.0):
    n = len(highs)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC").as_unit("ns"),
        "open": closes, "high": highs, "low": lows, "close": closes, "atr": atr, "complete": True,
    }).assign(available_at=lambda d: d["ts"] + pd.Timedelta(minutes=5))


def base(n=40):
    highs = np.full(n, 100.0)
    highs[10] = 105.0  # уровень
    return highs, np.full(n, 95.0), np.full(n, 99.0)


def detect(h, l, c):
    return events.detect("liquidity_sweep", bars_from(h, l, c))


def test_wick_sweep_with_close_back_detected():
    h, l, c = base()
    h[28], c[28] = 105.5, 104.0
    ev = detect(h, l, c)
    assert list(ev["bar"]) == [28]
    assert ev["dir"].iloc[0] == 1 and ev["level"].iloc[0] == 105.0
    assert ev["atr_ref"].iloc[0] == 1.0 and ev["event"].iloc[0] == "liquidity_sweep"


def test_close_beyond_level_is_not_a_sweep():
    h, l, c = base()
    h[28], c[28] = 105.5, 105.2
    assert detect(h, l, c).empty


def test_too_shallow_or_too_young_level_ignored():
    h, l, c = base()
    h[28], c[28] = 105.05, 104.0           # вынос 0.05 ATR < 0.10
    assert detect(h, l, c).empty
    h, l, c = base()
    h[26], c[26] = 106.0, 105.5            # бар 26 — пробой с закрытием выше, не свип
    h[28], c[28] = 106.5, 105.0            # максимум за 2 бара до свипа — моложе min_age=3
    assert detect(h, l, c).empty


def test_cooldown_suppresses_repeats_per_direction():
    h, l, c = base(60)
    for i, hi in ((30, 105.5), (35, 106.0), (50, 107.0)):  # 35 — в пределах cooldown после 30
        h[i], c[i] = hi, 104.0
    assert list(detect(h, l, c)["bar"]) == [30, 50]


def test_unknown_params_rejected():
    import pytest
    with pytest.raises(ValueError, match="неизвестные параметры"):
        events.params_for("liquidity_sweep", {"depth": 0.1})
