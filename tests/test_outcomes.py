import numpy as np
import pandas as pd

from qlab import outcomes

P = outcomes.OutcomeParams(r_atr=1.0, targets=(1.0, 2.0), horizon_minutes=5, cost_pct_round_trip=0.01)


def path(rows, start="2024-01-01 00:05"):
    ts = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC").as_unit("ns")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c})


def event(direction=1, atr=1.0, at="2024-01-01 00:05"):
    return pd.DataFrame({"direction": [direction], "atr_ref": [atr],
                         "available_at": [pd.Timestamp(at, tz="UTC").as_unit("ns")]})


def test_long_target_before_stop():
    m1 = path([(100, 100.5, 99.5, 100.2), (100.2, 101.2, 100.1, 101), (101, 102.1, 100.5, 102),
               (102, 102, 98, 98), (98, 98, 98, 98)])
    r = outcomes.compute(event(1), m1, P).iloc[0]
    assert r["res_1r"] == 1.0 and r["t_hit_1r"] == 1
    assert r["res_2r"] == 2.0 and r["t_hit_2r"] == 2
    assert r["t_stop"] == 3
    assert np.isclose(r["mfe_r"], 2.1) and np.isclose(r["mae_r"], 2.0)
    assert np.isclose(r["cost_r"], 1.0)  # 1% издержек при R = 1% цены
    assert np.isclose(r["res_1r_net"], 0.0)


def test_same_minute_counts_as_stop():
    m1 = path([(100, 101.5, 98.5, 100)] + [(100, 100, 100, 100)] * 4)
    r = outcomes.compute(event(1), m1, P).iloc[0]
    assert r["res_1r"] == -1.0


def test_short_direction_and_time_exit():
    m1 = path([(100, 100.2, 99.8, 99.9)] * 4 + [(99.9, 99.9, 99.5, 99.6)])
    r = outcomes.compute(event(-1), m1, P).iloc[0]
    assert np.isnan(r["t_stop"]) and np.isnan(r["t_hit_1r"])
    assert np.isclose(r["res_1r"], 0.4)  # выход по close последней минуты: (100 − 99.6) / 1


def test_gap_in_path_is_excluded():
    m1 = path([(100, 100, 100, 100)] * 6).drop(index=[2])
    r = outcomes.compute(event(1), m1, P).iloc[0]
    assert r["exclude_reason"] == "разрыв минутных данных внутри горизонта"
    assert np.isnan(r["res_1r"])
