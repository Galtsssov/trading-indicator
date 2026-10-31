"""Торговый терминал: структура, похожие ситуации, журнал, аналитика, API (на синтетике, режим повтора)."""
import json

import numpy as np
import pandas as pd
import pytest

from qlab.terminal import journal as jr
from qlab.terminal import similar, structure, sources

from conftest import write_clean


def bars_from(highs, lows, closes):
    n = len(highs)
    return pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
                         "open": closes, "high": highs, "low": lows, "close": closes})


def test_swings_liquidity_equal_and_structure():
    h = np.array([10, 11, 12, 15, 12, 11, 10, 11, 12, 14.98, 12, 11, 10, 11, 12, 13, 16, 13, 12, 11], float)
    l = h - 1
    b = bars_from(h, l, h - 0.5)
    sw = structure.swings(b, n=2)
    highs = [s for s in sw if s.kind == "high"]
    assert [s.i for s in highs][:2] == [3, 9]
    assert highs[0].swept_i == 16 and highs[1].swept_i == 16          # бар 16 выше обоих — ликвидность снята
    liq = structure.liquidity(sw, price=11.0, atr=1.0)
    assert all(x["price"] > 11 for x in liq["above"])
    pre = structure.swings(b.iloc[:15], n=2)                           # до снятия: два близких максимума = EQH
    eq = structure.liquidity(pre, price=11.0, atr=1.0)["equal"]
    assert eq and eq[0]["kind"] == "EQH" and eq[0]["price"] == 15
    bos = structure.internal_structure(b, sw)
    assert bos and bos[-1]["dir"] == 1


def test_similar_forward_ordering():
    c = np.array([100, 100, 101.5, 102.5, 99, 98], float)
    b = pd.DataFrame({"close": c, "high": c + 0.2, "low": c - 0.2, "vwap": np.full(6, 99.5)})
    fw = similar.forward(b, np.array([0]), np.array([1]), np.array([1.0]), h=4)
    r = fw.iloc[0]
    assert r["hit1"] and r["hit2"] and r["new_extreme"] and r["touch_vwap"]
    assert r["p1"] == pytest.approx(0) and r["p2"] == pytest.approx(1.5)
    fw2 = similar.forward(b, np.array([0]), np.array([-1]), np.array([1.0]), h=4)   # шорт: сначала против на 2.7
    assert not fw2.iloc[0]["hit1"]


def test_journal_results_autoclose_and_analytics(tmp_path):
    j = jr.Journal(tmp_path / "j.sqlite", tmp_path / "shots", cost_pct_round_trip=0.0)
    now = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    with pytest.raises(ValueError, match="стоп"):
        j.open(mode="live", symbol="ETHUSDT", tf="5m", side="long", entry=100, stop=101, target=None, setup="", note="",
               context={}, screenshot_data_url=None, opened_at=now)
    tid = j.open(mode="live", symbol="ETHUSDT", tf="5m", side="long", entry=100, stop=99, target=102, setup="тест",
                 note="", context={"conditions": {"is_x": True}}, screenshot_data_url="data:image/png;base64,iVBORw0KGgo=",
                 opened_at=now)
    m1 = pd.DataFrame({"ts": pd.date_range(now, periods=3, freq="1min"), "high": [100.5, 102.5, 101], "low": [99.5, 98.5, 100]})
    assert j.update_open("ETHUSDT", m1, now) == [tid]                  # стоп и цель в одной минуте → стоп
    t = j.list()[0]
    assert t["exit_reason"] == "stop" and t["result_r"] == pytest.approx(-1) and (tmp_path / "shots" / f"{tid}.png").exists()

    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        good = i % 2 == 0
        rows.append({"r": (1.0 if good else -1.0) + rng.normal(0, 0.3), "setup": "A" if i % 3 else "B", "side": "long",
                     "symbol": "ETHUSDT", "tf": "5m", "is_good": good, "is_bad": not good, "is_rare_good": i in (0, 2, 4, 6)})
    a = jr.analyze(pd.DataFrame(rows))
    verdict = {f["condition"]: f["verdict"] for f in a["filters"]}
    assert verdict["is_good"] == "улучшает" and verdict["is_bad"] == "ухудшает"
    assert verdict["is_rare_good"] == "наблюдать"      # 4 сделки < минимума 10 — выводов не делаем, даже если всё в плюс
    assert a["n"] == 80 and "Сетап" in a["groups"]


def test_binance_parsers(monkeypatch, tmp_path):
    row = [1704067200000, "100", "101", "99", "100.5", "10", 1704067259999, "1005", 7, "4", "402", "0"]
    k = sources._klines([row])
    assert k["ts"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC") and k["taker_buy_volume"].iloc[0] == 4
    replies = {
        "/futures/data/openInterestHist": [{"timestamp": 1704067200000, "sumOpenInterest": "123.4"}],
        "/futures/data/globalLongShortAccountRatio": [{"timestamp": 1704067200000, "longShortRatio": "1.5"}],
        "/futures/data/takerlongshortRatio": [{"timestamp": 1704067200000, "buySellRatio": "0.9"}],
        "/fapi/v1/fundingRate": [{"fundingTime": 1704067200000, "fundingRate": "0.0001"}],
        "/fapi/v1/premiumIndex": {"markPrice": "100.1", "lastFundingRate": "0.0002", "nextFundingTime": 1704096000000},
    }
    monkeypatch.setattr(sources, "_get", lambda path, **kw: replies[path])
    src = sources.LiveSource(["ETHUSDT"], tmp_path)
    src._poll_metrics("ETHUSDT")
    m = src.metrics("ETHUSDT")
    assert list(m.columns) == ["ts", "oi", "ls_ratio_accounts", "taker_ls_vol_ratio"] and m["oi"].iloc[0] == 123.4
    assert src.info("ETHUSDT")["funding_estimate"] == 0.0002 and src.funding("ETHUSDT")["funding_rate"].iloc[0] == 0.0001


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from qlab.cache import Data
    from qlab.config import Config
    from qlab.terminal.app import create_app
    from qlab.terminal.market import Market

    tmp = tmp_path_factory.mktemp("terminal")
    syms = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
    write_clean(tmp / "data", syms, days=50)
    cfg = Config.load("configs/platform.toml")
    cfg.raw["data"]["dir"] = str(tmp / "data")
    hist = tmp / "history"
    hist.mkdir()
    similar.build_history(Data(cfg).bars("ETHUSDT", "5m"), "ETHUSDT", "5m").to_parquet(hist / "ETHUSDT_5m.parquet")
    src = sources.ReplaySource(syms, tmp / "data" / "clean", "2024-02-10T15:00", speed=0)
    journal = jr.Journal(tmp / "j.sqlite", tmp / "shots", 0.0014)
    app = create_app(src, Market(src, cfg.section("features"), cfg.section("availability")), journal, hist, syms, ["5m", "15m"])
    with TestClient(app) as c:
        yield c


def test_api_end_to_end(client):
    cfg = client.get("/api/config").json()
    assert cfg["mode"] == "replay" and cfg["history"]["ETHUSDT_5m"] and not cfg["history"]["SOLUSDT_5m"]
    st = client.get("/api/state", params={"symbol": "ETHUSDT", "tf": "5m"}).json()
    assert len(st["candles"]) > 500 and st["facts"]["groups"] and st["lines"] and st["regime"]
    assert all(c["time"] <= st["now"] for c in st["candles"])                  # никаких свечей из будущего
    inc = client.get("/api/state", params={"symbol": "ETHUSDT", "tf": "5m", "since": st["candles"][-1]["time"]}).json()
    assert inc["incremental"] and len(inc["candles"]) <= 2
    sim = client.get("/api/similar", params={"symbol": "ETHUSDT", "tf": "5m"}).json()
    if sim["event"]:
        assert sim["available"] and pd.Timestamp(sim["event"]["ts"]) <= pd.Timestamp(st["now"], unit="s", tz="UTC")
    price, atr = st["price"], st["facts"]["atr"]
    bad = client.post("/api/trades", json={"symbol": "ETHUSDT", "tf": "5m", "side": "long", "stop": price + atr})
    assert bad.status_code == 400
    tid = client.post("/api/trades", json={"symbol": "ETHUSDT", "tf": "5m", "side": "short", "stop": price + atr,
                                           "target": price - 2 * atr, "setup": "тест"}).json()["id"]
    t = client.get("/api/trades").json()[0]
    assert t["id"] == tid and t["status"] == "open" and t["conditions"]["is_rr_2"]
    assert client.post(f"/api/trades/{tid}/close", json={"exit": price - atr}).json()["ok"]
    a = client.get("/api/analytics").json()
    assert a["n"] == 1 and a["overall"]["expectancy"] == pytest.approx(1 - 0.0014 * price / atr, rel=1e-6)
    assert client.get("/api/state", params={"symbol": "XRPUSDT", "tf": "5m"}).status_code == 404
