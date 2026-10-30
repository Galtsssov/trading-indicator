import pandas as pd

from qlab.data import binance

ROW = "1704067200000,100,101,99,100.5,10,1704067259999,1005,7,4,402,0"


def test_klines_with_and_without_header(tmp_path):
    no_header = tmp_path / "a.csv"
    no_header.write_text(ROW + "\n")
    with_header = tmp_path / "b.csv"
    with_header.write_text(",".join(binance.KLINE_COLS) + "\n" + ROW + "\n")
    a, b = binance.parse_klines(no_header), binance.parse_klines(with_header)
    pd.testing.assert_frame_equal(a, b)
    assert a["ts"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert a["taker_buy_volume"].iloc[0] == 4


def test_microsecond_epoch():
    s = pd.Series([1704067200000000])
    assert binance.epoch_to_utc(s).iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_plan_uses_daily_for_current_month(tmp_path):
    from datetime import date
    files = binance.plan("https://x", tmp_path, "ETHUSDT", "2024-01", date(2024, 3, 3))
    klines = [f.url for f in files if f.dataset == "klines_1m"]
    assert any("monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2024-02.zip" in u for u in klines)
    assert not any("2024-03.zip" in u for u in klines)
    assert any("daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-03-02.zip" in u for u in klines)
    assert not any("2024-03-03" in u for u in klines)  # сегодняшний день ещё не опубликован
