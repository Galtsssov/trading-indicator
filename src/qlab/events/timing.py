"""События по времени: открытия сессий, пробой азиатского диапазона, окна макро-новостей.

Часы открытия берутся в местном времени биржевого города (учёт летнего времени через zoneinfo).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.events import event, frame, sign_of_bar

SESSIONS = {"asia": ("UTC", 0, 0, False), "london": ("Europe/London", 8, 0, True),
            "ny": ("America/New_York", 9, 30, True)}


def _contains_local_time(bars: pd.DataFrame, tz: str, hour: int, minute: int) -> np.ndarray:
    local = bars["ts"].dt.tz_convert(tz)
    start = local.dt.hour * 60 + local.dt.minute
    step = (bars["ts"].iloc[1] - bars["ts"].iloc[0]) / pd.Timedelta(minutes=1)
    t = hour * 60 + minute
    return ((start <= t) & (t < start + step)).to_numpy().copy()


@event("session_open", requires=("ts", "open", "close"), defaults={"session": "ny"},
       description="бар, содержащий открытие сессии (asia 00:00 UTC, london 08:00, ny 09:30 местного); "
                   "dir — знак свечи")
def session_open(bars: pd.DataFrame, p) -> pd.DataFrame:
    tz, h, m, weekdays_only = SESSIONS[p["session"]]
    mask = _contains_local_time(bars, tz, h, m)
    if weekdays_only:
        mask = mask & (bars["ts"].dt.tz_convert(tz).dt.day_of_week < 5).to_numpy()
    return frame(bars, mask, sign_of_bar(bars))


@event("range_break", requires=("ts", "high", "low", "close"), defaults={"range_end_hour": 7},
       description="первое за день закрытие за диапазоном 00:00–07:00 UTC")
def range_break(bars: pd.DataFrame, p) -> pd.DataFrame:
    day = bars["ts"].dt.floor("D")
    in_range = bars["ts"].dt.hour < p["range_end_hour"]
    hi = bars["high"].where(in_range).groupby(day).cummax().groupby(day).ffill()
    lo = bars["low"].where(in_range).groupby(day).cummin().groupby(day).ffill()
    after = ~in_range
    up = after & (bars["close"] > hi)
    dn = after & (bars["close"] < lo)
    first_up = up & (up.astype(int).groupby(day).cumsum() == 1)
    first_dn = dn & (dn.astype(int).groupby(day).cumsum() == 1)
    return frame(bars, first_up | first_dn, np.where(first_up, 1, -1))


def macro_times(start: pd.Timestamp, end: pd.Timestamp, calendar_csv: str) -> pd.DatetimeIndex:
    """NFP по правилу «первая пятница месяца, 08:30 Нью-Йорк» (приблизительно: переносы не учтены)
    плюс любые события из CSV (колонка time_utc), если файл есть."""
    months = pd.date_range(start.tz_convert(None).normalize().replace(day=1), end.tz_convert(None), freq="MS")
    nfp = [m + pd.offsets.Week(weekday=4) if m.dayofweek != 4 else m for m in months]
    times = [pd.Timestamp(d.date()).replace(hour=8, minute=30).tz_localize("America/New_York").tz_convert("UTC")
             for d in nfp]
    path = Path(calendar_csv)
    if path.exists():
        times += list(pd.to_datetime(pd.read_csv(path)["time_utc"], utc=True))
    return pd.DatetimeIndex(sorted(set(times))).as_unit("ns")


@event("news_window", requires=("ts", "available_at", "open", "close"),
       defaults={"calendar_csv": "data/calendar/macro.csv"},
       description="бар, внутри которого вышла макро-новость (NFP по правилу + календарь из CSV); dir — знак свечи")
def news_window(bars: pd.DataFrame, p) -> pd.DataFrame:
    times = macro_times(bars["ts"].iloc[0], bars["ts"].iloc[-1], p["calendar_csv"])
    pos = np.searchsorted(bars["ts"].to_numpy(), times.to_numpy(), side="right") - 1
    mask = np.zeros(len(bars), dtype=bool)
    ok = pos >= 0
    hit = pos[ok][times[ok].to_numpy() < bars["available_at"].to_numpy()[pos[ok]]]
    mask[hit] = True
    return frame(bars, mask, sign_of_bar(bars))
