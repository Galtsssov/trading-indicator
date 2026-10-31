"""Структура рынка для графика — объективно, по формулам, без субъективной разметки.

Все функции берут только закрытые бары (complete). Свинг подтверждается через `n` баров справа —
до этого он не показывается (на графике видна задержка подтверждения, будущее не используется).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from qlab import events


@dataclass
class Swing:
    i: int
    ts: pd.Timestamp
    price: float
    kind: str                # "high" / "low"
    confirmed_i: int
    swept_i: int | None = None


def swings(bars: pd.DataFrame, n: int = 3) -> list[Swing]:
    """Свинг-хай: high[i] строго выше n баров слева и не ниже n баров справа (свинг-лоу — зеркально)."""
    h, l = bars["high"].to_numpy(), bars["low"].to_numpy()
    out: list[Swing] = []
    for i in range(n, len(bars) - n):
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:i + n + 1].max():
            out.append(Swing(i, bars["ts"].iloc[i], float(h[i]), "high", i + n))
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:i + n + 1].min():
            out.append(Swing(i, bars["ts"].iloc[i], float(l[i]), "low", i + n))
    for s in out:                          # когда уровень снят (цена прошла за него после подтверждения)
        after = slice(s.confirmed_i + 1, len(bars))
        hit = np.flatnonzero(h[after] > s.price) if s.kind == "high" else np.flatnonzero(l[after] < s.price)
        s.swept_i = int(s.confirmed_i + 1 + hit[0]) if len(hit) else None
    return sorted(out, key=lambda s: (s.confirmed_i, s.i))


def liquidity(swing_list: list[Swing], price: float, atr: float, keep: int = 4, eq_tol_atr: float = 0.15) -> dict:
    """Неснятые свинги = лежащая ликвидность (стопы за экстремумами). Близкие — equal highs / lows."""
    open_hi = sorted([s for s in swing_list if s.kind == "high" and s.swept_i is None and s.price > price], key=lambda s: s.price)
    open_lo = sorted([s for s in swing_list if s.kind == "low" and s.swept_i is None and s.price < price], key=lambda s: -s.price)
    equal = []
    for side, group in (("EQH", open_hi), ("EQL", open_lo)):
        for a, b in zip(group, group[1:]):
            if atr > 0 and abs(a.price - b.price) <= eq_tol_atr * atr:
                equal.append({"kind": side, "price": max(a.price, b.price) if side == "EQH" else min(a.price, b.price),
                              "touches": 2, "from_ts": min(a.ts, b.ts)})
    return {"above": [{"price": s.price, "ts": s.ts} for s in open_hi[:keep]],
            "below": [{"price": s.price, "ts": s.ts} for s in open_lo[:keep]], "equal": equal}


def internal_structure(bars: pd.DataFrame, swing_list: list[Swing]) -> list[dict]:
    """BOS / CHoCH: закрытие за последним подтверждённым свингом. BOS — по текущему направлению структуры,
    CHoCH — против него (смена характера). Направление задаётся последним пробоем."""
    c = bars["close"].to_numpy()
    by_confirm: dict[int, list[Swing]] = {}
    for s in swing_list:
        by_confirm.setdefault(s.confirmed_i, []).append(s)
    last_hi = last_lo = None
    trend = 0
    out = []
    for i in range(len(bars)):
        for s in by_confirm.get(i, []):
            if s.kind == "high":
                last_hi = s
            else:
                last_lo = s
        if last_hi is not None and c[i] > last_hi.price:
            out.append({"i": i, "ts": bars["ts"].iloc[i], "dir": 1, "label": "BOS" if trend >= 0 else "CHoCH",
                        "level": last_hi.price})
            trend, last_hi = 1, None
        elif last_lo is not None and c[i] < last_lo.price:
            out.append({"i": i, "ts": bars["ts"].iloc[i], "dir": -1, "label": "BOS" if trend <= 0 else "CHoCH",
                        "level": last_lo.price})
            trend, last_lo = -1, None
    return out


def key_levels(bars: pd.DataFrame, price: float) -> list[dict]:
    """Вчерашние и недельные экстремумы, открытие дня, диапазон азиатской сессии, ближайшее круглое число."""
    ts = bars["ts"]
    day = ts.dt.floor("D")
    today = day.iloc[-1]
    out = []
    prev = bars[day == today - pd.Timedelta(days=1)]
    if len(prev):
        out += [{"name": "PDH", "price": float(prev["high"].max())}, {"name": "PDL", "price": float(prev["low"].min())}]
    week_start = today - pd.Timedelta(days=today.dayofweek)
    pw = bars[(ts >= week_start - pd.Timedelta(days=7)) & (ts < week_start)]
    if len(pw):
        out += [{"name": "PWH", "price": float(pw["high"].max())}, {"name": "PWL", "price": float(pw["low"].min())}]
    cur = bars[day == today]
    out.append({"name": "Открытие дня", "price": float(cur["open"].iloc[0])})
    asia = cur[cur["ts"].dt.hour < 7]
    if len(asia) and cur["ts"].iloc[-1].hour >= 7:
        out += [{"name": "Азия max", "price": float(asia["high"].max())}, {"name": "Азия min", "price": float(asia["low"].min())}]
    step = 10 ** (np.floor(np.log10(price)) - 1)
    out.append({"name": "Круглое", "price": float(round(price / step) * step)})
    return out


def session_range(bars: pd.DataFrame) -> dict:
    """Сессии по UTC: asia 00–07, europe 07–13, us 13–20, late 20–24 — как в признаке session."""
    starts = (0, 7, 13, 20)
    names = ("asia", "europe", "us", "late")
    last = bars["ts"].iloc[-1]
    k = int(np.searchsorted(starts, last.hour, side="right") - 1)
    begin = last.floor("D") + pd.Timedelta(hours=starts[k])
    cur = bars[bars["ts"] >= begin]
    return {"name": names[k], "high": float(cur["high"].max()), "low": float(cur["low"].min()), "start": begin}


IMPULSE_EVENTS = ("atr_expansion", "return_extreme")


def impulses(bars: pd.DataFrame, last: int = 8) -> list[dict]:
    out = []
    for name in IMPULSE_EVENTS:
        ev = events.detect(name, bars)
        out += [{"ts": r.ts, "dir": int(r.dir), "kind": name} for r in ev.tail(last).itertuples()]
    out.sort(key=lambda x: x["ts"])
    return out[-last:]


def as_dict(s: Swing) -> dict:
    return asdict(s)
