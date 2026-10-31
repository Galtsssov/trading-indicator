"""Снимок рынка для терминала: бары, оверлеи графика, факты панели, последние события.

Использует существующий движок qlab без изменений: dataset.build → features.compute → events.detect.
Незакрытый текущий бар показывается на графике, но в расчётах событий и структуры не участвует.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab import dataset, events, features
from qlab.config import TF_MINUTES
from qlab.terminal import similar, structure
from qlab.terminal.sources import Source

SESSION_RU = {"asia": "Азия", "europe": "Европа", "us": "США", "late": "Поздняя (после США)"}
DISPLAY_BARS = 1500
STRUCTURE_BARS = 1500


@dataclass
class Snapshot:
    symbol: str
    tf: str
    bars: pd.DataFrame          # все бары с признаками (включая незакрытый последний)
    closed: pd.DataFrame        # только закрытые
    payload: dict               # JSON для интерфейса


def _t(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())


def _num(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def _pct(x, nd: int = 2):
    return None if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


class Market:
    """Строит и кэширует снимки по (монета, ТФ, версия данных источника)."""

    def __init__(self, source: Source, feature_params: dict, availability: dict):
        self.source = source
        self.feature_params = feature_params
        self.av = availability
        self._cache: dict[tuple, Snapshot] = {}
        self._lock = threading.Lock()

    def snapshot(self, symbol: str, tf: str) -> Snapshot | None:
        key = (symbol, tf, self.source.version)
        with self._lock:
            if key not in self._cache:
                snap = self._build(symbol, tf)
                if snap is None:
                    return None
                self._cache = {k: v for k, v in self._cache.items() if k[:2] != (symbol, tf)}
                self._cache[key] = snap
            return self._cache[key]

    def _build(self, symbol: str, tf: str) -> Snapshot | None:
        src = self.source
        m1, btc = src.minutes(symbol), src.minutes("BTCUSDT")
        if m1 is None or len(m1) < 600:
            return None
        bars = dataset.build(m1, tf, oi=src.metrics(symbol), funding=src.funding(symbol),
                             btc_m1=btc if len(btc) else m1, oi_lag_minutes=self.av["oi_lag_minutes"],
                             oi_max_staleness_minutes=self.av["oi_max_staleness_minutes"])
        bars = features.compute(bars, TF_MINUTES[tf], self.feature_params)
        now = src.now()
        bars["complete"] = bars["complete"] & (bars["available_at"] <= now)
        closed = bars[bars["complete"]].reset_index(drop=True)
        if len(closed) < 100:
            return None
        payload = self._payload(symbol, tf, bars, closed, m1, now)
        return Snapshot(symbol, tf, bars, closed, payload)

    # ------------------------------------------------------------------
    def _payload(self, symbol: str, tf: str, bars: pd.DataFrame, closed: pd.DataFrame, m1: pd.DataFrame,
                 now: pd.Timestamp) -> dict:
        view = bars.tail(DISPLAY_BARS)
        tail = closed.tail(STRUCTURE_BARS).reset_index(drop=True)
        last = closed.iloc[-1]
        price = float(bars["close"].iloc[-1])
        atr = float(last["atr"])
        sw = structure.swings(tail)
        liq = structure.liquidity(sw, price, atr)
        bos = structure.internal_structure(tail, sw)
        levels = structure.key_levels(tail, price)
        sess = structure.session_range(tail)
        imp = structure.impulses(tail)
        evs = self._events(tail, tf)

        markers = []
        t0 = view["ts"].iloc[0]
        for s in sw[-40:]:
            if s.ts >= t0:
                markers.append({"time": _t(s.ts), "position": "aboveBar" if s.kind == "high" else "belowBar",
                                "shape": "circle", "color": "#8a93a6", "size": 0.4, "text": ""})
        for b in bos[-12:]:
            if b["ts"] >= t0:
                markers.append({"time": _t(b["ts"]), "position": "aboveBar" if b["dir"] > 0 else "belowBar",
                                "shape": "square", "size": 0.1, "color": "#c9a227" if b["label"] == "CHoCH" else "#5b8def",
                                "text": b["label"]})
        for e in evs:
            if e["type"] == "liquidity_sweep" and pd.Timestamp(e["ts"]) >= t0:
                markers.append({"time": e["time"], "position": "aboveBar" if e["dir"] > 0 else "belowBar",
                                "shape": "arrowDown" if e["dir"] > 0 else "arrowUp", "color": "#e0569b", "text": "Sweep"})
        for m in imp:
            if m["ts"] >= t0:
                markers.append({"time": _t(m["ts"]), "position": "belowBar" if m["dir"] > 0 else "aboveBar",
                                "shape": "circle", "size": 0.6, "color": "#3ccf91" if m["dir"] > 0 else "#ef5a5a", "text": "Imp"})
        markers.sort(key=lambda m: m["time"])

        lines = [{"price": l["price"], "title": l["name"], "color": "#6c7a96", "style": 2} for l in levels]
        lines += [{"price": x["price"], "title": "Ликв.↑", "color": "#e0569b", "style": 1} for x in liq["above"][:3]]
        lines += [{"price": x["price"], "title": "Ликв.↓", "color": "#e0569b", "style": 1} for x in liq["below"][:3]]
        eq_up = sorted([x for x in liq["equal"] if x["kind"] == "EQH"], key=lambda x: x["price"])[:2]
        eq_dn = sorted([x for x in liq["equal"] if x["kind"] == "EQL"], key=lambda x: -x["price"])[:2]
        lines += [{"price": x["price"], "title": x["kind"], "color": "#f08c3a", "style": 0} for x in eq_up + eq_dn]
        lines += [{"price": sess["high"], "title": "Сессия max", "color": "#4a5368", "style": 3},
                  {"price": sess["low"], "title": "Сессия min", "color": "#4a5368", "style": 3}]

        def series(col):
            v = view[col].to_numpy(float)
            return [{"time": _t(t), "value": round(x, 6)} for t, x in zip(view["ts"], v) if np.isfinite(x)]

        candles = [{"time": _t(r.ts), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                   for r in view[["ts", "open", "high", "low", "close"]].itertuples()]
        facts = self._facts(symbol, tf, bars, closed, m1, liq, sess, now)
        return {"symbol": symbol, "tf": tf, "price": price, "now": _t(now), "candles": candles,
                "volume": [{"time": _t(t), "value": float(v), "color": "rgba(60,207,145,0.25)" if c >= o else "rgba(239,90,90,0.25)"}
                           for t, v, o, c in zip(view["ts"], view["volume"], view["open"], view["close"])],
                "vwap": series("vwap"), "atr": series("atr"), "markers": markers, "lines": lines,
                "facts": facts, "events": evs, "regime": facts["regime_badge"],
                "structure": {"bos": [{"label": b["label"], "dir": b["dir"], "level": b["level"], "ts": str(b["ts"])}
                                      for b in bos[-5:]],
                              "liquidity": {k: [{"price": x["price"], "kind": x.get("kind", k),
                                                 "ts": str(x.get("ts", x.get("from_ts")))} for x in v]
                                            for k, v in liq.items()}}}

    def _events(self, tail: pd.DataFrame, tf: str, lookback: int = 60) -> list[dict]:
        out = []
        for et in similar.EVENT_TYPES:
            ev = events.detect(et, tail)
            ev = ev[ev["bar"] >= len(tail) - lookback]
            for r in ev.itertuples():
                ctx = similar.context(tail, np.array([r.bar])).iloc[0].to_dict()
                out.append({"id": f"{et}:{_t(r.ts)}", "type": et, "name": similar.EVENT_NAMES[et], "dir": int(r.dir),
                            "ts": str(r.ts), "time": _t(r.ts), "bars_ago": int(len(tail) - 1 - r.bar),
                            "price": float(tail["close"].iloc[r.bar]),
                            "context": {k: _num(v) for k, v in ctx.items()}})
        out.sort(key=lambda e: -e["time"])
        return out

    def _facts(self, symbol, tf, bars, closed, m1, liq, sess, now) -> dict:
        last = closed.iloc[-1]
        price = float(bars["close"].iloc[-1])
        atr = float(last["atr"])
        c = closed["close"]
        bpd = 1440 // TF_MINUTES[tf]
        n = min(96, len(closed) - 1)
        er = abs(c.iloc[-1] - c.iloc[-1 - n]) / max(c.diff().abs().tail(n).sum(), 1e-12)
        ret_n = c.iloc[-1] / c.iloc[-1 - n] - 1
        btc_ret_day = last["btc_close"] / closed["btc_close"].iloc[-1 - min(bpd, len(closed) - 1)] - 1
        tb = closed["taker_buy_volume"] / closed["volume"].replace(0, np.nan)
        oi = closed["oi"]
        info = self.source.info(symbol)
        f = self.source.funding(symbol)
        fund_last = float(f["funding_rate"].iloc[-1]) if f is not None and len(f) else None

        def oi_chg(k):
            prev = oi.iloc[-1 - k] if len(oi) > k else np.nan
            return oi.iloc[-1] / prev - 1 if np.isfinite(prev) and prev > 0 else np.nan

        day = bars[bars["ts"].dt.floor("D") == bars["ts"].iloc[-1].floor("D")]
        hour_ranges = (m1.set_index("ts")["high"].resample("1h").max() - m1.set_index("ts")["low"].resample("1h").min())
        hour_ranges = hour_ranges[hour_ranges.index < now.floor("h")]
        same_hour = hour_ranges[hour_ranges.index.hour == ((now + pd.Timedelta(minutes=30)).hour)]
        exp_range = float(same_hour.tail(30).median()) if len(same_hour) else np.nan
        next_ny = _next_ny_open(now)
        regime_trend = {"trend": "Тренд", "range": "Флэт"}.get(last["regime_trend"], "—")
        regime_vol = {"high_vol": "высокая волатильность", "low_vol": "низкая волатильность"}.get(last["regime_vol"], "—")

        def dist(level):
            return None if level is None else {"price": round(level, 6), "atr": _num((level - price) / atr, 2),
                                               "pct": _pct(level / price - 1)}

        up = liq["above"][0]["price"] if liq["above"] else None
        dn = liq["below"][0]["price"] if liq["below"] else None
        groups = [
            ("Режим", [
                ("Рынок", f"{regime_trend}, {regime_vol}", "Тренд/флэт — эффективность движения за сутки выше/ниже медианы 90 дней", "neutral"),
                ("Тренд", f"{'вверх' if ret_n > 0 else 'вниз'} {_pct(ret_n)} · сила {er:.2f}",
                 f"Изменение и эффективность пути (0 — пила, 1 — прямая) за {n} баров", "up" if ret_n > 0 else "down"),
                ("BTC", f"{_pct(btc_ret_day)} за сутки · {'risk on' if last['regime_risk'] == 'risk_on' else 'risk off'}",
                 "Risk on/off — BTC выше/ниже своей средней за 20 дней", "up" if btc_ret_day > 0 else "down"),
            ]),
            ("Волатильность", [
                ("ATR", f"{atr:.4g} · {atr / price * 100:.2f}% · процентиль {last['atr_pctl'] * 100:.0f}",
                 "ATR(14) и его место среди последних 30 дней (0 — самый тихий, 100 — самый бурный)", "neutral"),
                ("Ожидаемый размах часа", f"≈ {exp_range:.4g} ({exp_range / price * 100:.2f}%)" if np.isfinite(exp_range) else "—",
                 "Медианный размах этого же часа суток за последние 30 дней", "neutral"),
            ]),
            ("Позиционирование", [
                ("Funding", f"{fund_last * 100:+.4f}%" if fund_last is not None else "—",
                 (f"Следующий расчёт через {_mins(info.get('next_funding'), now)}" if info.get("next_funding") is not None else ""),
                 "warn" if fund_last is not None and abs(fund_last) > 0.0005 else "neutral"),
                ("Open Interest", f"1 ч {_pct(oi_chg(60 // TF_MINUTES[tf]))} · 24 ч {_pct(oi_chg(bpd))}",
                 "Рост — открываются новые позиции, падение — закрываются (Binance, задержка ~5 мин)", "neutral"),
                ("Long/Short аккаунтов", f"{last['ls_ratio_accounts']:.2f}" if np.isfinite(last["ls_ratio_accounts"]) else "—",
                 "Отношение аккаунтов в лонге к аккаунтам в шорте (Binance)", "neutral"),
                ("Агрессоры", f"покупки {tb.tail(6).mean() * 100:.0f}% (6 баров) · {tb.tail(24).mean() * 100:.0f}% (24)",
                 "Доля объёма, купленного по рынку (taker buy). 50% — равновесие", "up" if tb.tail(6).mean() > 0.55 else "down" if tb.tail(6).mean() < 0.45 else "neutral"),
            ]),
            ("Расстояния", [
                ("VWAP дня", f"{(price - last['vwap']) / atr:+.2f} ATR · {_pct(price / last['vwap'] - 1)}", "Цена относительно VWAP с 00:00 UTC", "neutral"),
                ("Ликвидность выше", _fmt_dist(dist(up)), "Ближайший неснятый свинг-хай (за ним — стопы шортистов)", "neutral"),
                ("Ликвидность ниже", _fmt_dist(dist(dn)), "Ближайший неснятый свинг-лоу (за ним — стопы лонгистов)", "neutral"),
                ("Max / min дня", f"{_fmt_dist(dist(float(day['high'].max())))} / {_fmt_dist(dist(float(day['low'].min())))}", "С 00:00 UTC", "neutral"),
                ("Max / min сессии", f"{_fmt_dist(dist(sess['high']))} / {_fmt_dist(dist(sess['low']))}", f"Сессия {SESSION_RU[sess['name']]}", "neutral"),
            ]),
            ("Время", [
                ("Сессия", f"{SESSION_RU[sess['name']]} · {_mins(now, sess['start'], reverse=True)} с начала", "Часы UTC: Азия 00–07, Европа 07–13, США 13–20", "neutral"),
                ("Открытие NYSE", f"через {_mins(next_ny, now)}", "09:30 Нью-Йорк (с учётом летнего времени)", "neutral"),
            ]),
        ]
        return {"groups": [{"title": t, "items": [{"label": a, "value": b, "hint": h, "tone": tone} for a, b, h, tone in items]}
                           for t, items in groups],
                "regime_badge": f"{regime_trend} · {regime_vol}",
                "atr": atr, "price": price}


def _fmt_dist(d: dict | None) -> str:
    return "—" if d is None else f"{d['price']:.6g} ({d['atr']:+.2f} ATR)"


def _mins(a, b, reverse: bool = False) -> str:
    if a is None or b is None:
        return "—"
    m = int((pd.Timestamp(a) - pd.Timestamp(b)).total_seconds() // 60)
    m = abs(m)
    return f"{m // 60} ч {m % 60:02d} мин" if m >= 60 else f"{m} мин"


def _next_ny_open(now: pd.Timestamp) -> pd.Timestamp:
    local = now.tz_convert("America/New_York")
    day = local.normalize()
    for k in range(8):
        cand = (day + pd.Timedelta(days=k)).replace(hour=9, minute=30)
        if cand > local and cand.dayofweek < 5:
            return cand.tz_convert("UTC")
    return now
