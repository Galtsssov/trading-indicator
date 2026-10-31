"""Источники рыночных данных терминала.

- LiveSource — публичный REST Binance USDⓈ-M Futures (без ключей): минутные свечи, funding, OI,
  long/short. Минутки кэшируются на диск, при старте догружается только недостающее.
- ReplaySource — проигрывание локального архива (`python -m qlab fetch`) с ускорением:
  тренажёр без риска и способ проверить терминал без интернета.

Оба источника отдают одинаковые таблицы в формате qlab: 1m-свечи (ts, open, high, low, close, volume,
quote_volume, taker_buy_volume, trades), метрики (ts, oi, ls_ratio_accounts, taker_ls_vol_ratio) и funding.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
FAPI = "https://fapi.binance.com"
MINUTE = pd.Timedelta(minutes=1)


@dataclass
class Status:
    ok: bool = False
    message: str = "запуск"
    updated_at: pd.Timestamp | None = None
    progress: dict[str, str] = field(default_factory=dict)


class Source:
    """Общий интерфейс: всё, что нужно терминалу от рынка."""
    mode = "base"
    symbols: list[str]
    status: Status
    version: int = 0                          # растёт при каждом обновлении данных

    def now(self) -> pd.Timestamp: ...
    def minutes(self, symbol: str) -> pd.DataFrame: ...
    def metrics(self, symbol: str) -> pd.DataFrame | None: ...
    def funding(self, symbol: str) -> pd.DataFrame | None: ...
    def info(self, symbol: str) -> dict: ...
    def start(self) -> None: ...


# ---------------- Live ----------------

def _get(path: str, **params) -> object:
    url = f"{FAPI}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _klines(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
                                     "trades", "taker_buy_volume", "taker_buy_quote", "ignore"])
    out = pd.DataFrame({"ts": pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True).dt.as_unit("ns")})
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"):
        out[c] = df[c].astype(float)
    out["trades"] = df["trades"].astype("int64")
    return out


class LiveSource(Source):
    mode = "live"

    def __init__(self, symbols: list[str], cache_dir: Path, history_days: int = 30, poll_seconds: float = 4.0,
                 metrics_seconds: float = 60.0):
        self.symbols = list(symbols)
        self.cache_dir = cache_dir
        self.history_days = history_days
        self.poll_seconds = poll_seconds
        self.metrics_seconds = metrics_seconds
        self.status = Status()
        self._m1: dict[str, pd.DataFrame] = {}
        self._metrics: dict[str, pd.DataFrame] = {}
        self._funding: dict[str, pd.DataFrame] = {}
        self._info: dict[str, dict] = {}
        self._lock = threading.Lock()

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def minutes(self, symbol: str) -> pd.DataFrame:
        with self._lock:
            return self._m1.get(symbol, pd.DataFrame())

    def metrics(self, symbol: str) -> pd.DataFrame | None:
        return self._metrics.get(symbol)

    def funding(self, symbol: str) -> pd.DataFrame | None:
        return self._funding.get(symbol)

    def info(self, symbol: str) -> dict:
        return self._info.get(symbol, {})

    def start(self) -> None:
        threading.Thread(target=self._run, name="live-source", daemon=True).start()

    # --- загрузка ---
    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"m1_{symbol}.parquet"

    def _backfill(self, symbol: str) -> None:
        path = self._cache_path(symbol)
        df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        start = self.now().floor("min") - pd.Timedelta(days=self.history_days)
        if not df.empty:
            df = df[df["ts"] >= start]
            start = max(start, df["ts"].iloc[-1] + MINUTE)
        parts = [df] if not df.empty else []
        while start < self.now() - MINUTE:
            self.status.progress[symbol] = f"загрузка истории с {start:%d.%m %H:%M}"
            rows = _get("/fapi/v1/klines", symbol=symbol, interval="1m", startTime=int(start.timestamp() * 1000), limit=1500)
            if not rows:
                break
            chunk = _klines(rows)
            parts.append(chunk)
            start = chunk["ts"].iloc[-1] + MINUTE
        full = pd.concat(parts, ignore_index=True).drop_duplicates("ts", keep="last").sort_values("ts")
        closed = full[full["ts"] + MINUTE <= self.now()]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        closed.to_parquet(path, index=False)
        with self._lock:
            self._m1[symbol] = full.reset_index(drop=True)
        self.status.progress[symbol] = "готово"

    def _poll_klines(self, symbol: str) -> None:
        tail = _klines(_get("/fapi/v1/klines", symbol=symbol, interval="1m", limit=5))
        with self._lock:
            df = self._m1[symbol]
            df = pd.concat([df[df["ts"] < tail["ts"].iloc[0]], tail], ignore_index=True)
            cutoff = self.now() - pd.Timedelta(days=self.history_days)
            self._m1[symbol] = df[df["ts"] >= cutoff].reset_index(drop=True)

    def _poll_metrics(self, symbol: str) -> None:
        oi = _get("/futures/data/openInterestHist", symbol=symbol, period="5m", limit=500)
        ls = _get("/futures/data/globalLongShortAccountRatio", symbol=symbol, period="5m", limit=500)
        tk = _get("/futures/data/takerlongshortRatio", symbol=symbol, period="5m", limit=500)
        frame = pd.DataFrame({"ts": [r["timestamp"] for r in oi], "oi": [float(r["sumOpenInterest"]) for r in oi]})
        for rows, col, key in ((ls, "ls_ratio_accounts", "longShortRatio"), (tk, "taker_ls_vol_ratio", "buySellRatio")):
            frame = frame.merge(pd.DataFrame({"ts": [r["timestamp"] for r in rows], col: [float(r[key]) for r in rows]}),
                                on="ts", how="outer")
        # метка метрики Binance — начало 5-минутного периода; как в архиве, считаем её отметкой снимка
        frame["ts"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True).dt.as_unit("ns")
        self._metrics[symbol] = frame.sort_values("ts").reset_index(drop=True)
        fr = _get("/fapi/v1/fundingRate", symbol=symbol, limit=100)
        self._funding[symbol] = pd.DataFrame({
            "ts": pd.to_datetime([r["fundingTime"] for r in fr], unit="ms", utc=True).as_unit("ns"),
            "funding_rate": [float(r["fundingRate"]) for r in fr]})
        pi = _get("/fapi/v1/premiumIndex", symbol=symbol)
        self._info[symbol] = {"mark_price": float(pi["markPrice"]), "funding_estimate": float(pi["lastFundingRate"]),
                              "next_funding": pd.Timestamp(int(pi["nextFundingTime"]), unit="ms", tz="UTC")}

    def _run(self) -> None:
        try:
            for s in self.symbols:
                self._backfill(s)
                self._poll_metrics(s)
            self.version += 1
        except Exception as e:  # сеть может быть недоступна при старте — продолжаем пытаться
            log.exception("загрузка истории")
            self.status.ok, self.status.message = False, f"нет связи с Binance: {e}"
        last_metrics = time.monotonic()
        while True:
            try:
                for s in self.symbols:
                    if s not in self._m1:
                        self._backfill(s)
                    self._poll_klines(s)
                if time.monotonic() - last_metrics > self.metrics_seconds:
                    for s in self.symbols:
                        self._poll_metrics(s)
                    last_metrics = time.monotonic()
                self.version += 1
                self.status.ok, self.status.message, self.status.updated_at = True, "Binance онлайн", self.now()
            except Exception as e:
                log.warning("опрос Binance: %s", e)
                self.status.ok, self.status.message = False, f"нет связи с Binance: {e}"
            time.sleep(self.poll_seconds)


# ---------------- Replay ----------------

class ReplaySource(Source):
    """Проигрывание архива qlab (data/clean) с ускорением speed: 1 секунда реального времени = speed секунд рынка."""
    mode = "replay"

    def __init__(self, symbols: list[str], clean_dir: Path, start: str, speed: float = 60.0, history_days: int = 30):
        self.symbols = list(symbols)
        self.clean_dir = clean_dir
        self.speed = speed
        self.history_days = history_days
        self.t0_market = pd.Timestamp(start, tz="UTC")
        self.t0_real = time.monotonic()
        self.status = Status(ok=True, message=f"повтор истории с {self.t0_market:%d.%m.%Y}, ×{speed:g}")
        self._m1, self._metrics, self._funding = {}, {}, {}
        for s in self.symbols:
            p = clean_dir / "klines_1m" / f"{s}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"нет архива {p}: сначала `python -m qlab fetch`")
            self._m1[s] = pd.read_parquet(p)
            for ds, store in (("metrics", self._metrics), ("funding", self._funding)):
                q = clean_dir / ds / f"{s}.parquet"
                store[s] = pd.read_parquet(q) if q.exists() else None

    def now(self) -> pd.Timestamp:
        return self.t0_market + pd.Timedelta(seconds=(time.monotonic() - self.t0_real) * self.speed)

    @property
    def version(self) -> int:  # type: ignore[override]
        return int(self.now().timestamp() // 60)

    def _window(self, df: pd.DataFrame | None, lag: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame | None:
        if df is None:
            return None
        now = self.now()
        lo, hi = df["ts"].searchsorted([now - pd.Timedelta(days=self.history_days), now - lag])
        return df.iloc[lo:hi].reset_index(drop=True)

    def minutes(self, symbol: str) -> pd.DataFrame:
        return self._window(self._m1[symbol], MINUTE)        # только закрытые минуты

    def metrics(self, symbol: str) -> pd.DataFrame | None:
        return self._window(self._metrics[symbol])

    def funding(self, symbol: str) -> pd.DataFrame | None:
        return self._window(self._funding[symbol])

    def info(self, symbol: str) -> dict:
        f = self.funding(symbol)
        nxt = self.now().floor("8h") + pd.Timedelta(hours=8)
        return {"funding_estimate": float(f["funding_rate"].iloc[-1]) if f is not None and len(f) else None,
                "next_funding": nxt}

    def start(self) -> None:
        self.status.updated_at = self.now()
