"""Загрузка публичного архива Binance USDⓈ-M futures (data.binance.vision) и разбор в Parquet.

Наборы:
- klines_1m  — свечи 1m (monthly за полные месяцы, daily за текущий);
- metrics    — открытый интерес и пр. (daily, снимки раз в 5 минут);
- funding    — история ставок funding (monthly).

Сырые zip-файлы неизменяемы и проверяются по sha256 из .CHECKSUM.
Отсутствующий файл (404) запоминается маркером, чтобы не запрашивать его снова.
"""
from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


@dataclass(frozen=True)
class RemoteFile:
    dataset: str
    symbol: str
    url: str
    path: Path


def _months(start: date, end: date) -> list[str]:
    out, d = [], date(start.year, start.month, 1)
    while d <= end:
        out.append(f"{d:%Y-%m}")
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out


def _days(start: date, end: date) -> list[str]:
    return [f"{start + timedelta(n):%Y-%m-%d}" for n in range((end - start).days + 1)]


def plan(base_url: str, raw_dir: Path, symbol: str, start: str, today: date) -> list[RemoteFile]:
    """Список файлов для скачивания: полные месяцы — monthly, текущий месяц — daily."""
    first = date.fromisoformat(start + "-01")
    month_start = date(today.year, today.month, 1)
    last_full_month = month_start - timedelta(days=1)
    yesterday = today - timedelta(days=1)
    files: list[RemoteFile] = []

    def add(dataset: str, rel: str) -> None:
        files.append(RemoteFile(dataset, symbol, f"{base_url}/{rel}", raw_dir / dataset / symbol / Path(rel).name))

    for m in _months(first, last_full_month):
        add("klines_1m", f"monthly/klines/{symbol}/1m/{symbol}-1m-{m}.zip")
        add("funding", f"monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip")
    for d in _days(month_start, yesterday):
        add("klines_1m", f"daily/klines/{symbol}/1m/{symbol}-1m-{d}.zip")
    for d in _days(first, yesterday):
        add("metrics", f"daily/metrics/{symbol}/{symbol}-metrics-{d}.zip")
    return files


def _get(url: str, retries: int = 4) -> bytes | None:
    """Скачать URL; None, если 404. Сетевые ошибки — повтор с экспоненциальной паузой."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err: Exception = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        if attempt < retries:
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"не удалось скачать {url}: {err}")


def _fetch_one(f: RemoteFile) -> str:
    missing = f.path.with_suffix(".missing")
    if f.path.exists():
        return "cached"
    if missing.exists():
        return "missing"
    body = _get(f.url)
    f.path.parent.mkdir(parents=True, exist_ok=True)
    if body is None:
        missing.touch()
        return "missing"
    checksum = _get(f.url + ".CHECKSUM")
    if checksum is not None:
        expected = checksum.decode().split()[0]
        if hashlib.sha256(body).hexdigest() != expected:
            raise RuntimeError(f"sha256 не совпал: {f.url}")
    tmp = f.path.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.rename(f.path)
    return "downloaded"


def fetch(files: list[RemoteFile], workers: int) -> dict[str, int]:
    stats = {"downloaded": 0, "cached": 0, "missing": 0}
    with ThreadPoolExecutor(workers) as pool:
        for status in pool.map(_fetch_one, files):
            stats[status] += 1
    return stats


# ---------- разбор ----------

def _has_header(path: Path) -> bool:
    first = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
    return not str(first).lstrip("-").isdigit()


def epoch_to_utc(values: pd.Series) -> pd.Series:
    """Эпоха в мс или мкс (Binance перешла на мкс в части архивов) → datetime UTC (ns)."""
    v = values.astype("int64")
    unit = "us" if v.iloc[0] > 10**14 else "ms"
    return pd.to_datetime(v, unit=unit, utc=True).dt.as_unit("ns")


def parse_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0 if _has_header(path) else None)
    df.columns = KLINE_COLS
    out = pd.DataFrame({"ts": epoch_to_utc(df["open_time"])})
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
        out[c] = df[c].astype("float64")
    out["trades"] = df["trades"].astype("int64")
    return out


def parse_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "ts": pd.to_datetime(df["create_time"], utc=True).dt.as_unit("ns"),
        "oi": df["sum_open_interest"].astype("float64"),
        "oi_value": df["sum_open_interest_value"].astype("float64"),
        "ls_ratio_accounts": pd.to_numeric(df.get("count_long_short_ratio"), errors="coerce"),
        "taker_ls_vol_ratio": pd.to_numeric(df.get("sum_taker_long_short_vol_ratio"), errors="coerce"),
    })


def parse_funding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "ts": epoch_to_utc(df["calc_time"]),
        "funding_rate": df["last_funding_rate"].astype("float64"),
    })


PARSERS = {"klines_1m": parse_klines, "metrics": parse_metrics, "funding": parse_funding}


def build_clean(raw_dir: Path, clean_dir: Path, dataset: str, symbol: str) -> int | None:
    """Склеить все zip набора в один Parquet: сортировка по ts, без дублей. Возвращает число дублей."""
    zips = sorted((raw_dir / dataset / symbol).glob("*.zip"))
    if not zips:
        log.warning("%s %s: нет файлов", dataset, symbol)
        return None
    df = pd.concat([PARSERS[dataset](z) for z in zips], ignore_index=True)
    n_raw = len(df)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    out = clean_dir / dataset / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("%s %s: %d строк (%d дублей удалено)", dataset, symbol, len(df), n_raw - len(df))
    return n_raw - len(df)


def raw_manifest_hash(raw_dir: Path) -> str:
    """Версия сырых данных: хэш списка файлов и их размеров."""
    h = hashlib.sha256()
    for p in sorted(raw_dir.rglob("*.zip")):
        h.update(f"{p.relative_to(raw_dir)}:{p.stat().st_size}".encode())
    return h.hexdigest()[:12]
