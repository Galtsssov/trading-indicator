"""Feature Cache и доступ к данным.

Бары со всеми признаками считаются один раз на (монета, ТФ) и сохраняются в data/cache/.
Ключ кэша — хэш от: содержимого файлов данных, исходного кода признаков и сборщика
датасета, параметров признаков и доступности. Изменилось что-то из этого — кэш пересчитается сам.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from qlab import dataset, factory, features
from qlab.config import TF_MINUTES, Config

log = logging.getLogger(__name__)
SOURCES = [Path(dataset.__file__), *sorted(Path(features.__file__).parent.glob("*.py"))]
FACTORY_SOURCES = sorted(Path(factory.__file__).parent.glob("f_*.py")) + [Path(factory.__file__)]


class Data:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.clean = cfg.data_dir / "clean"
        self.cache_dir = cfg.data_dir / "cache"
        self._m1: dict[str, pd.DataFrame] = {}

    def table(self, ds: str, symbol: str) -> pd.DataFrame | None:
        p = self.clean / ds / f"{symbol}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def m1(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._m1:
            df = self.table("klines_1m", symbol)
            if df is None:
                raise SystemExit(f"нет свечей {symbol}: сначала `python -m qlab fetch`")
            self._m1 = {symbol: df}  # держим в памяти одну монету: минутки весят сотни МБ
        return self._m1[symbol]

    def _key(self, symbol: str, tf: str) -> str:
        h = hashlib.sha256()
        for ds in ("klines_1m", "metrics", "funding"):
            for s in sorted({symbol, self.cfg.btc_symbol}):
                p = self.clean / ds / f"{s}.parquet"
                if p.exists():
                    with p.open("rb") as fh:
                        h.update(f"{ds}/{s}:".encode() + hashlib.file_digest(fh, "sha256").digest())
        for src in SOURCES:
            h.update(src.read_bytes())
        h.update(json.dumps([self.cfg.section("features"), self.cfg.section("availability"), tf],
                            sort_keys=True).encode())
        return h.hexdigest()[:16]

    def bars(self, symbol: str, tf: str) -> pd.DataFrame:
        path = self.cache_dir / f"{symbol}_{tf}_{self._key(symbol, tf)}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        log.info("кэш признаков: считаю %s %s", symbol, tf)
        av = self.cfg.section("availability")
        btc = self.m1(self.cfg.btc_symbol) if symbol != self.cfg.btc_symbol else None
        m1 = self.m1(symbol)
        bars = dataset.build(m1, tf, oi=self.table("metrics", symbol), funding=self.table("funding", symbol),
                             btc_m1=btc if btc is not None else m1, oi_lag_minutes=av["oi_lag_minutes"],
                             oi_max_staleness_minutes=av["oi_max_staleness_minutes"])
        bars = features.compute(bars, TF_MINUTES[tf], self.cfg.section("features"))
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in path.parent.glob(f"{symbol}_{tf}_*.parquet"):
            old.unlink()
        bars.to_parquet(path, index=False)
        return bars

    def factory(self, symbol: str, tf: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Признаки Feature Factory (float32), строки выровнены с bars(symbol, tf); есть колонка ts.
        Считаются один раз; ключ — ключи баров всех монет + исходный код фабрики."""
        h = hashlib.sha256()
        for s in self.cfg.symbols:
            h.update(self._key(s, tf).encode())
        for src in FACTORY_SOURCES:
            h.update(src.read_bytes())
        path = self.cache_dir / f"factory_{symbol}_{tf}_{h.hexdigest()[:16]}.parquet"
        if not path.exists():
            log.info("Feature Factory: считаю %s %s", symbol, tf)
            bars = self.bars(symbol, tf)
            others = {s: self.bars(s, tf).set_index("ts")[["close", "volume"]] for s in self.cfg.symbols}
            out = factory.compute(bars, symbol, TF_MINUTES[tf], others)
            out.insert(0, "ts", bars["ts"].to_numpy())
            for old in path.parent.glob(f"factory_{symbol}_{tf}_*.parquet"):
                old.unlink()
            out.to_parquet(path, index=False)
        return pd.read_parquet(path, columns=None if columns is None else ["ts", *columns])
