"""Загрузка конфигурации исследования (TOML) и её хэш для журнала."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        with path.open("rb") as f:
            return cls(tomllib.load(f), path)

    def section(self, *keys: str) -> dict[str, Any]:
        node: Any = self.raw
        for k in keys:
            node = node.get(k, {})
        return dict(node)

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["data"]["dir"])

    @property
    def symbols(self) -> list[str]:
        return list(self.raw["data"]["symbols"])

    @property
    def btc_symbol(self) -> str:
        return self.raw["data"]["btc_symbol"]

    @property
    def timeframes(self) -> list[str]:
        return list(self.raw["dataset"]["timeframes"])

    @property
    def cost_pct_round_trip(self) -> float:
        c = self.raw["costs"]
        return 2 * (c["taker_fee"] + c["slippage"])

    def hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
