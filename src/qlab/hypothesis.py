"""Гипотеза = YAML-файл. Ядро понимает её без изменения кода.

Отпечаток (fingerprint) — хэш смысловой части: событие, направление, фильтры, рынки, ТФ, исход.
Имя и описание в него не входят, поэтому ту же идею нельзя зарегистрировать под другим названием.
Замок (lock) — всё содержимое, кроме названия и описания: после регистрации менять нельзя.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qlab import events, expr

DIRECTIONS = ("with", "against", "long", "short")
REQUIRED = ("id", "name", "description", "source", "created", "rationale", "event", "direction",
            "markets", "timeframes", "outcome", "features_used", "confirmation", "refutation", "criteria")
CRITERIA_KEYS = {"split", "min_events", "alpha", "practical_min_delta_r", "require_markets",
                 "require_positive_net", "beat_baselines"}


@dataclass(frozen=True)
class Hypothesis:
    raw: dict[str, Any]
    path: Path
    filters: tuple[str, ...] = field(default=())

    @classmethod
    def load(cls, path: str | Path) -> Hypothesis:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in raw]
        if missing:
            raise ValueError(f"{path}: нет обязательных полей {missing}")
        h = cls(raw, path, tuple(raw.get("filters") or ()))
        h._validate()
        return h

    def _validate(self) -> None:
        r = self.raw
        if r["direction"] not in DIRECTIONS:
            raise ValueError(f"{self.id}: direction должен быть одним из {DIRECTIONS}")
        events.params_for(r["event"]["type"], r["event"].get("params"))
        for f in self.filters:
            expr.names(f)  # синтаксис проверяется сразу; неизвестные колонки — при запуске
        o = r["outcome"]
        if set(o) - {"r_atr", "target_r", "horizon_bars"}:
            raise ValueError(f"{self.id}: outcome поддерживает только r_atr, target_r, horizon_bars")
        unknown = set(r["criteria"]) - CRITERIA_KEYS
        if unknown:
            raise ValueError(f"{self.id}: неизвестные критерии {sorted(unknown)}")

    def __getattr__(self, key: str) -> Any:
        try:
            return self.raw[key]
        except KeyError:
            raise AttributeError(key) from None

    @property
    def event_params(self) -> dict[str, Any]:
        return events.params_for(self.raw["event"]["type"], self.raw["event"].get("params"))

    def spec(self) -> dict[str, Any]:
        return {
            "event": {"type": self.raw["event"]["type"], "params": self.event_params},
            "direction": self.raw["direction"],
            "filters": sorted(" ".join(f.split()) for f in self.filters),
            "markets": sorted(self.raw["markets"]),
            "timeframes": sorted(self.raw["timeframes"]),
            "outcome": dict(sorted(self.raw["outcome"].items())),
        }

    def fingerprint(self) -> str:
        return _hash(self.spec())

    def lock(self) -> str:
        """Всё, кроме названия и описания: условия, критерии, обоснование, прогноз, clean_after."""
        return _hash({"spec": self.spec(), **{k: v for k, v in self.raw.items() if k not in ("name", "description")}})


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:16]
