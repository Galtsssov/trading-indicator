"""Event Library.

Событие = функция в отдельном файле этого пакета с декоратором `@event`. Новый файл подключается сам.

Контракт:
- функция `(bars, params) -> DataFrame` с индексом-подмножеством строк `bars` и колонкой `dir`
  (+1 / −1) плюс любые доп. колонки;
- `dir` — направление движения, которое определяет событие (свип вверх, пробой вверх = +1);
  для ненаправленных событий — знак свечи события. Гипотеза затем выбирает `with` или `against`;
- только прошлые данные: значение в строке t зависит от строк ≤ t (тест причинности).
Ядро само отбрасывает неполные бары, применяет cooldown (по каждому направлению) и добавляет
ts, available_at, atr_ref (ATR предыдущего бара — единица риска 1R).
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EventFn = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]


@dataclass(frozen=True)
class EventType:
    name: str
    fn: EventFn
    requires: tuple[str, ...]
    defaults: Mapping[str, Any]
    description: str


REGISTRY: dict[str, EventType] = {}


def event(name: str, *, requires: tuple[str, ...], defaults: Mapping[str, Any] | None = None,
          description: str = "") -> Callable[[EventFn], EventFn]:
    def wrap(fn: EventFn) -> EventFn:
        if name in REGISTRY:
            raise ValueError(f"событие {name!r} уже зарегистрировано")
        REGISTRY[name] = EventType(name, fn, requires, {"cooldown": 12, **(defaults or {})}, description)
        return fn
    return wrap


def load_all() -> dict[str, EventType]:
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    return REGISTRY


def bars_per_day(bars: pd.DataFrame) -> int:
    step = bars["ts"].iloc[1] - bars["ts"].iloc[0]
    return int(pd.Timedelta(days=1) / step)


def _cooldown(idx: np.ndarray, bars: int) -> list[int]:
    keep, last = [], -10**12
    for i in idx:
        if i - last > bars:
            keep.append(int(i))
            last = i
    return keep


def params_for(name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reg = load_all()
    if name not in reg:
        raise ValueError(f"неизвестное событие {name!r}; есть: {sorted(reg)}")
    et = reg[name]
    unknown = set(params or {}) - set(et.defaults)
    if unknown:
        raise ValueError(f"{name}: неизвестные параметры {sorted(unknown)}; допустимы {sorted(et.defaults)}")
    return {**et.defaults, **(params or {})}


def detect(name: str, bars: pd.DataFrame, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """События типа `name` на барах. Индекс `bars` должен быть 0..n−1."""
    p = params_for(name, params)
    et = REGISTRY[name]
    missing = set(et.requires) - set(bars.columns)
    if missing:
        raise ValueError(f"{name}: в данных нет колонок {sorted(missing)}")
    raw = et.fn(bars, p)
    raw = raw[raw["dir"].isin([1, -1])]
    if "complete" in bars:
        raw = raw[bars["complete"].to_numpy()[raw.index.to_numpy()]]
    keep: list[int] = []
    for d in (1, -1):
        keep += _cooldown(raw.index[raw["dir"] == d].to_numpy(), int(p["cooldown"]))
    ev = raw.loc[sorted(keep)].copy()
    ev["dir"] = ev["dir"].astype("int64")
    idx = ev.index.to_numpy()
    ev.insert(0, "bar", idx)
    ev.insert(1, "event", name)
    ev["ts"] = bars["ts"].loc[idx]                    # индекс ev = номера баров → выравнивание по меткам
    ev["available_at"] = bars["available_at"].loc[idx]
    ev["atr_ref"] = bars["atr"].shift(1).loc[idx]
    return ev.reset_index(drop=True)


def sign_of_bar(bars: pd.DataFrame) -> np.ndarray:
    return np.sign(bars["close"] - bars["open"]).to_numpy()


def frame(bars: pd.DataFrame, mask: np.ndarray | pd.Series, direction: np.ndarray | pd.Series,
          **extra: Any) -> pd.DataFrame:
    """Помощник для модулей событий: строки, где mask, с колонкой dir и доп. колонками."""
    mask = np.asarray(mask, dtype=bool)
    out = pd.DataFrame({"dir": np.asarray(direction)}, index=bars.index)
    for k, v in extra.items():
        out[k] = np.asarray(v)
    return out[mask]
