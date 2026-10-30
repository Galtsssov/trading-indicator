"""Feature Engine.

Признак = функция в отдельном файле этого пакета, помеченная декоратором `@feature`.
Чтобы добавить признак, достаточно создать файл: пакет найдёт его сам (`load_all`).

Контракт признака:
- `requires` — колонки, которые нужны на входе (сырые или выходы других признаков);
- `provides` — колонки, которые признак добавляет;
- `defaults` — параметры; переопределяются секцией `[features.<name>]` конфига;
- функция `(df, ctx) -> DataFrame` возвращает ТОЛЬКО новые колонки и обязана быть причинной:
  значение в строке t зависит только от строк ≤ t (проверяется тестом причинности).
  Значение считается известным в `available_at` строки (закрытие бара).
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Context:
    tf_minutes: int
    params: Mapping[str, Any]

    @property
    def bars_per_day(self) -> int:
        return 1440 // self.tf_minutes


FeatureFn = Callable[[pd.DataFrame, Context], pd.DataFrame]


@dataclass(frozen=True)
class Feature:
    name: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    fn: FeatureFn
    defaults: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""


REGISTRY: dict[str, Feature] = {}


def feature(name: str, *, requires: tuple[str, ...], provides: tuple[str, ...],
            defaults: Mapping[str, Any] | None = None, description: str = "") -> Callable[[FeatureFn], FeatureFn]:
    def wrap(fn: FeatureFn) -> FeatureFn:
        if name in REGISTRY:
            raise ValueError(f"признак {name!r} уже зарегистрирован")
        REGISTRY[name] = Feature(name, requires, provides, fn, dict(defaults or {}), description)
        return fn
    return wrap


def load_all() -> dict[str, Feature]:
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    return REGISTRY


def order(features: list[Feature], available: set[str]) -> list[Feature]:
    """Топологический порядок по requires/provides; ошибка, если зависимость не удовлетворить."""
    done, out, pending = set(available), [], list(features)
    while pending:
        ready = [f for f in pending if set(f.requires) <= done]
        if not ready:
            missing = {f.name: sorted(set(f.requires) - done) for f in pending}
            raise ValueError(f"неудовлетворённые зависимости признаков: {missing}")
        for f in ready:
            out.append(f)
            done |= set(f.provides)
            pending.remove(f)
    return out


def compute(df: pd.DataFrame, tf_minutes: int, config: Mapping[str, Mapping[str, Any]] | None = None,
            names: list[str] | None = None) -> pd.DataFrame:
    """Посчитать признаки (все или `names`) и вернуть df с новыми колонками."""
    reg = load_all()
    selected = [reg[n] for n in (names or reg)]
    config = config or {}
    out = df.copy()
    for f in order(selected, set(out.columns)):
        ctx = Context(tf_minutes, {**f.defaults, **config.get(f.name, {})})
        new = f.fn(out, ctx)
        if set(new.columns) != set(f.provides):
            raise ValueError(f"{f.name}: вернул {sorted(new.columns)}, а заявлено {sorted(f.provides)}")
        out[list(new.columns)] = new
    return out
