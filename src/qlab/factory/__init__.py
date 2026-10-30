"""Feature Factory: сотни объективных признаков для каждого бара и, значит, для каждого события.

Признак = спецификация `Spec` в одном из модулей-семейств `f_*.py` этого пакета. Семейство регистрирует
признаки функцией `family(...)` по сетке параметров (обычно окна W). Новое семейство — новый файл.

Контракт каждого признака:
- чёткая формула (поле `formula`) и функция `fn(ctx) -> Series`;
- только прошлое: значение в баре t зависит от баров ≤ t (тест причинности гоняется по ВСЕМ признакам);
- значение известно на закрытии бара t (`available_at`);
- поля каталога: категория, тип, единица, зависимости, стоимость, механизм, что измеряет,
  чем полезен, ограничения. Каталог docs/FEATURES.md генерируется из этих полей.

Обозначения в формулах: o,h,l,c,v — OHLCV бара; r = ln(c_t / c_{t−1}); a = ATR(14) предыдущего бара;
W — окно в барах; Σ_W, mean_W, std_W, max_W — по барам t−W+1..t; ctx(X) — тот же бар монеты X.
"""
from __future__ import annotations

import importlib
import itertools
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

WIN = (6, 24, 96, 288)        # окна в барах: на 5m — 30 мин, 2 ч, 8 ч, 1 сутки; на 15m — в 3 раза длиннее
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
KINDS = ("непрерывный", "бинарный", "категориальный", "циклический", "счётчик")


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    category: str
    formula: str
    kind: str
    unit: str
    deps: tuple[str, ...]
    cost: str
    mechanism: str
    measures: str
    why: str
    limits: str
    fn: Callable[[Ctx], pd.Series] = field(repr=False, compare=False)


REGISTRY: dict[str, Spec] = {}
_counters: dict[str, itertools.count] = {}


def family(category: str, code: str, name: str, params: Iterable[dict[str, Any]] | None, *, formula: str,
           kind: str, unit: str, deps: Iterable[str], cost: str, mechanism: str, measures: str, why: str,
           limits: str, fn: Callable[..., pd.Series]) -> None:
    """Зарегистрировать семейство признаков: по одному на каждый набор параметров.
    В name, formula и текстах можно использовать {параметры}."""
    if kind not in KINDS:
        raise ValueError(f"{name}: тип {kind!r} не из {KINDS}")
    for p in (list(params) if params else [{}]):
        n = name.format(**p)
        if n in REGISTRY:
            raise ValueError(f"признак {n!r} уже зарегистрирован")
        num = next(_counters.setdefault(code, itertools.count(1)))
        REGISTRY[n] = Spec(f"{code}{num:03d}", n, category, formula.format(**p), kind, unit.format(**p),
                           tuple(deps), cost.format(**p), mechanism.format(**p), measures.format(**p),
                           why.format(**p), limits.format(**p), (lambda ctx, p=p: fn(ctx, **p)))


def grid(**axes: Iterable[Any]) -> list[dict[str, Any]]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


def load_all() -> dict[str, Spec]:
    for mod in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if mod.name.startswith("f_"):
            importlib.import_module(f"{__name__}.{mod.name}")
    return REGISTRY


# ---------- контекст расчёта: общие промежуточные ряды считаются один раз ----------

class Ctx:
    """Бары одной монеты (с базовыми признаками) + бары других монет того же ТФ, выровненные по ts."""

    def __init__(self, bars: pd.DataFrame, symbol: str, tf_minutes: int,
                 others: dict[str, pd.DataFrame] | None = None):
        self.b = bars
        self.symbol = symbol
        self.tf_minutes = tf_minutes
        self.bpd = 1440 // tf_minutes
        self._others = others or {}
        self._memo: dict[Any, pd.Series] = {}

    def col(self, name: str) -> pd.Series:
        return self.b[name].astype(float)

    def memo(self, key: Any, make: Callable[[], pd.Series]) -> pd.Series:
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    @cached_property
    def o(self) -> pd.Series: return self.col("open")
    @cached_property
    def h(self) -> pd.Series: return self.col("high")
    @cached_property
    def l(self) -> pd.Series: return self.col("low")
    @cached_property
    def c(self) -> pd.Series: return self.col("close")
    @cached_property
    def v(self) -> pd.Series: return self.col("volume")
    @cached_property
    def r(self) -> pd.Series: return np.log(self.c).diff()
    @cached_property
    def a(self) -> pd.Series: return self.col("atr").shift(1)
    @cached_property
    def a_pct(self) -> pd.Series: return self.a / self.c.shift(1)
    @cached_property
    def rng(self) -> pd.Series: return self.h - self.l
    @cached_property
    def ts(self) -> pd.Series: return self.b["ts"]
    @cached_property
    def day(self) -> pd.Series: return self.ts.dt.floor("D")
    @cached_property
    def week(self) -> pd.Series: return self.day - pd.to_timedelta(self.ts.dt.dayofweek, unit="D")

    def hmax(self, w: int) -> pd.Series:
        return self.memo(("hmax", w), lambda: self.h.rolling(w, min_periods=w).max())

    def lmin(self, w: int) -> pd.Series:
        return self.memo(("lmin", w), lambda: self.l.rolling(w, min_periods=w).min())

    def rv(self, w: int) -> pd.Series:
        return self.memo(("rv", w), lambda: self.r.rolling(w, min_periods=w).std())

    def ret(self, w: int) -> pd.Series:
        return self.memo(("ret", w), lambda: np.log(self.c / self.c.shift(w)))

    def other(self, symbol: str) -> pd.DataFrame | None:
        """Бары другой монеты по тем же ts (close, volume) или None, если это та же монета / нет данных."""
        if symbol == self.symbol or symbol not in self._others:
            return None
        return self.memo(("other", symbol), lambda: self._others[symbol].reindex(self.ts.to_numpy())
                         .set_axis(self.b.index))


def nan_series(ctx: Ctx) -> pd.Series:
    return pd.Series(np.nan, index=ctx.b.index)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b != 0)


def rolling_pctl(x: pd.Series, w: int) -> pd.Series:
    """Процентиль текущего значения среди последних w (включая текущее), 0..1."""
    return x.rolling(w, min_periods=max(w // 4, 2)).rank(pct=True)


def rolling_z(x: pd.Series, w: int) -> pd.Series:
    m, s = x.rolling(w, min_periods=w).mean(), x.rolling(w, min_periods=w).std()
    return safe_div(x - m, s)


def bars_since(mask: pd.Series) -> pd.Series:
    """Сколько баров прошло с последнего True (0 — в этом баре); NaN, если ещё не было."""
    idx = pd.Series(np.where(mask.to_numpy(dtype=bool), np.arange(len(mask)), np.nan), index=mask.index).ffill()
    return pd.Series(np.arange(len(mask)), index=mask.index) - idx


def streak(mask: pd.Series) -> pd.Series:
    """Длина текущей серии подряд идущих True."""
    m = mask.fillna(False).astype(bool)
    grp = (~m).cumsum()
    return m.astype(int).groupby(grp).cumsum().astype(float)


def rolling_argext_age(x: pd.Series, w: int, use_max: bool) -> pd.Series:
    """Сколько баров назад был экстремум окна (0 — текущий бар), по кускам, чтобы не раздувать память."""
    from numpy.lib.stride_tricks import sliding_window_view
    arr = x.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) < w:
        return pd.Series(out, index=x.index)
    win = sliding_window_view(arr, w)
    for s in range(0, len(win), 50_000):
        chunk = win[s:s + 50_000]
        filled = np.where(np.isnan(chunk), -np.inf if use_max else np.inf, chunk)
        pos = filled.argmax(axis=1) if use_max else filled.argmin(axis=1)
        out[w - 1 + s: w - 1 + s + len(chunk)] = (w - 1) - pos
    return pd.Series(out, index=x.index)


def compute(bars: pd.DataFrame, symbol: str, tf_minutes: int, others: dict[str, pd.DataFrame] | None = None,
            names: Iterable[str] | None = None) -> pd.DataFrame:
    """Все признаки фабрики (или выбранные) как DataFrame float32 с индексом bars."""
    reg = load_all()
    ctx = Ctx(bars, symbol, tf_minutes, others)
    out = {}
    for n in (names or reg):
        s = np.asarray(reg[n].fn(ctx), dtype=np.float64)
        out[n] = np.where(np.isfinite(s), s, np.nan).astype(np.float32)   # сразу float32: меньше пиковая память
    return pd.DataFrame(out, index=bars.index)


def catalog() -> pd.DataFrame:
    rows = [{"id": s.id, "name": s.name, "category": s.category, "formula": s.formula, "kind": s.kind,
             "unit": s.unit, "deps": ", ".join(s.deps), "cost": s.cost, "mechanism": s.mechanism,
             "measures": s.measures, "why": s.why, "limits": s.limits} for s in load_all().values()]
    return pd.DataFrame(rows)
