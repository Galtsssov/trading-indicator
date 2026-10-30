"""Проверка качества данных: покрытие, пропуски, дубли, некорректные свечи, выбросы."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

EXPECTED_STEP = {"klines_1m": pd.Timedelta(minutes=1), "metrics": pd.Timedelta(minutes=5)}


@dataclass
class QualityReport:
    dataset: str
    symbol: str
    rows: int
    first: str | None
    last: str | None
    expected_rows: int | None = None
    coverage_pct: float | None = None
    missing_rows: int | None = None
    duplicates_dropped: int = 0
    largest_gaps: list[tuple[str, str]] = field(default_factory=list)  # (начало, длительность)
    issues: dict[str, int] = field(default_factory=dict)
    verdict: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _gaps(ts: pd.Series, step: pd.Timedelta, top: int = 5) -> tuple[int, list[tuple[str, str]]]:
    d = ts.diff()
    gaps = d[d > step]
    missing = int(((gaps / step) - 1).round().sum())
    biggest = gaps.sort_values(ascending=False).head(top)
    return missing, [(str(ts[i - 1]), str(g)) for i, g in biggest.items()]


def check(dataset: str, symbol: str, df: pd.DataFrame | None, duplicates: int = 0) -> QualityReport:
    if df is None or df.empty:
        return QualityReport(dataset, symbol, 0, None, None, verdict="НЕТ ДАННЫХ")
    ts = df["ts"]
    r = QualityReport(dataset, symbol, len(df), str(ts.iloc[0]), str(ts.iloc[-1]), duplicates_dropped=duplicates)
    issues: dict[str, int] = {}

    step = EXPECTED_STEP.get(dataset)
    if step is not None:
        r.expected_rows = int((ts.iloc[-1] - ts.iloc[0]) / step) + 1
        r.missing_rows, r.largest_gaps = _gaps(ts, step)
        r.coverage_pct = round(100 * len(df) / r.expected_rows, 3)

    if dataset == "klines_1m":
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        issues["ohlc_inconsistent"] = int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l <= 0)).sum())
        issues["zero_volume_bars"] = int((df["volume"] == 0).sum())
        ret = np.abs(np.diff(np.log(c)))
        issues["abs_1m_return_gt_10pct"] = int((ret > np.log(1.10)).sum())
        issues["taker_buy_gt_volume"] = int((df["taker_buy_volume"] > df["volume"] * (1 + 1e-9)).sum())
    elif dataset == "metrics":
        issues["oi_nonpositive_or_nan"] = int((~(df["oi"] > 0)).sum())
        jump = np.abs(np.diff(np.log(df["oi"].clip(lower=1e-12).to_numpy())))
        issues["oi_5m_jump_gt_20pct"] = int((jump > np.log(1.2)).sum())
    elif dataset == "funding":
        issues["funding_nan"] = int(df["funding_rate"].isna().sum())
        issues["funding_abs_gt_1pct"] = int((df["funding_rate"].abs() > 0.01).sum())
        intervals = ts.diff().dropna().dt.total_seconds() / 3600
        for hours, n in intervals.round().value_counts().items():
            issues[f"interval_{int(hours)}h"] = int(n)
    r.issues = issues

    bad = sum(v for k, v in issues.items() if not k.startswith("interval_"))
    if r.coverage_pct is not None and r.coverage_pct < 99.0:
        r.verdict = "ОСТОРОЖНО: покрытие < 99%"
    elif bad > 0.001 * len(df):
        r.verdict = "ОСТОРОЖНО: аномалий > 0.1% строк"
    else:
        r.verdict = "ГОДНО"
    return r
