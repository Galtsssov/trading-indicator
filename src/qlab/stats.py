"""Статистика сравнения групп: метрики, бутстреп по дням, тесты, размер эффекта, поправка BH.

Почему так:
- исходы барьеров дискретны (−1, +k) с примесью выходов по времени → распределение заведомо
  не нормальное. Тест выбирается автоматически: если нормальность отвергнута (D'Agostino–Pearson),
  основным тестом разницы ожиданий служит перестановочный тест (без допущений о форме);
  t-тест Уэлча — только справочно. Различие распределений целиком — Манн–Уитни и Колмогоров–Смирнов;
- соседние события одного дня зависимы → доверительные интервалы через блочный бутстреп по дням;
- p-value не главное: решение о практической значимости — по интервалу разницы ожиданий
  и минимальному практически значимому порогу в R.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as st


@dataclass
class GroupStats:
    n: int
    n_days: int
    win_rate: float
    expectancy: float
    expectancy_ci: tuple[float, float]
    median: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    mfe_mean: float
    mae_mean: float
    mfe_median: float
    mae_median: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comparison:
    delta_expectancy: float           # B − A
    delta_ci: tuple[float, float]
    delta_win_rate: float
    p_primary: float
    primary_test: str
    test_rationale: str
    p_permutation: float
    p_welch: float
    p_mannwhitney: float
    p_ks: float
    p_win_rate: float
    cliffs_delta: float
    cliffs_magnitude: str
    cohens_d: float
    p_adjusted: float = float("nan")  # заполняется поправкой BH по всем сравнениям

    def as_dict(self) -> dict:
        return asdict(self)


def day_bootstrap_dist(values: list[np.ndarray], days: list[np.ndarray], fn, b: int, rng) -> np.ndarray:
    """Блочный бутстреп по дням: выбираем дни с возвращением, берём все события этих дней во всех группах.
    Возвращает распределение статистики fn(список средних по группам)."""
    all_days = np.unique(np.concatenate(days))
    sums, cnts = [], []
    for v, d in zip(values, days):
        k = np.searchsorted(all_days, d)
        sums.append(np.bincount(k, weights=v, minlength=len(all_days)))
        cnts.append(np.bincount(k, minlength=len(all_days)).astype(float))
    draw = rng.integers(0, len(all_days), size=(b, len(all_days)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(fn([s[draw].sum(1) / c[draw].sum(1) for s, c in zip(sums, cnts)]))


def _day_bootstrap(values: list[np.ndarray], days: list[np.ndarray], fn, b: int, rng) -> tuple[float, float]:
    lo, hi = np.nanpercentile(day_bootstrap_dist(values, days, fn, b, rng), [2.5, 97.5])
    return float(lo), float(hi)


def summarize(r: np.ndarray, mfe: np.ndarray, mae: np.ndarray, days: np.ndarray, b: int, rng) -> GroupStats:
    if len(r) == 0:
        nan = float("nan")
        return GroupStats(0, 0, nan, nan, (nan, nan), nan, nan, nan, nan, nan, nan, nan, nan)
    wins, losses = r[r > 0], r[r < 0]
    return GroupStats(
        n=len(r), n_days=len(np.unique(days)),
        win_rate=float((r > 0).mean()),
        expectancy=float(r.mean()),
        expectancy_ci=_day_bootstrap([r], [days], lambda m: m[0], b, rng),
        median=float(np.median(r)),
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        avg_win=float(wins.mean()) if len(wins) else float("nan"),
        avg_loss=float(losses.mean()) if len(losses) else float("nan"),
        mfe_mean=float(mfe.mean()), mae_mean=float(mae.mean()),
        mfe_median=float(np.median(mfe)), mae_median=float(np.median(mae)),
    )


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int, rng) -> float:
    """Двусторонний перестановочный тест разницы средних (mean(b) − mean(a))."""
    x = np.concatenate([a, b])
    total, nb, na = x.sum(), len(b), len(a)
    observed = b.mean() - a.mean()
    hits = 0
    for _ in range(n_perm):
        sb = x[rng.permutation(len(x))[:nb]].sum()
        if abs(sb / nb - (total - sb) / na) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(B > A) − P(B < A), через U-статистику Манна–Уитни (с учётом совпадений)."""
    u = st.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def cliffs_magnitude(d: float) -> str:
    a = abs(d)  # пороги Romano et al. (2006)
    return "пренебрежимый" if a < 0.147 else "малый" if a < 0.33 else "средний" if a < 0.474 else "большой"


def compare(a: np.ndarray, b: np.ndarray, a_days: np.ndarray, b_days: np.ndarray,
            n_boot: int, n_perm: int, rng) -> Comparison | None:
    if len(a) < 2 or len(b) < 2:
        return None
    normal = all(st.normaltest(x).pvalue > 0.05 for x in (a, b) if len(x) >= 20)
    p_perm = permutation_p(a, b, n_perm, rng)
    p_welch = float(st.ttest_ind(b, a, equal_var=False).pvalue)
    if normal:
        primary, p_primary = "Welch t-test", p_welch
        rationale = "нормальность не отвергнута в обеих группах → t-тест Уэлча"
    else:
        primary, p_primary = "permutation (разница средних)", p_perm
        rationale = "нормальность отвергнута (D'Agostino–Pearson) → перестановочный тест без допущений о форме"
    wa, wb = int((a > 0).sum()), int((b > 0).sum())
    table = np.array([[wa, len(a) - wa], [wb, len(b) - wb]])
    if (st.contingency.expected_freq(table) < 5).any():
        p_wr = float(st.fisher_exact(table).pvalue)
    else:
        p_wr = float(st.chi2_contingency(table).pvalue)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    cd = cliffs_delta(a, b)
    return Comparison(
        delta_expectancy=float(b.mean() - a.mean()),
        delta_ci=_day_bootstrap([a, b], [a_days, b_days], lambda m: m[1] - m[0], n_boot, rng),
        delta_win_rate=float((b > 0).mean() - (a > 0).mean()),
        p_primary=p_primary, primary_test=primary, test_rationale=rationale,
        p_permutation=p_perm, p_welch=p_welch,
        p_mannwhitney=float(st.mannwhitneyu(b, a, alternative="two-sided").pvalue),
        p_ks=float(st.ks_2samp(a, b, method="asymp").pvalue),
        p_win_rate=p_wr,
        cliffs_delta=cd, cliffs_magnitude=cliffs_magnitude(cd),
        cohens_d=float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan"),
    )


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Поправка Бенджамини–Хохберга (FDR). NaN остаются NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    if len(q) == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * len(q) / np.arange(1, len(q) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out
