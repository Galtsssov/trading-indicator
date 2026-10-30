"""Анализ Feature Factory: корреляции, удаление дублей, кластеры. Только training.

1. Для каждой пары (монета, ТФ) — ранговая корреляция Спирмена между всеми признаками на выборке баров
   training (пропуски после ранжирования заменяются средним рангом — небольшое занижение |ρ|).
2. Консенсус: среднее |ρ| по всем парам, где оба признака определены.
3. Вырожденные признаки (константа или > 50% пропусков во всех парах) исключаются.
4. Дубли: признаки упорядочены по стоимости и номеру в каталоге; признак удаляется, если |ρ| ≥ порога
   с уже оставленным. Остаётся более дешёвый и более ранний.
5. Кластеры: иерархическая кластеризация (средняя связь) по расстоянию 1 − |ρ|, разрез на 1 − cluster_rho.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from qlab import md, splits  # noqa: E402
from qlab.factory import catalog  # noqa: E402

log = logging.getLogger(__name__)
COST_RANK = {"O(1)": 0, "O(n)": 1, "O(n log W)": 2, "O(20n)": 3}


def spearman(frame: pd.DataFrame, max_nan: float = 0.5) -> tuple[pd.DataFrame, list[str]]:
    """Матрица ρ Спирмена и список вырожденных колонок (константа или много пропусков)."""
    nan_share = frame.isna().mean()
    degenerate = [c for c in frame.columns if nan_share[c] > max_nan or frame[c].nunique(dropna=True) <= 1]
    f = frame.drop(columns=degenerate)
    r = f.rank()
    z = (r - r.mean()) / r.std(ddof=0)
    z = z.fillna(0.0).to_numpy(dtype=np.float64)
    rho = z.T @ z / len(z)
    return pd.DataFrame(rho, index=f.columns, columns=f.columns), degenerate


def consensus(mats: list[pd.DataFrame]) -> pd.DataFrame:
    cols = sorted(set().union(*(m.columns for m in mats)), key=lambda c: c)
    stack = np.stack([np.abs(m.reindex(index=cols, columns=cols).to_numpy()) for m in mats])
    return pd.DataFrame(np.nanmean(stack, axis=0), index=cols, columns=cols)


def dedupe(rho: pd.DataFrame, order: list[str], threshold: float) -> tuple[list[str], pd.DataFrame]:
    """Жадно: идём по order, оставляем признак, если |ρ| < threshold со всеми оставленными."""
    kept: list[str] = []
    dropped = []
    a = rho.abs()
    for f in [c for c in order if c in a.index]:
        if kept:
            sims = a.loc[f, kept]
            twin = sims.idxmax()
            if sims[twin] >= threshold:
                dropped.append({"feature": f, "twin": twin, "abs_rho": float(sims[twin])})
                continue
        kept.append(f)
    return kept, pd.DataFrame(dropped, columns=["feature", "twin", "abs_rho"])


def clusters(rho: pd.DataFrame, cluster_rho: float) -> tuple[pd.Series, np.ndarray]:
    """Номер кластера для каждого признака и порядок листьев дендрограммы."""
    d = (1 - rho.abs().clip(0, 1)).to_numpy(copy=True)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    link = hierarchy.linkage(squareform(d, checks=False), method="average")
    labels = hierarchy.fcluster(link, t=1 - cluster_rho, criterion="distance")
    return pd.Series(labels, index=rho.index, name="cluster"), hierarchy.leaves_list(link)


def run(cfg, data, out: Path, threshold: float = 0.95, cluster_rho: float = 0.7, max_rows: int = 60_000) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    start, end = splits.window(cfg, "training")
    cat = catalog().set_index("name")
    mats, degenerate_by_pair = [], {}
    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            f = data.factory(symbol, tf)
            f = f[(f["ts"] >= start) & (f["ts"] < end)].drop(columns="ts")
            f = f.iloc[30 * 1440 // int(tf.rstrip("m")):]                 # первый месяц — прогрев окон
            f = f.iloc[:: max(1, len(f) // max_rows)]
            rho, deg = spearman(f)
            mats.append(rho)
            degenerate_by_pair[f"{symbol} {tf}"] = deg
            log.info("корреляции %s %s: %d признаков, %d строк, вырожденных %d", symbol, tf, rho.shape[0], len(f), len(deg))
    cons = consensus(mats)
    all_names = list(cat.index)
    degenerate = [n for n in all_names if n not in cons.index]
    order = sorted(cons.index, key=lambda n: (COST_RANK.get(cat.loc[n, "cost"], 4), all_names.index(n)))
    kept, dropped = dedupe(cons, order, threshold)
    labels, leaves = clusters(cons, cluster_rho)

    rows = []
    for k, members in labels.groupby(labels):
        names = list(members.index)
        sub = cons.loc[names, names]
        rep = sub.mean().idxmax()
        cats = cat.loc[names, "category"].value_counts()
        rows.append({"cluster": int(k), "size": len(names), "representative": rep,
                     "categories": ", ".join(f"{c} ({n})" for c, n in cats.items()),
                     "mean_abs_rho": float(sub.to_numpy()[np.triu_indices(len(names), 1)].mean()) if len(names) > 1 else 1.0,
                     "members": ", ".join(names)})
    cl = pd.DataFrame(rows).sort_values(["size", "mean_abs_rho"], ascending=False)
    cl.to_csv(out / "clusters.csv", index=False)
    dropped.to_csv(out / "duplicates.csv", index=False)
    labels.to_frame().join(cat[["id", "category"]]).to_csv(out / "feature_clusters.csv")
    (out / "core_features.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(14, 14))
    ordered = cons.index[leaves]
    ax.imshow(cons.loc[ordered, ordered].to_numpy(), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("|ρ| Спирмена между признаками (консенсус ETH/SOL/BTC × 5m/15m, training), порядок дендрограммы")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "correlation_clustered.png", dpi=90)
    plt.close(fig)

    multi = cl[cl["size"] > 1]
    cross = multi[multi["categories"].str.count(r"\(") > 1]
    w = ["# Feature Factory: корреляции, дубли, кластеры\n",
         f"Данные: training {start.date()} — {end.date()}, {len(mats)} пар монета × ТФ, до {max_rows} баров на пару "
         "(первый месяц пропущен — прогрев окон). Validation и holdout не использовались.\n",
         "## Итог\n",
         md.table(["Признаков в каталоге", "Вырожденных", "Дублей (\\|ρ\\| ≥ %.2f)" % threshold, "Ядро (уникальные)",
                   f"Кластеров (\\|ρ\\| ≥ {cluster_rho})", "Из них одиночек"],
                  [[len(all_names), len(degenerate), len(dropped), len(kept), len(cl), int((cl['size'] == 1).sum())]]) + "\n",
         "Ядро — `core_features.json`: признаки, из которых собираются гипотезы. Удалённые дубли можно "
         "использовать, но они не несут новой информации.\n",
         "![корреляции](correlation_clustered.png)\n",
         "## Вырожденные признаки\n",
         (", ".join(f"`{n}`" for n in degenerate) or "нет") + "\n",
         "Вырожденные в отдельных парах (например, контекст монеты для самой себя — NaN по определению):\n",
         "\n".join(f"- {k}: {len(v)}" for k, v in degenerate_by_pair.items()) + "\n",
         "## Дубли: что удалено и чем заменяется\n",
         md.table(["Удалён", "Оставлен вместо него", "\\|ρ\\|"],
                  [[f"`{r.feature}`", f"`{r.twin}`", md.f(r.abs_rho)] for r in dropped.itertuples()]) + "\n",
         f"## Кластеры: какие признаки описывают одно и то же (|ρ| ≥ {cluster_rho} в среднем)\n",
         f"Кластеров из нескольких признаков: {len(multi)}; из них смешивают разные категории: {len(cross)} — "
         "это места, где разные «идеи» на деле измеряют одно и то же.\n",
         md.table(["#", "Размер", "Средний \\|ρ\\|", "Представитель", "Категории", "Состав"],
                  [[r.cluster, r.size, md.f(r.mean_abs_rho), f"`{r.representative}`", r.categories,
                    r.members if len(r.members) < 400 else r.members[:400] + " …"] for r in multi.itertuples()]) + "\n"]
    path = out / "REPORT.md"
    path.write_text("\n".join(w), encoding="utf-8")
    return path
