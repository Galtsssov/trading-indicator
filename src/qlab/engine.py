"""Hypothesis Engine: проверка YAML-гипотезы на одном периоде истории.

Шаги для каждой пары (монета, ТФ) из гипотезы:
события → окно периода (с запасом на горизонт) → фильтры → сделки → исходы →
эталоны (та же модель исхода, тот же период) → сравнения → разбивка по режимам.
Вердикт выносится строго по критериям из YAML; результат записывается в реестр и в отчёт.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qlab import baselines, events, expr, md, outcomes, splits, stats  # noqa: E402
from qlab.cache import Data  # noqa: E402
from qlab.config import TF_MINUTES, Config  # noqa: E402
from qlab.hypothesis import Hypothesis  # noqa: E402
from qlab.registry import Registry, RegistryError  # noqa: E402

log = logging.getLogger(__name__)
REGIME_COLS = ("regime_trend", "regime_vol", "regime_btc", "regime_risk")
BASE_COLS = ["hour", "atr_pctl", *REGIME_COLS]
MIN_HOLDOUT_DAYS = 90


@dataclass
class MarketResult:
    market: str
    tf: str
    counts: dict[str, int]
    hypothesis: dict
    baselines: dict[str, dict] = field(default_factory=dict)
    effect_gross_vs_random: dict | None = None
    regimes: list[dict] = field(default_factory=list)
    fails: list[str] = field(default_factory=list)
    passed: bool = False


def _days(df: pd.DataFrame) -> np.ndarray:
    return (df["ts"].dt.floor("D").dt.as_unit("ns").astype("int64") // 86_400_000_000_000).to_numpy()


def _outcomes(trades: pd.DataFrame, bars: pd.DataFrame, m1: pd.DataFrame, op: outcomes.OutcomeParams) -> pd.DataFrame:
    trades = trades.drop(columns=[c for c in BASE_COLS if c in trades.columns]).join(bars[BASE_COLS], on="bar")
    res = outcomes.compute(trades.reset_index(drop=True), m1, op)
    return res[res["exclude_reason"].isna()]


def _summ(df: pd.DataFrame, col: str, b: int, rng) -> dict:
    return stats.summarize(df[col].to_numpy(), df["mfe_r"].to_numpy(), df["mae_r"].to_numpy(),
                           _days(df), b, rng).as_dict()


def _cmp(base: pd.DataFrame, hyp: pd.DataFrame, col: str, b: int, n_perm: int, rng) -> dict | None:
    c = stats.compare(base[col].to_numpy(), hyp[col].to_numpy(), _days(base), _days(hyp), b, n_perm, rng)
    return c.as_dict() if c else None


def evaluate_market(cfg: Config, data: Data, h: Hypothesis, market: str, tf: str,
                    start: pd.Timestamp, end: pd.Timestamp, rng) -> MarketResult:
    st, o = cfg.section("stats"), h.outcome
    tf_min = TF_MINUTES[tf]
    horizon = pd.Timedelta(minutes=o["horizon_bars"] * tf_min)
    op = outcomes.OutcomeParams(r_atr=o["r_atr"], targets=(o["target_r"],), horizon_minutes=o["horizon_bars"] * tf_min,
                                cost_pct_round_trip=cfg.cost_pct_round_trip)
    net, gross = f"res_{o['target_r']:g}r_net", f"res_{o['target_r']:g}r"
    bars, m1 = data.bars(market, tf), data.m1(market)

    filter_cols = sorted(set().union(*(expr.names(f) for f in h.filters)) - set(BASE_COLS)) if h.filters else []
    unknown = [c for c in filter_cols if c not in bars.columns]
    if unknown:
        raise ValueError(f"{h.id}: в фильтрах неизвестные признаки {unknown}")
    ev = events.detect(h.event["type"], bars, h.event.get("params"))
    ev = ev.join(bars[filter_cols + BASE_COLS], on="bar")
    counts = {"events_all_history": len(ev)}
    ev = splits.select(ev, start, end, horizon)
    counts["events_in_period"] = len(ev)
    for f in h.filters:
        ev = ev[expr.evaluate(f, ev)]
    counts["after_filters"] = len(ev)
    sign = {"with": ev["dir"], "against": -ev["dir"], "long": 1, "short": -1}[h.direction]
    trades = ev.assign(direction=sign)
    hyp = _outcomes(trades, bars, m1, op)
    counts["trades_with_outcome"] = len(hyp)
    res = MarketResult(market, tf, counts, _summ(hyp, net, st["bootstrap"], rng))
    if len(hyp) < 2:
        res.fails.append(f"сделок {len(hyp)}")
        return res

    pool = splits.select(bars, start, end, horizon)
    pool = pool[pool["complete"]]
    base_trades: dict[str, pd.DataFrame] = {}
    for name in h.criteria.get("beat_baselines", baselines.NAMES):
        bt = baselines.build(name, bars, hyp, cfg.section("baselines"), int(rng.integers(2**31)), pool=pool)
        bt = splits.select(bt, start, end, horizon)
        base_trades[name] = _outcomes(bt, bars, m1, op)
        counts[f"baseline_{name}"] = len(base_trades[name])
        res.baselines[name] = {"stats": _summ(base_trades[name], net, st["bootstrap"], rng),
                               "cmp": _cmp(base_trades[name], hyp, net, st["bootstrap"], st["permutations"], rng)}
    rnd = base_trades.get("random_entry")
    if rnd is not None and len(rnd) > 1:
        res.effect_gross_vs_random = _cmp(rnd, hyp, gross, st["bootstrap"], st["permutations"], rng)
        for col in REGIME_COLS:
            for val in sorted(v for v in hyp[col].dropna().unique()):
                hs, rs = hyp[hyp[col] == val], rnd[rnd[col] == val]
                c = _cmp(rs, hs, net, st["bootstrap"], 500, rng) if len(hs) > 1 and len(rs) > 1 else None
                res.regimes.append({"regime": col.removeprefix("regime_"), "value": val, "n": len(hs),
                                    "E_net": float(hs[net].mean()), "E_gross": float(hs[gross].mean()),
                                    "E_random_net": float(rs[net].mean()) if len(rs) else float("nan"),
                                    "delta_vs_random": c["delta_expectancy"] if c else float("nan"),
                                    "delta_ci": c["delta_ci"] if c else (float("nan"), float("nan"))})
    return res


def judge(results: list[MarketResult], h: Hypothesis) -> tuple[bool, str]:
    """Вердикт по критериям YAML. BH-поправка — по всем сравнениям с эталонами в этой проверке."""
    c = h.criteria
    cmps = [(r, name, b["cmp"]) for r in results for name, b in r.baselines.items() if b["cmp"]]
    adj = stats.benjamini_hochberg(np.array([x["p_primary"] for _, _, x in cmps]))
    for (_, _, x), p in zip(cmps, adj):
        x["p_adjusted"] = float(p)
    for r in results:
        if r.counts.get("trades_with_outcome", 0) < c["min_events"]:
            r.fails.append(f"сделок {r.counts.get('trades_with_outcome', 0)} < {c['min_events']}")
        if c.get("require_positive_net", True) and not r.hypothesis["expectancy_ci"][0] > 0:
            r.fails.append(f"E после издержек {r.hypothesis['expectancy']:+.3f}R, CI {md.f(r.hypothesis['expectancy_ci'])}")
        for name in c.get("beat_baselines", baselines.NAMES):
            x = r.baselines.get(name, {}).get("cmp")
            ok = x and x["delta_expectancy"] >= c["practical_min_delta_r"] and x["delta_ci"][0] > 0 \
                and x["p_adjusted"] < c["alpha"]
            if not ok:
                r.fails.append(f"не лучше {name}" + (f" (Δ {x['delta_expectancy']:+.3f}R)" if x else ""))
        r.passed = not r.fails
    required = c.get("require_markets") or h.markets
    for tf in h.timeframes:
        if all(any(r.passed for r in results if r.market == m and r.tf == tf) for m in required):
            return True, f"пройдено на {tf} для {', '.join(required)}"
    reasons = [f"{r.market} {r.tf}: {r.fails[0]}" for r in results if r.market in required and r.fails]
    return False, "; ".join(reasons[:4])


def run(cfg: Config, data: Data, h: Hypothesis, reg: Registry, split: str, *, confirm_holdout: bool = False,
        out_root: Path = Path("reports/checks")) -> tuple[bool, Path]:
    reg.require_unchanged(h)
    if split == "holdout":
        if not confirm_holdout:
            raise splits.HoldoutLocked("holdout открывается только с --confirm-holdout")
        start, end = splits.window(cfg, "holdout", allow_holdout=True, clean_after=h.raw.get("clean_after"))
        last = max(data.bars(m, h.timeframes[0])["ts"].max() for m in h.markets)
        if (last - start) < pd.Timedelta(days=MIN_HOLDOUT_DAYS):
            raise RegistryError(f"{h.id}: в holdout меньше {MIN_HOLDOUT_DAYS} дней чистых данных "
                                f"(с {start.date()} по {last.date()}) — ждать, holdout не тронут")
        reg.open_holdout(h)
    else:
        start, end = splits.window(cfg, split)
    rng = np.random.default_rng(cfg.section("stats")["seed"])
    results = [evaluate_market(cfg, data, h, m, tf, start, end, rng) for m in h.markets for tf in h.timeframes]
    passed, reason = judge(results, h)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = out_root / h.id
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"{split}-{stamp}.md"
    _write_report(report, h, split, start, end, results, passed, reason)
    (out / f"{split}-{stamp}.json").write_text(json.dumps(
        {"id": h.id, "split": split, "start": str(start), "end": str(end), "passed": passed, "reason": reason,
         "lock": h.lock(), "results": [asdict(r) for r in results]}, indent=1, ensure_ascii=False, default=str))
    reg.record(h, split, passed, reason, str(report))
    log.info("%s %s: %s — %s", h.id, split, "ПРОЙДЕНО" if passed else "ОТКЛОНЕНО", reason)
    return passed, report


def _figure(path: Path, results: list[MarketResult]) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.9 * len(results)))
    y = 0
    labels = []
    for r in results:
        rows = [("гипотеза", r.hypothesis)] + [(n, b["stats"]) for n, b in r.baselines.items()]
        for i, (name, s) in enumerate(rows):
            lo, hi = s["expectancy_ci"]
            if np.isfinite(s["expectancy"]):
                ax.errorbar(s["expectancy"], y + i * 0.12, xerr=[[s["expectancy"] - lo], [hi - s["expectancy"]]],
                            fmt="o" if i == 0 else ".", color="#d1495b" if i == 0 else "#8d8d8d", capsize=2)
        labels.append((y + 0.3, f"{r.market} {r.tf}"))
        y += 1
    ax.set_yticks([v for v, _ in labels], [t for _, t in labels])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ожидание после издержек, R (красное — гипотеза, серое — эталоны)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_report(path: Path, h: Hypothesis, split: str, start, end, results: list[MarketResult],
                  passed: bool, reason: str) -> None:
    fig = path.with_suffix(".png")
    _figure(fig, results)
    w: list[str] = []
    w.append(f"# {h.id}: {h.name} — проверка на {split}\n")
    w.append(f"Период: {start.date()} — {min(end, pd.Timestamp.now(tz='UTC')).date()}. Замок гипотезы `{h.lock()}`. "
             f"Сформировано {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.\n")
    w.append("> Расчёт исследовательский, не торговая рекомендация.\n")
    w.append(f"## Вердикт: {'ПРОЙДЕНО' if passed else 'ОТКЛОНЕНО'}\n\n{reason}\n")
    w.append(f"Событие `{h.event['type']}` {h.event_params}, направление `{h.direction}`, "
             f"фильтры: {list(h.filters) or 'нет'}; исход: цель {h.outcome['target_r']}R / стоп 1R, "
             f"1R = {h.outcome['r_atr']} ATR, горизонт {h.outcome['horizon_bars']} баров.\n")
    rows = []
    for r in results:
        s, e = r.hypothesis, r.effect_gross_vs_random
        rows.append([r.market, r.tf, r.counts.get("trades_with_outcome", 0), md.f(s["win_rate"]),
                     md.f(s["expectancy"]), md.f(s["expectancy_ci"]),
                     md.f(e["delta_expectancy"]) if e else "—", md.f(e["delta_ci"]) if e else "—",
                     "да" if r.passed else "нет"])
    w.append(md.table(["Монета", "ТФ", "Сделок", "Win rate", "E после издержек, R", "95% CI",
                       "Эффект до издержек vs случайный, R", "CI эффекта", "Пройдено"], rows) + "\n")
    w.append(f"![сравнение]({fig.name})\n")
    w.append("## Сравнение с эталонами (ΔE = гипотеза − эталон, после издержек)\n")
    rows = []
    for r in results:
        for name, b in r.baselines.items():
            x = b["cmp"] or {}
            rows.append([r.market, r.tf, name, b["stats"]["n"], md.f(b["stats"]["expectancy"]),
                         md.f(x.get("delta_expectancy")), md.f(x.get("delta_ci")), md.p(x.get("p_adjusted", float("nan"))),
                         md.f(x.get("cliffs_delta")), x.get("primary_test", "—")])
    w.append(md.table(["Монета", "ТФ", "Эталон", "N", "E эталона", "ΔE", "95% CI", "p (BH)", "Cliff's δ", "Тест"],
                      rows) + "\n")
    w.append("## Режимы рынка (описательно; режим-специфичный эффект = новая гипотеза)\n")
    rows = [[r.market, r.tf, g["regime"], g["value"], g["n"], md.f(g["E_net"]), md.f(g["E_gross"]),
             md.f(g["E_random_net"]), md.f(g["delta_vs_random"]), md.f(g["delta_ci"])]
            for r in results for g in r.regimes]
    w.append(md.table(["Монета", "ТФ", "Ось", "Режим", "N", "E net", "E gross", "E случайный", "Δ vs случ.",
                       "95% CI"], rows) + "\n")
    w.append("## Причины по рынкам\n")
    w.append("\n".join(f"- {r.market} {r.tf}: " + ("пройдено" if r.passed else "; ".join(r.fails)) for r in results))
    w.append("\n## Счётчики\n")
    keys = list(dict.fromkeys(k for r in results for k in r.counts))
    w.append(md.table(["Монета", "ТФ", *keys], [[r.market, r.tf, *(r.counts.get(k, "—") for k in keys)]
                                                 for r in results]) + "\n")
    path.write_text("\n".join(w), encoding="utf-8")
