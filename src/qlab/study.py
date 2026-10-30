"""Исследование NIF (этап 1, гипотеза 1): исход свипа определяется характером потока.

Группа A — OI растёт к моменту решения (oi_change > 0): ожидаем продолжение, реверсия хуже.
Группа B — OI падает (oi_change < 0): ожидаем реверсию.
Сделка в обеих группах одна и та же — реверсия против свипа. Проверяем: E[B] − E[A] > 0.

Пререгистрированное правило вердикта для пары (монета, ТФ) при основном depth и торгуемой задержке OI:
- «подтверждено»: Δ > 0, нижняя граница 95% CI(Δ) > 0, p после BH < alpha, Δ ≥ practical_min_delta_r;
- «обратный эффект»: те же условия с противоположным знаком;
- иначе «не подтверждено».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qlab import controls, dataset, features, outcomes, stats
from qlab.config import TF_MINUTES, Config
from qlab.events import sweep

log = logging.getLogger(__name__)

EVENT_FEATURES = ["oi_change", "atr_pctl", "hour", "session", "weekend", "vol_spike",
                  "vwap_dist_atr", "btc_dir", "funding_rate"]


@dataclass
class Inputs:
    m1: dict[str, pd.DataFrame]
    oi: dict[str, pd.DataFrame | None]
    funding: dict[str, pd.DataFrame | None]


@dataclass
class RunResult:
    symbol: str
    tf: str
    depth_atr: float
    oi_lag_minutes: float
    tradable: bool
    counts: dict[str, int]
    groups: dict[str, dict]
    a_vs_b: dict | None
    sweeps_vs_control: dict | None
    by_year: list[dict] = field(default_factory=list)
    by_side: list[dict] = field(default_factory=list)
    verdict: str = ""


def bars_for(cfg: Config, inp: Inputs, symbol: str, tf: str, oi_lag: float) -> pd.DataFrame:
    av = cfg.section("availability")
    bars = dataset.build(inp.m1[symbol], tf, oi=inp.oi[symbol], funding=inp.funding[symbol],
                         btc_m1=inp.m1.get(cfg.btc_symbol), oi_lag_minutes=oi_lag,
                         oi_max_staleness_minutes=av["oi_max_staleness_minutes"])
    return features.compute(bars, TF_MINUTES[tf], cfg.section("features"))


def outcome_params(cfg: Config, tf: str) -> outcomes.OutcomeParams:
    o = cfg.section("outcomes")
    return outcomes.OutcomeParams(r_atr=o["r_atr"], targets=tuple(o["targets"]),
                                  horizon_minutes=o["horizon_bars"] * TF_MINUTES[tf],
                                  cost_pct_round_trip=cfg.cost_pct_round_trip)


def events_with_outcomes(cfg: Config, bars: pd.DataFrame, m1: pd.DataFrame, tf: str,
                         depth: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sp = sweep.SweepParams(**{**cfg.section("events", "sweep"), "depth_atr": depth})
    ev = sweep.detect(bars, sp)
    ev = ev.join(bars[EVENT_FEATURES], on="bar")
    op = outcome_params(cfg, tf)
    ev = outcomes.compute(ev, m1, op)
    ctrl = controls.matched(bars, ev, seed)
    ctrl = outcomes.compute(ctrl, m1, op)
    return ev, ctrl


def _days(df: pd.DataFrame) -> np.ndarray:
    return (df["ts"].dt.floor("D").dt.as_unit("ns").astype("int64") // 86_400_000_000_000).to_numpy()


def _group(df: pd.DataFrame, col: str, b: int, rng) -> dict:
    return stats.summarize(df[col].to_numpy(), df["mfe_r"].to_numpy(), df["mae_r"].to_numpy(),
                           _days(df), b, rng).as_dict()


def _cmp(a: pd.DataFrame, bb: pd.DataFrame, col: str, s: dict, rng) -> dict | None:
    c = stats.compare(a[col].to_numpy(), bb[col].to_numpy(), _days(a), _days(bb),
                      s["bootstrap"], s["permutations"], rng)
    return c.as_dict() if c else None


def split_groups(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """A: OI растёт, B: OI падает. Всё исключённое считается с причиной."""
    counts = {"sweeps_total": len(ev)}
    no_outcome = ev["exclude_reason"].notna()
    for reason, n in ev.loc[no_outcome, "exclude_reason"].value_counts().items():
        counts[f"excluded: {reason}"] = int(n)
    rest = ev[~no_outcome]
    no_oi = rest["oi_change"].isna()
    counts["excluded: OI неизвестен (нет данных или устарел)"] = int(no_oi.sum())
    rest = rest[~no_oi]
    flat = rest["oi_change"] == 0
    counts["excluded: OI не изменился (ровно 0)"] = int(flat.sum())
    a, b = rest[rest["oi_change"] > 0], rest[rest["oi_change"] < 0]
    counts["group_A_oi_up"], counts["group_B_oi_down"] = len(a), len(b)
    return a, b, counts


def run_one(cfg: Config, bars: pd.DataFrame, m1: pd.DataFrame, symbol: str, tf: str, depth: float,
            oi_lag: float, tradable: bool, rng, keep_events: list | None = None) -> RunResult:
    s = cfg.section("study")
    col = f"res_{cfg.section('outcomes')['primary_target']:g}r_net"
    ev, ctrl = events_with_outcomes(cfg, bars, m1, tf, depth, int(rng.integers(2**31)))
    a, b, counts = split_groups(ev)
    ctrl_ok = ctrl[ctrl["exclude_reason"].isna()]
    sweeps_ok = pd.concat([a, b])
    counts["control"] = len(ctrl_ok)
    res = RunResult(symbol, tf, depth, oi_lag, tradable, counts,
                    groups={"A": _group(a, col, s["bootstrap"], rng), "B": _group(b, col, s["bootstrap"], rng),
                            "all_sweeps": _group(sweeps_ok, col, s["bootstrap"], rng),
                            "control": _group(ctrl_ok, col, s["bootstrap"], rng)},
                    a_vs_b=_cmp(a, b, col, s, rng),
                    sweeps_vs_control=_cmp(ctrl_ok, sweeps_ok, col, s, rng))
    for name, key in (("by_year", a["ts"].dt.year), ("by_side", a["side"])):
        bkey = b["ts"].dt.year if name == "by_year" else b["side"]
        rows = []
        for k in sorted(set(key) | set(bkey)):
            ga, gb = a[key == k][col], b[bkey == k][col]
            rows.append({"key": str(k), "n_A": len(ga), "n_B": len(gb),
                         "E_A": float(ga.mean()) if len(ga) else float("nan"),
                         "E_B": float(gb.mean()) if len(gb) else float("nan")})
        setattr(res, name, rows)
    if keep_events is not None:
        keep_events.append(pd.concat([a.assign(group="A"), b.assign(group="B"),
                                      ctrl_ok.assign(group="control")]).assign(symbol=symbol, tf=tf))
    log.info("%s %s depth=%.2f lag=%g: A=%d B=%d ΔE=%s", symbol, tf, depth, oi_lag, len(a), len(b),
             f"{res.a_vs_b['delta_expectancy']:+.4f}" if res.a_vs_b else "—")
    return res


def verdict(r: RunResult, alpha: float, practical: float) -> str:
    c = r.a_vs_b
    if not c or not np.isfinite(c["p_adjusted"]):
        return "недостаточно данных"
    d, (lo, hi), p = c["delta_expectancy"], c["delta_ci"], c["p_adjusted"]
    if d >= practical and lo > 0 and p < alpha:
        return "подтверждено"
    if d <= -practical and hi < 0 and p < alpha:
        return "обратный эффект"
    return "не подтверждено"


def run(cfg: Config, inp: Inputs) -> tuple[list[RunResult], pd.DataFrame]:
    s = cfg.section("study")
    rng = np.random.default_rng(s["seed"])
    main_depth = cfg.section("events", "sweep")["depth_atr"]
    lag = cfg.section("availability")["oi_lag_minutes"]
    results: list[RunResult] = []
    kept: list[pd.DataFrame] = []
    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            for oi_lag, tradable in ((lag, True), (s["oi_lag_diagnostic"], False)):
                bars = bars_for(cfg, inp, symbol, tf, oi_lag)
                depths = s["depth_grid"] if tradable else [main_depth]
                for depth in depths:
                    keep = kept if (tradable and depth == main_depth) else None
                    results.append(run_one(cfg, bars, inp.m1[symbol], symbol, tf, depth, oi_lag,
                                           tradable, rng, keep))
    for tradable in (True, False):  # отдельные семейства для поправки BH
        fam = [r for r in results if r.tradable == tradable and r.a_vs_b]
        adj = stats.benjamini_hochberg(np.array([r.a_vs_b["p_primary"] for r in fam]))
        for r, p in zip(fam, adj):
            r.a_vs_b["p_adjusted"] = float(p)
    for r in results:
        r.verdict = verdict(r, s["alpha"], s["practical_min_delta_r"])
    events = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    return results, events


def overall(results: list[RunResult], cfg: Config) -> dict[str, Any]:
    """Итог по пререгистрированному правилу: подтверждено на ETH и SOL на одном ТФ и в ≥3 из 4 depth."""
    main_depth = cfg.section("events", "sweep")["depth_atr"]
    trad = [r for r in results if r.tradable]
    by_tf: dict[str, dict] = {}
    for tf in cfg.timeframes:
        per_sym = {}
        for sym in cfg.symbols:
            rs = [r for r in trad if r.symbol == sym and r.tf == tf]
            main = next((r for r in rs if r.depth_atr == main_depth), None)
            per_sym[sym] = {"main": main.verdict if main else "—",
                            "confirmed_depths": sum(r.verdict == "подтверждено" for r in rs),
                            "n_depths": len(rs)}
        ok = all(per_sym.get(s, {}).get("main") == "подтверждено" and per_sym[s]["confirmed_depths"] >= 3
                 for s in ("ETHUSDT", "SOLUSDT") if s in per_sym)
        by_tf[tf] = {"per_symbol": per_sym, "supported": ok}
    return {"by_tf": by_tf, "supported": any(v["supported"] for v in by_tf.values())}
