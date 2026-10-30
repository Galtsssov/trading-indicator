"""Automatic Discovery: что статистически чаще происходит после событий и аномалий.

Работает ТОЛЬКО на training. Для каждой пары (монета, ТФ), каждого события из библиотеки и каждой
аномалии (значение признака за своим 99-м скользящим процентилем):
- считаем будущее движение через h баров в ATR со знаком направления события
  (> 0 — движение продолжилось, < 0 — развернулось);
- сравниваем с контрольной выборкой (тот же час, та же дециль ATR, то же направление);
- то же отдельно внутри каждого режима рынка;
- контроль — 5 случайных моментов на событие; строка учитывается, только если событий ≥ min_events
  и они пришлись хотя бы на min_days разных дней;
- p-value — t-статистика с бутстреп-SE по дням и df = (число дней − 1), поправка BH по ВСЕМ проверкам скана.
Находка «устойчива», если значима на ETH и SOL с одним знаком. Устойчивые находки становятся
кандидатами-гипотезами в YAML и регистрируются ДО проверки на validation по шаблону критериев из конфига.
Это поиск без затрат на сделку (gross): издержки учитываются при проверке гипотезы.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats as st  # noqa: E402

from qlab import controls, events, md, splits, stats  # noqa: E402
from qlab.cache import Data  # noqa: E402
from qlab.config import TF_MINUTES, Config  # noqa: E402
from qlab.hypothesis import Hypothesis  # noqa: E402
from qlab.registry import Registry, RegistryError  # noqa: E402

log = logging.getLogger(__name__)
REGIME_COLS = ("regime_trend", "regime_vol", "regime_btc", "regime_risk")
ROBUST_MARKETS = ("ETHUSDT", "SOLUSDT")

# Аномалии: (признак, сторона, как задаётся направление)
ANOMALIES = [
    ("vol_spike", "high", "bar"),        # объём выше 99-го процентиля
    ("wick_up_atr", "high", "up"),       # самые длинные верхние тени
    ("wick_dn_atr", "high", "down"),     # самые длинные нижние тени
    ("body_atr", "abs", "sign"),         # самые сильные импульсные свечи
    ("range_atr", "high", "bar"),        # самый большой размах бара
    ("atr_pct", "high", "bar"),          # экстремальный ATR
    ("btc_gap", "abs", "sign"),          # экстремальное расхождение с BTC
    ("taker_imb", "abs", "sign"),        # экстремальный перекос агрессоров
    ("vwap_dist_atr", "abs", "sign"),    # экстремальное удаление от VWAP
    ("oi_change", "abs", "sign"),        # экстремальное изменение OI
]


def specs(cfg: Config) -> list[tuple[str, str, dict]]:
    d = cfg.section("discovery")
    out = [(name, name, {}) for name in sorted(events.load_all()) if name not in ("anomaly", "session_open")]
    out += [(f"session_open:{s}", "session_open", {"session": s}) for s in ("asia", "london", "ny")]
    out += [(f"anomaly:{f}:{side}", "anomaly", {"feature": f, "side": side, "dir_mode": mode,
                                               "quantile": d["quantile"], "window_days": d["window_days"]})
            for f, side, mode in ANOMALIES]
    return out


def _days(ts: pd.Series) -> np.ndarray:
    return (ts.dt.floor("D").dt.as_unit("ns").astype("int64") // 86_400_000_000_000).to_numpy()


def scan_market(cfg: Config, data: Data, market: str, tf: str, rng) -> list[dict]:
    d = cfg.section("discovery")
    start, end = splits.window(cfg, "training")
    bars = data.bars(market, tf)
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    close, atr = bars["close"], bars["atr"]
    fwd = {}
    for h in d["horizons"]:
        x = ((close.shift(-h) - close) / atr).to_numpy(copy=True)
        x[(bars["available_at"] + h * step > end).to_numpy()] = np.nan  # не заглядываем за конец training
        fwd[h] = x
    cost_atr = (cfg.cost_pct_round_trip / bars["atr_pct"]).to_numpy()
    pool = splits.select(bars, start, end)
    pool = pool[pool["complete"]]
    rows = []
    for label, etype, params in specs(cfg):
        if params.get("feature") == "btc_gap" and market == cfg.btc_symbol:
            continue
        try:
            ev = events.detect(etype, bars, params)
        except (KeyError, ValueError) as e:
            log.warning("%s %s %s: пропуск (%s)", market, tf, label, e)
            continue
        ev = splits.select(ev, start, end)
        if len(ev) < d["min_events"]:
            continue
        ev = ev.join(bars[["hour", "atr_pctl", *REGIME_COLS]], on="bar").assign(direction=lambda x: x["dir"])
        ctrl = pd.concat([controls.matched(pool, ev, int(rng.integers(2**31))) for _ in range(d["controls_per_event"])],
                         ignore_index=True)
        ctrl = ctrl.join(bars[list(REGIME_COLS)], on="bar")
        subsets = [("all", "all")] + [(c, v) for c in REGIME_COLS for v in sorted(ev[c].dropna().unique())]
        for col, val in subsets:
            e = ev if col == "all" else ev[ev[col] == val]
            c = ctrl if col == "all" else ctrl[ctrl[col] == val]
            for h in d["horizons"]:
                x = e["dir"].to_numpy() * fwd[h][e["bar"].to_numpy()]
                y = c["direction"].to_numpy() * fwd[h][c["bar"].to_numpy()]
                xm, ym = np.isfinite(x), np.isfinite(y)
                ev_days = _days(e["ts"])[xm]
                n_days = len(np.unique(ev_days))
                if xm.sum() < d["min_events"] or n_days < d["min_days"] or ym.sum() < 30:
                    continue
                dist = stats.day_bootstrap_dist([y[ym], x[xm]], [_days(c["ts"])[ym], ev_days],
                                                lambda m: m[1] - m[0], d["bootstrap"], rng)
                delta = float(x[xm].mean() - y[ym].mean())
                se = float(np.nanstd(dist))
                z = delta / se if se > 0 else 0.0
                rows.append({
                    "market": market, "tf": tf, "spec": label, "event": etype, "params": params,
                    "regime": col.removeprefix("regime_"), "value": val, "h": h,
                    "n": int(xm.sum()), "n_days": n_days, "n_ctrl": int(ym.sum()), "mean": float(x[xm].mean()),
                    "mean_ctrl": float(y[ym].mean()), "delta": delta, "se": se, "z": z,
                    # t с df = число дней − 1: дни — независимые кластеры, их может быть немного
                    "p": float(2 * st.t.sf(abs(z), df=n_days - 1)), "p_up": float((x[xm] > 0).mean()),
                    "cost_atr": float(np.nanmean(cost_atr[e["bar"].to_numpy()[xm]])),
                    "vol_ratio": float(np.abs(x[xm]).mean() / np.abs(y[ym]).mean()),
                })
    log.info("discovery %s %s: %d проверок", market, tf, len(rows))
    return rows


def scan(cfg: Config, data: Data) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.section("stats")["seed"])
    rows = [r for m in cfg.symbols for tf in cfg.timeframes for r in scan_market(cfg, data, m, tf, rng)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    d = cfg.section("discovery")
    df["p_adj"] = stats.benjamini_hochberg(df["p"].to_numpy())
    df["significant"] = (df["p_adj"] < d["alpha"]) & (df["delta"].abs() >= d["min_delta_atr"])
    return df


def robust(df: pd.DataFrame) -> pd.DataFrame:
    """Находки, значимые на ETH и SOL с одним знаком (тот же спецификатор, ТФ, режим, горизонт)."""
    key = ["spec", "tf", "regime", "value", "h"]
    sub = df[df["market"].isin(ROBUST_MARKETS)]
    rows = []
    for k, g in sub.groupby(key):
        if len(g) == len(ROBUST_MARKETS) and g["significant"].all() and len(set(np.sign(g["delta"]))) == 1:
            rows.append({**dict(zip(key, k)), "event": g["event"].iloc[0], "params": g["params"].iloc[0],
                         "delta_eth": g.loc[g.market == "ETHUSDT", "delta"].iloc[0],
                         "delta_sol": g.loc[g.market == "SOLUSDT", "delta"].iloc[0],
                         "z_min": float(g["z"].abs().min()), "n_min": int(g["n"].min()),
                         "cost_atr": float(g["cost_atr"].mean()), "sign": int(np.sign(g["delta"].iloc[0]))})
    return pd.DataFrame(rows)


def candidates(rob: pd.DataFrame, max_n: int) -> pd.DataFrame:
    """Одна лучшая строка (горизонт) на (спецификатор, ТФ, режим) — по минимальному |z| из двух монет."""
    if rob.empty:
        return rob
    best = rob.sort_values("z_min", ascending=False).groupby(["spec", "tf", "regime", "value"]).head(1)
    return best.head(max_n).reset_index(drop=True)


def register_candidates(cfg: Config, reg: Registry, cand: pd.DataFrame, stamp: str,
                        root: Path = Path("hypotheses/discovered")) -> list[Hypothesis]:
    c = cfg.section("discovery")
    out = []
    for _, r in cand.iterrows():
        hid = reg.next_id()
        cont = r["sign"] > 0
        regime = "" if r["regime"] == "all" else f", режим {r['value']}"
        raw = {
            "id": hid,
            "name": f"После {r['spec']}: {'продолжение' if cont else 'разворот'} ({r['tf']}, {r['h']} бар.{regime})",
            "description": f"Автоматическая находка discovery {stamp}: через {r['h']} баров после события цена "
                           f"чаще {'продолжает' if cont else 'разворачивает'} движение, чем в случайный момент.",
            "source": f"discovery {stamp} (training): ΔETH={r['delta_eth']:+.3f} ATR, ΔSOL={r['delta_sol']:+.3f} ATR, "
                      f"min|z|={r['z_min']:.1f}, издержки ≈ {r['cost_atr']:.2f} ATR",
            "created": str(date.today()),
            "rationale": "Механизм заранее не предполагается: находка получена автоматическим поиском на training "
                         "с поправкой BH по всем проверкам скана и требованием одного знака на ETH и SOL. "
                         "Только проверка на validation решает, существует ли эффект после издержек.",
            "event": {"type": r["event"], "params": dict(r["params"])},
            "direction": "with" if cont else "against",
            "filters": [] if r["regime"] == "all" else [f"regime_{r['regime']} == '{r['value']}'"],
            "markets": list(ROBUST_MARKETS),
            "timeframes": [r["tf"]],
            "outcome": {"r_atr": 1.0, "target_r": 1.0, "horizon_bars": int(r["h"])},
            "features_used": [r["params"].get("feature", r["event"]), "atr", "hour", "atr_pctl"]
                             + ([f"regime_{r['regime']}"] if r["regime"] != "all" else []),
            "confirmation": "Критерии шаблона [discovery.criteria] на validation: E после издержек > 0 (CI) и "
                            "превосходство над всеми эталонами (ΔE ≥ порога, CI > 0, p(BH) < alpha) на ETH и SOL.",
            "refutation": "Любой критерий не выполнен на validation.",
            "criteria": {**c["criteria"], "require_markets": list(ROBUST_MARKETS)},
        }
        path = root / f"{hid}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        h = Hypothesis.load(path)
        try:
            reg.register(h)
            out.append(h)
        except RegistryError as e:
            log.info("кандидат не зарегистрирован: %s", e)
            path.unlink()
    return out


def _heatmap(df: pd.DataFrame, tf: str, path: Path) -> None:
    sub = df[(df["tf"] == tf) & (df["regime"] == "all")]
    if sub.empty:
        return
    piv = sub.pivot_table(index="spec", columns=["market", "h"], values="z")
    fig, ax = plt.subplots(figsize=(1 + 0.45 * piv.shape[1], 0.5 + 0.32 * piv.shape[0]))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
    ax.set_xticks(range(piv.shape[1]), [f"{m[:3]} {h}" for m, h in piv.columns], rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, label="z: > 0 продолжение, < 0 разворот (vs случайный момент)")
    ax.set_title(f"Discovery, training, {tf}, все режимы")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(cfg: Config, df: pd.DataFrame, rob: pd.DataFrame, cand: pd.DataFrame,
                 registered: list[Hypothesis], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    d = cfg.section("discovery")
    start, end = splits.window(cfg, "training")
    df.drop(columns=["params"]).to_csv(out / "scan.csv", index=False)
    w = [f"# Discovery: что происходит после событий и аномалий\n",
         f"Период: **training** {start.date()} — {end.date()} (validation и holdout не использовались). "
         f"Сформировано {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.\n",
         "> Расчёт исследовательский, не торговая рекомендация. Движения до издержек (gross).\n",
         "## Метод\n",
         f"- {df['spec'].nunique() if not df.empty else 0} типов событий и аномалий (аномалия — значение признака выше "
         f"своего {d['quantile']:.0%}-процентиля за {d['window_days']} дней); горизонты {d['horizons']} баров; "
         "9 срезов: все бары + 8 режимов.",
         "- Метрика: движение через h баров в ATR со знаком направления события (> 0 — продолжение, < 0 — разворот), "
         "минус то же для контрольной выборки (тот же час и дециль ATR).",
         f"- p-value из бутстреп-SE по дням; поправка Бенджамини–Хохберга по **всем {len(df)} проверкам**; "
         f"значимо: p(BH) < {d['alpha']} и |Δ| ≥ {d['min_delta_atr']} ATR.",
         "- Устойчиво: значимо и на ETH, и на SOL с одним знаком.\n",
         "## Итог\n",
         md.table(["Проверок", "Значимых", "Устойчивых (ETH и SOL)", "Кандидатов зарегистрировано"],
                  [[len(df), int(df["significant"].sum()) if not df.empty else 0, len(rob), len(registered)]]) + "\n"]
    if not df.empty:
        top = df[df["significant"]].reindex(df[df["significant"]]["z"].abs().sort_values(ascending=False).index)
        w.append("## Значимые находки (топ-40 по |z|)\n")
        w.append(md.table(["Монета", "ТФ", "Событие/аномалия", "Режим", "h", "N", "Δ ATR", "z", "p(BH)",
                           "P(движение > 0)", "Издержки ATR", "Вол. после / контроль"],
                          [[r.market, r.tf, r.spec, f"{r.regime}:{r.value}", r.h, r.n, md.f(r.delta), md.f(r.z, 1),
                            md.p(r.p_adj), md.f(r.p_up), md.f(r.cost_atr, 2), md.f(r.vol_ratio, 2)]
                           for r in top.head(40).itertuples()]) + "\n")
        for tf in cfg.timeframes:
            _heatmap(df, tf, out / f"heatmap_{tf}.png")
            w.append(f"![heatmap {tf}](heatmap_{tf}.png)\n")
    w.append("## Устойчивые находки (ETH и SOL, один знак)\n")
    w.append(md.table(["Событие/аномалия", "ТФ", "Режим", "h", "Δ ETH", "Δ SOL", "min|z|", "min N", "Издержки ATR"],
                      [[r.spec, r.tf, f"{r.regime}:{r.value}", r.h, md.f(r.delta_eth), md.f(r.delta_sol),
                        md.f(r.z_min, 1), r.n_min, md.f(r.cost_atr, 2)] for r in rob.itertuples()])
             if not rob.empty else "Нет.")
    w.append("\n## Зарегистрированные кандидаты\n")
    w.append("\n".join(f"- {h.id}: {h.name} — {h.source}" for h in registered) or "Нет.")
    path = out / "REPORT.md"
    path.write_text("\n".join(w) + "\n", encoding="utf-8")
    return path
