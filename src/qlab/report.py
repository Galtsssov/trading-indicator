"""Автоматический отчёт исследования: markdown + графики. Только отображает результаты, ничего не считает заново."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qlab.config import Config  # noqa: E402
from qlab.study import RunResult  # noqa: E402

COLORS = {"A": "#d1495b", "B": "#2e86ab", "control": "#8d8d8d"}
LABELS = {"A": "A: OI растёт", "B": "B: OI падает", "control": "контроль"}


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (tuple, list)):
        return f"[{_f(x[0], nd)}; {_f(x[1], nd)}]"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _p(x: float) -> str:
    return "—" if not np.isfinite(x) else ("<0.0001" if x < 1e-4 else f"{x:.4f}")


def _table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------- графики ----------

def _fig_coverage(quality: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(quality) + 1))
    for i, q in enumerate(quality):
        if q["first"]:
            a, b = pd.Timestamp(q["first"]), pd.Timestamp(q["last"])
            ax.barh(i, (b - a).days, left=a, color="#2e86ab", height=0.5)
    ax.set_yticks(range(len(quality)), [f"{q['symbol']} {q['dataset']}" for q in quality])
    ax.set_title("Доступная история по наборам данных")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _per_symbol_grid(events: pd.DataFrame, symbols: list[str], tf: str, path: Path, title: str, draw) -> None:
    fig, axes = plt.subplots(len(symbols), 1, figsize=(9, 2.8 * len(symbols)), squeeze=False)
    for ax, sym in zip(axes[:, 0], symbols):
        sub = events[(events["symbol"] == sym) & (events["tf"] == tf)]
        draw(ax, sub)
        ax.set_title(f"{sym} {tf}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _draw_dist(col: str):
    def draw(ax, sub):
        values = sub[col].dropna()
        if values.empty:
            return
        bins = np.linspace(values.min() - 0.05, values.max() + 0.05, 61)
        for g in ("A", "B", "control"):
            x = sub.loc[sub["group"] == g, col].dropna()
            if len(x):
                ax.hist(x, bins=bins, density=True, histtype="step", lw=1.6,
                        color=COLORS[g], label=f"{LABELS[g]} (n={len(x)})")
        ax.set_xlabel("исход сделки, R после издержек")
        ax.legend(fontsize=8)
    return draw


def _draw_excursions(ax, sub):
    data, labels, colors = [], [], []
    for metric in ("mfe_r", "mae_r"):
        for g in ("A", "B", "control"):
            data.append(sub.loc[sub["group"] == g, metric].dropna().clip(upper=6))
            labels.append(f"{metric[:3].upper()} {g}")
            colors.append(COLORS[g])
    if any(len(d) for d in data):
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_ylabel("R")


def _draw_time(target: str):
    def draw(ax, sub):
        for g in ("A", "B", "control"):
            x = sub[sub["group"] == g]
            if len(x):
                t = np.fmin(x[f"t_hit_{target}r"].to_numpy(), x["t_stop"].to_numpy())
                t = np.sort(t[np.isfinite(t)])
                if len(t):
                    ax.step(t, np.arange(1, len(t) + 1) / len(x), color=COLORS[g], label=LABELS[g])
        ax.set_xlabel(f"минут до исхода (+{target}R или −1R)")
        ax.set_ylabel("доля событий")
        ax.legend(fontsize=8)
    return draw


def _fig_robustness(results: list[RunResult], cfg: Config, path: Path) -> None:
    tfs = cfg.timeframes
    fig, axes = plt.subplots(1, len(tfs), figsize=(5 * len(tfs), 3.6), squeeze=False)
    for ax, tf in zip(axes[0], tfs):
        for i, sym in enumerate(cfg.symbols):
            rs = sorted([r for r in results if r.tradable and r.symbol == sym and r.tf == tf and r.a_vs_b],
                        key=lambda r: r.depth_atr)
            if not rs:
                continue
            x = np.array([r.depth_atr for r in rs]) + (i - 1) * 0.006
            y = np.array([r.a_vs_b["delta_expectancy"] for r in rs])
            lo = np.array([r.a_vs_b["delta_ci"][0] for r in rs])
            hi = np.array([r.a_vs_b["delta_ci"][1] for r in rs])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=sym)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"ΔE = E[B] − E[A], {tf}")
        ax.set_xlabel("глубина свипа, ATR")
        ax.set_ylabel("R")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_by_year(results: list[RunResult], main: list[RunResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / max(len(main), 1)
    years = sorted({row["key"] for r in main for row in r.by_year})
    for i, r in enumerate(main):
        rows = {row["key"]: row for row in r.by_year}
        y = [rows[k]["E_B"] - rows[k]["E_A"] if k in rows else np.nan for k in years]
        ax.bar(np.arange(len(years)) + i * width, y, width, label=f"{r.symbol} {r.tf}")
    ax.set_xticks(np.arange(len(years)) + 0.4 - width / 2, years)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("ΔE = E[B] − E[A] по годам (основные параметры)")
    ax.set_ylabel("R")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- текст ----------

def build(cfg: Config, quality: list[dict], results: list[RunResult], events: pd.DataFrame,
          summary: dict, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    s, o = cfg.section("study"), cfg.section("outcomes")
    main_depth = cfg.section("events", "sweep")["depth_atr"]
    target = f"{o['primary_target']:g}"
    main = [r for r in results if r.tradable and r.depth_atr == main_depth]
    md: list[str] = []
    w = md.append

    w(f"# Отчёт: {s['name']} — свипы с ростом OI против свипов с падением OI\n")
    w(f"Сформирован автоматически {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
      f"Конфиг `{cfg.path}` (хэш `{meta['config_hash']}`), версия сырых данных `{meta['data_hash']}`, "
      f"коммит `{meta['git_commit']}`.\n")
    w("> Расчёт исследовательский, не торговая рекомендация.\n")

    w("## 1. Гипотеза и правило вердикта (зафиксированы до запуска)\n")
    w("Главная гипотеза этапа 1 (NIF): разворот после свипа определяется характером потока. "
      "**Группа A** — OI к моменту решения вырос (новые позиции) → ожидается продолжение. "
      "**Группа B** — OI упал (закрытия, стопы, ликвидации) → ожидается разворот. "
      "В обеих группах одна и та же сделка — реверсия против свипа. Проверяется **ΔE = E[B] − E[A] > 0**.\n")
    w(f"- Основная метрика: исход «цель {target}R / стоп 1R / выход через {o['horizon_bars']} баров», "
      f"в R после издержек ({cfg.cost_pct_round_trip:.2%} за круг).")
    w(f"- «Подтверждено» для пары монета×ТФ: ΔE ≥ {s['practical_min_delta_r']}R, нижняя граница 95% CI(ΔE) > 0, "
      f"p после поправки Бенджамини–Хохберга < {s['alpha']}.")
    w("- Гипотеза поддержана в целом, если подтверждена на ETH **и** SOL на одном ТФ при основных параметрах "
      "и минимум в 3 из 4 значений глубины свипа.\n")

    w("## 2. Итог\n")
    verdict = "**ПОДДЕРЖАНА**" if summary["supported"] else "**НЕ ПОДДЕРЖАНА**"
    w(f"Главная гипотеза по пререгистрированному правилу: {verdict}.\n")
    rows = []
    for tf, v in summary["by_tf"].items():
        for sym, p in v["per_symbol"].items():
            rows.append([sym, tf, p["main"], f"{p['confirmed_depths']} из {p['n_depths']}"])
    w(_table(["Монета", "ТФ", "Вердикт (основные параметры)", "Подтверждено по глубинам"], rows) + "\n")
    trad = [[r.symbol, r.tf, _f(r.groups["B"]["expectancy"]), _f(r.groups["B"]["expectancy_ci"]),
             "да" if r.groups["B"]["expectancy_ci"][0] > 0 else "нет"] for r in main]
    w("Торгуемость самой реверсии в группе B (отдельно от разницы групп):\n")
    w(_table(["Монета", "ТФ", "E[B], R", "95% CI", "CI > 0"], trad) + "\n")

    w("## 3. Данные\n")
    w(f"Источник — публичный архив Binance USDⓈ-M futures. Задержка публикации OI: "
      f"{cfg.section('availability')['oi_lag_minutes']} мин после метки снимка; OI старше "
      f"{cfg.section('availability')['oi_max_staleness_minutes']} мин считается неизвестным.\n")
    qrows = [[q["symbol"], q["dataset"], q["first"] or "—", q["last"] or "—", q["rows"],
              _f(q["coverage_pct"], 2), q["missing_rows"] if q["missing_rows"] is not None else "—",
              q["duplicates_dropped"], ", ".join(f"{k}={v}" for k, v in q["issues"].items() if v) or "—",
              q["verdict"]] for q in quality]
    w(_table(["Монета", "Набор", "Начало", "Конец", "Строк", "Покрытие %", "Пропущено", "Дублей",
              "Аномалии", "Вердикт"], qrows) + "\n")
    gaps = [[q["symbol"], q["dataset"], g[0], g[1]] for q in quality for g in q["largest_gaps"][:3]]
    if gaps:
        w("Крупнейшие разрывы:\n")
        w(_table(["Монета", "Набор", "Начало разрыва", "Длительность"], gaps) + "\n")
    _fig_coverage(quality, fig_dir / "coverage.png")
    w("![coverage](figures/coverage.png)\n")

    w("## 4. Методика\n")
    sw = cfg.section("events", "sweep")
    w(f"- **Свип**: уровень — экстремум предыдущих {sw['lookback']} баров, не моложе {sw['min_age']} баров; "
      f"вынос за уровень ≥ depth·ATR предыдущего бара (основное depth = {main_depth}); закрытие обратно за уровень; "
      f"не чаще одного события стороны за {sw['cooldown']} баров. Только прошлые данные.")
    w(f"- **OI**: ΔOI за {cfg.section('features', 'oi').get('window', 3)} бара по значениям, опубликованным "
      "до закрытия бара свипа.")
    w(f"- **Вход**: открытие минуты сразу после закрытия бара свипа. 1R = {o['r_atr']}·ATR. Путь — 1m-свечи; "
      "если цель и стоп в одной минуте — засчитывается стоп.")
    w("- **Контроль**: на каждое событие — случайный бар без свипа с тем же часом UTC и той же децилью "
      "процентиля ATR, та же сделка.")
    w(f"- **Статистика**: блочный бутстреп по дням ({s['bootstrap']} повторов) для 95% CI; выбор основного "
      f"теста по нормальности (перестановочный, {s['permutations']} перестановок, или Уэлч); "
      "Манн–Уитни и Колмогоров–Смирнов для распределений; χ²/Фишер для win rate; "
      "Cliff's delta и Cohen's d; поправка BH внутри семейства торгуемых прогонов.\n")

    w("## 5. Результаты по рынкам (основные параметры)\n")
    for r in main:
        w(f"### {r.symbol} {r.tf}\n")
        w(_table(["Счётчик", "Число"], [[k, v] for k, v in r.counts.items()]) + "\n")
        g = r.groups
        grows = [[LABELS.get(k, "все свипы (A+B)"), v["n"], v["n_days"], _f(v["win_rate"]), _f(v["expectancy"]),
                  _f(v["expectancy_ci"]), _f(v["median"]), _f(v["profit_factor"]), _f(v["mfe_mean"]),
                  _f(v["mae_mean"])] for k, v in g.items()]
        w(_table(["Группа", "N", "Дней", "Win rate", "Expectancy R", "95% CI", "Медиана", "PF",
                  "MFE ср.", "MAE ср."], grows) + "\n")
        for name, c in (("B против A", r.a_vs_b), ("все свипы против контроля", r.sweeps_vs_control)):
            if not c:
                w(f"{name}: недостаточно данных.\n")
                continue
            w(f"**{name}**: ΔE = {_f(c['delta_expectancy'], 4)} R, 95% CI {_f(c['delta_ci'], 4)}; "
              f"Δ win rate = {_f(c['delta_win_rate'])}; основной тест — {c['primary_test']} "
              f"({c['test_rationale']}): p = {_p(c['p_primary'])}"
              + (f", после BH = {_p(c['p_adjusted'])}" if name == "B против A" else "")
              + f"; Манн–Уитни p = {_p(c['p_mannwhitney'])}; KS p = {_p(c['p_ks'])}; "
              f"Welch p = {_p(c['p_welch'])}; win rate p = {_p(c['p_win_rate'])}; "
              f"Cliff's δ = {_f(c['cliffs_delta'])} ({c['cliffs_magnitude']}); Cohen's d = {_f(c['cohens_d'])}.\n")
        w(f"Вердикт: **{r.verdict}**.\n")

    w("## 6. Устойчивость к параметру свипа\n")
    rrows = [[r.symbol, r.tf, r.depth_atr, r.counts["group_A_oi_up"], r.counts["group_B_oi_down"],
              _f(r.a_vs_b["delta_expectancy"], 4) if r.a_vs_b else "—",
              _f(r.a_vs_b["delta_ci"], 4) if r.a_vs_b else "—",
              _p(r.a_vs_b["p_adjusted"]) if r.a_vs_b else "—", r.verdict]
             for r in results if r.tradable]
    w(_table(["Монета", "ТФ", "depth ATR", "N A", "N B", "ΔE R", "95% CI", "p (BH)", "Вердикт"], rrows) + "\n")
    _fig_robustness(results, cfg, fig_dir / "robustness.png")
    w("![robustness](figures/robustness.png)\n")

    w("## 7. Диагностика: OI без задержки публикации (не торгуемо)\n")
    w("Показывает, есть ли механизм, если бы OI был известен мгновенно. Если эффект виден только здесь, "
      "он существует, но до него не дотянуться с этими данными.\n")
    drows = [[r.symbol, r.tf, r.counts["group_A_oi_up"], r.counts["group_B_oi_down"],
              _f(r.a_vs_b["delta_expectancy"], 4) if r.a_vs_b else "—",
              _f(r.a_vs_b["delta_ci"], 4) if r.a_vs_b else "—",
              _p(r.a_vs_b["p_adjusted"]) if r.a_vs_b else "—", r.verdict]
             for r in results if not r.tradable]
    w(_table(["Монета", "ТФ", "N A", "N B", "ΔE R", "95% CI", "p (BH)", "Вердикт"], drows) + "\n")

    w("## 8. Стабильность: по годам и по сторонам свипа\n")
    yrows = [[r.symbol, r.tf, row["key"], row["n_A"], row["n_B"], _f(row["E_A"]), _f(row["E_B"]),
              _f(row["E_B"] - row["E_A"])] for r in main for row in r.by_year + r.by_side]
    w(_table(["Монета", "ТФ", "Год / сторона", "N A", "N B", "E[A]", "E[B]", "ΔE"], yrows) + "\n")
    _fig_by_year(results, main, fig_dir / "by_year.png")
    w("![by_year](figures/by_year.png)\n")

    w("## 9. Распределения\n")
    if not events.empty:
        for tf in cfg.timeframes:
            for name, title, draw in (
                (f"dist_{tf}", f"Распределение исходов (цель {target}R), R после издержек",
                 _draw_dist(f"res_{target}r_net")),
                (f"excursions_{tf}", "MFE и MAE за горизонт, R", _draw_excursions),
                (f"time_{tf}", "Время до исхода", _draw_time(target)),
            ):
                _per_symbol_grid(events, cfg.symbols, tf, fig_dir / f"{name}.png", title, draw)
                w(f"![{name}](figures/{name}.png)\n")
        sec = events.groupby(["symbol", "tf", "group"])[[f"res_{k:g}r_net" for k in o["targets"]]].mean()
        w("Ожидание при других целях (стоп 1R), R после издержек:\n")
        w(_table(["Монета", "ТФ", "Группа", *[f"цель {k:g}R" for k in o["targets"]]],
                 [[*idx, *(_f(v) for v in row)] for idx, row in sec.iterrows()]) + "\n")

    w("## 10. Вывод\n")
    w(_conclusion(summary, main, s["practical_min_delta_r"]) + "\n")

    w("## 11. Ограничения\n")
    w("\n".join(f"- {x}" for x in LIMITATIONS) + "\n")
    w("## 12. Что проверить следующим\n")
    w("\n".join(f"- {x}" for x in _next_steps(summary, results)) + "\n")

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


LIMITATIONS = [
    "OI только Binance и снимками раз в 5 минут; задержка публикации задана консервативно, а не измерена.",
    "ΔOI берётся за окно до момента решения, а не точно за бар свипа: на 5m часть «OI на свипе» ещё не опубликована.",
    "Ликвидации не используются (нет бесплатной истории); падение OI — лишь их прокси.",
    "Сессии по фиксированным часам UTC без учёта перехода на летнее время.",
    "Издержки — фиксированная модель (тейкер + проскальзывание), без зависимости от глубины стакана.",
    "Одна сделка на событие без управления позицией (частичные выходы, безубыток — этап 2.2+).",
    "ETH, SOL и BTC сильно коррелированы: подтверждения на разных монетах не независимы.",
    "Перестановочный тест предполагает взаимозаменяемость событий; зависимость внутри дня учтена только в бутстреп-интервалах.",
    "Разбивки на обучающий/проверочный/отложенный периоды ещё нет: правило проверялось на всей истории сразу (параметры не подбирались).",
]


def _conclusion(summary: dict, main: list[RunResult], practical: float) -> str:
    parts = []
    for r in main:
        c = r.a_vs_b
        if not c:
            parts.append(f"- {r.symbol} {r.tf}: данных недостаточно.")
            continue
        practical_ok = abs(c["delta_expectancy"]) >= practical
        parts.append(f"- {r.symbol} {r.tf}: ΔE = {c['delta_expectancy']:+.4f}R "
                     f"(CI {_f(c['delta_ci'], 4)}), Cliff's δ {c['cliffs_delta']:+.3f} ({c['cliffs_magnitude']}); "
                     f"практически {'значимо' if practical_ok else 'незначимо'} → {r.verdict}.")
    head = ("Главная гипотеза по заранее зафиксированному правилу **поддержана** данными."
            if summary["supported"] else
            "Главная гипотеза по заранее зафиксированному правилу **не поддержана**. Параметры после просмотра "
            "результатов не менялись — это честный результат.")
    return head + "\n\n" + "\n".join(parts)


def _next_steps(summary: dict, results: list[RunResult]) -> list[str]:
    diag = [r for r in results if not r.tradable and r.verdict == "подтверждено"]
    steps = []
    if summary["supported"]:
        steps.append("Разбить историю на обучающий/проверочный/отложенный периоды и повторить проверку на отложенном.")
        steps.append("Построить симуляцию сделок с управлением позицией и лимитным входом (этап 2.2–2.6).")
    else:
        steps.append("Не подстраивать параметры под результат. Записать отрицательный результат и перейти к следующей "
                     "гипотезе этапа 1 (SMT против BTC, перп против спота).")
    if diag:
        steps.append("Эффект виден при нулевой задержке OI: проверить источник OI с меньшей задержкой "
                     "(запись потока в реальном времени) и купить историю ликвидаций.")
    steps.append("Проверить эталонную реверсию без условия на OI (К36) — обыгрывают ли свипы её.")
    steps.append("Добавить ликвидации как отдельный признак, когда появится история.")
    return steps
