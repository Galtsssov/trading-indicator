"""Отчёт Information Ceiling Study: собирает результаты всех пар монета × ТФ и выносит вердикт по правилу
из docs/stage-3-preregistration.md. Ничего не пересчитывает."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qlab import md  # noqa: E402
from qlab.ceiling import feature_groups  # noqa: E402

VERDICT_NO = ("В текущем информационном пространстве (OHLCV + производные + OI + Funding + Long/Short + "
              "внутрибаровые признаки) достаточного сигнала для решения поставленной задачи не обнаружено.")


def load(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*_*.json"))]


def verdict(results: list[dict], primary_model: str) -> tuple[bool, list[list]]:
    rows, ok = [], False
    for r in results:
        prim = next(iter(r["configs"].values()))
        sane = prim.get("sanity", {}).get("passed", False)
        for k, s in prim["models"][primary_model]["top_k"].items():
            hit = s.get("n", 0) > 0 and s["wr_lo"] > 0.5 and s["ev_lo"] > 0
            ok |= hit and sane
            rows.append([r["symbol"], r["tf"], k, s.get("n", 0), md.f(s.get("wr")), md.f(s.get("wr_lo")),
                         md.f(s.get("ev")), md.f(s.get("ev_lo")), md.f(prim["breakeven_wr"]),
                         "да" if hit and sane else "нет"])
    return ok, rows


def _figures(results: list[dict], out: Path, primary_model: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    for ax, r in zip(axes[0], results):
        lc = pd.DataFrame(next(iter(r["configs"].values()))["learning_curve"])
        ax.plot(lc["rows"], lc["val_auc"], "o-", label="validation")
        ax.plot(lc["rows"], lc["train_auc"], "s--", label="training (in-sample)")
        ax.axhline(0.5, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_title(f"{r['symbol']} {r['tf']}")
        ax.set_xlabel("строк обучения")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=7)
    fig.suptitle("Кривые обучения (LightGBM): упирается ли модель в объём данных")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), squeeze=False)
    for ax, r in zip(axes[0], results):
        cal = next(iter(r["configs"].values()))["models"][primary_model]["calibration"]
        for d, c in cal.items():
            ax.plot(c["mean_p"], c["observed"], "o-", label=f"{d} (ECE {c['ece']:.3f})")
        lim = [0, max(max(c["mean_p"] + c["observed"]) for c in cal.values()) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_title(f"{r['symbol']} {r['tf']}")
        ax.set_xlabel("предсказанная вероятность")
        ax.set_ylabel("наблюдаемая доля целей")
        ax.legend(fontsize=7)
    fig.suptitle("Калибровка основного прибора на validation (децили вероятности)")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=100)
    plt.close(fig)

    abl = pd.DataFrame({f"{r['symbol']} {r['tf']}": {g: v["delta"] for g, v in next(iter(r["configs"].values()))["ablation"].items()}
                        for r in results})
    ax = abl.sort_values(abl.columns[0]).plot.barh(figsize=(8, 5))
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("изменение AUC на validation при удалении группы")
    ax.set_title("Абляция: сколько информации теряется без группы признаков")
    plt.tight_layout()
    plt.savefig(out / "ablation.png", dpi=100)
    plt.close()


def build(results: list[dict], cfg_ceiling: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pm = cfg_ceiling["primary_model"]
    results = sorted(results, key=lambda r: (r["symbol"], r["tf"]))
    _figures(results, out, pm)
    ok, vrows = verdict(results, pm)
    w = ["# Information Ceiling Study: сколько информации в текущих данных\n",
         "Правило решения зафиксировано до запуска: `docs/stage-3-preregistration.md`. Training → validation, "
         "holdout не открывался. Модели — измерительные приборы, не стратегии.\n",
         "## Ответ\n",
         ("**ДОСТАТОЧНО:** правило выполнено (см. таблицу)." if ok else f"**НЕДОСТАТОЧНО.** {VERDICT_NO}") + "\n",
         "## 1. Решающая таблица: лучшие сигналы дня, основной прибор (LightGBM), основная конфигурация\n",
         "Цель 2R, стоп 1R, 1R = 2 ATR, издержки 0.14 % за круг. «Нижняя граница» — односторонний бутстреп по дням "
         "с поправкой Бонферрони (0.05/12). Требование: нижняя граница WR > 0.5 **и** нижняя граница EV > 0.\n",
         md.table(["Монета", "ТФ", "Сигналов в день", "Сделок", "WR", "WR, нижн.", "EV, R", "EV, нижн.",
                   "Безубыточный WR", "Выполнено"], vrows) + "\n"]

    w.append("## 2. Верхняя граница: все модели, все признаки\n")
    rows = []
    for r in results:
        for key, cfg in r["configs"].items():
            for name, m in cfg["models"].items():
                a = m["auc"]
                t2 = m["top_k"].get("2", {})
                rows.append([r["symbol"], r["tf"], key.replace("r", "1R = ") + " ATR", name,
                             f"{a['long'][0]:.4f} [{a['long'][1]:.3f}; {a['long'][2]:.3f}]",
                             f"{a['short'][0]:.4f} [{a['short'][1]:.3f}; {a['short'][2]:.3f}]",
                             md.f(t2.get("wr")), md.f(t2.get("ev")), md.f(cfg["breakeven_wr"]),
                             md.f(cfg["base_rate"]["long"])])
    w.append(md.table(["Монета", "ТФ", "Конфигурация", "Модель", "AUC long [95% CI]", "AUC short [95% CI]",
                       "WR топ-2/день", "EV топ-2, R", "Безубыточный WR", "Доля целей (база)"], rows) + "\n")
    w.append("Случайный выбор 2 сигналов в день для сравнения (та же процедура, случайные оценки):\n")
    w.append(md.table(["Монета", "ТФ", "Конфигурация", "WR", "EV, R"],
                      [[r["symbol"], r["tf"], key, md.f(c["random_picks"]["2"].get("wr")), md.f(c["random_picks"]["2"].get("ev"))]
                       for r in results for key, c in r["configs"].items()]) + "\n")
    w.append("Точность на самых уверенных долях validation (основной прибор, long):\n")
    w.append(md.table(["Монета", "ТФ", "Топ-доля", "N", "Precision", "Recall", "EV, R"],
                      [[r["symbol"], r["tf"], f"{x['top_frac']:.1%}", x["n"], md.f(x["precision"]), md.f(x["recall"]), md.f(x["ev"])]
                       for r in results for x in next(iter(r["configs"].values()))["models"][pm]["precision_at"]["long"]]) + "\n")

    w.append("## 3. Кривые обучения\n")
    w.append("Если AUC на validation растёт с объёмом обучения — модель упирается в данные; если плато — в отсутствие информации.\n")
    w.append(md.table(["Монета", "ТФ", "Доля", "Строк", "AUC val", "AUC train"],
                      [[r["symbol"], r["tf"], x["fraction"], x["rows"], md.f(x["val_auc"], 4), md.f(x["train_auc"], 4)]
                       for r in results for x in next(iter(r["configs"].values()))["learning_curve"]]) + "\n")
    w.append("![кривые обучения](learning_curves.png)\n")

    w.append("## 4. Абляция: сколько информации теряется без группы\n")
    first = next(iter(results[0]["configs"].values()))
    groups = list(first["ablation"])
    w.append(md.table(["Группа", "Признаков", *[f"Δ AUC {r['symbol'][:3]} {r['tf']}" for r in results]],
                      [[g, first["ablation"][g]["n_features"],
                        *[f"{next(iter(r['configs'].values()))['ablation'][g]['delta']:+.4f}" for r in results]] for g in groups]) + "\n")
    w.append("![абляция](ablation.png)\n")

    w.append("## 5. Важность признаков: четыре метода\n")
    for r in results:
        imp = next(iter(r["configs"].values()))["importance"]
        ranks = pd.DataFrame({k: pd.Series(v).rank(ascending=False) for k, v in imp.items() if isinstance(v, dict)})
        corr = ranks.corr(method="spearman")
        top = {k: pd.Series(v).sort_values(ascending=False).head(10).index.tolist() for k, v in imp.items() if isinstance(v, dict)}
        w.append(f"### {r['symbol']} {r['tf']}\n")
        w.append("Согласие методов (ранговая корреляция по признакам, где посчитаны оба):\n")
        w.append(md.table(["", *corr.columns], [[i, *[md.f(x, 2) for x in row]] for i, row in corr.iterrows()]) + "\n")
        w.append(md.table(["Ранг", *top.keys()], [[i + 1, *[f"`{top[k][i]}`" if i < len(top[k]) else "" for k in top]]
                                                  for i in range(10)]) + "\n")
        grp = feature_groups(list(imp["shap"]))
        shap = pd.Series(imp["shap"])
        share = {g: shap[c].sum() / shap.sum() for g, c in grp.items()}
        w.append("Доля SHAP по группам: " + ", ".join(f"{g} {v:.0%}" for g, v in sorted(share.items(), key=lambda x: -x[1])) + "\n")

    w.append("## 6. Калибровка\n")
    w.append(md.table(["Монета", "ТФ", "Направление", "Доля целей", "Brier", "ECE"],
                      [[r["symbol"], r["tf"], d, md.f(c["base_rate"]), md.f(c["brier"], 4), md.f(c["ece"], 4)]
                       for r in results for d, c in next(iter(r["configs"].values()))["models"][pm]["calibration"].items()]) + "\n")
    w.append("![калибровка](calibration.png)\n")

    w.append("## 7. Проверки исправности\n")
    w.append(md.table(["Монета", "ТФ", "AUC на перемешанных метках", "AUC со встроенным эффектом",
                       "AUC без знания эффекта", "Ранг встроенной колонки (SHAP)", "Лучший ранг случайного признака",
                       "Пройдено"],
                      [[r["symbol"], r["tf"], md.f(s["shuffled_auc"], 4), md.f(s["planted_auc"], 4),
                        md.f(s["planted_auc_without_effect_model"], 4), s["planted_shap_rank"], s["noise_best_shap_rank"],
                        "да" if s["passed"] else "нет"]
                       for r in results for s in [next(iter(r["configs"].values()))["sanity"]]]) + "\n")
    path = out / "REPORT.md"
    path.write_text("\n".join(w), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps({"sufficient": ok}, indent=1))
    return path
