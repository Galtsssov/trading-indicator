"""Генерация каталога признаков docs/FEATURES.md из спецификаций (описание всегда совпадает с кодом)."""
from __future__ import annotations

from pathlib import Path

from qlab import md
from qlab.factory import catalog

HEADER = """# Каталог признаков Feature Factory

Файл сгенерирован командой `python -m qlab catalog` из спецификаций в `src/qlab/factory/f_*.py`.
Не редактировать вручную — тест `test_catalog_is_up_to_date` проверяет совпадение с кодом.

**Обозначения:** o, h, l, c, v — OHLCV бара; r = ln(c_t / c_(t−1)); a = ATR(14) предыдущего бара;
W — окно в барах (на 5m: 6 = 30 мин, 24 = 2 ч, 96 = 8 ч, 288 = сутки; на 15m окна в 3 раза длиннее по времени);
Σ_W, mean_W, std_W, max_W — по барам t−W+1..t; z_W(x) = (x − mean_W x) / std_W x; X — монета контекста.
Каждый признак использует только бары ≤ t и известен на закрытии бара t (проверяется тестом причинности для всех).
Корреляции, дубли и кластеры на реальных данных — в `reports/factory/REPORT.md`.
"""


def render() -> str:
    cat = catalog()
    out = [HEADER, f"Всего признаков: **{len(cat)}**, категорий: **{cat['category'].nunique()}**.\n"]
    counts = cat.groupby("category", sort=False).size()
    out.append(md.table(["Категория", "Признаков"], [[c, n] for c, n in counts.items()]) + "\n")
    for category, g in cat.groupby("category", sort=False):
        out.append(f"## {category}\n")
        out.append(md.table(["ID", "Название", "Формула", "Тип", "Размерность", "Зависимости", "Стоимость", "Механизм"],
                            [[r.id, f"`{r.name}`", r.formula, r.kind, r.unit, r.deps, r.cost, r.mechanism]
                             for r in g.itertuples()]) + "\n")
        out.append("**Объяснение:**\n")
        out.append(md.table(["ID", "Что реально измеряет", "Чем может быть полезен", "Ограничения"],
                            [[r.id, r.measures, r.why, r.limits] for r in g.itertuples()]) + "\n")
    return "\n".join(out)


def write(path: Path = Path("docs/FEATURES.md")) -> Path:
    path.write_text(render(), encoding="utf-8")
    return path
