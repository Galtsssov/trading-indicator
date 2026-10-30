"""Помощники для markdown-отчётов."""
from __future__ import annotations

from typing import Any

import numpy as np


def f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (tuple, list)):
        return f"[{f(x[0], nd)}; {f(x[1], nd)}]"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def p(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else ("<0.0001" if x < 1e-4 else f"{x:.4f}")


def table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
