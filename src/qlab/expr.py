"""Безопасный язык условий для YAML-гипотез. Никакого eval: разбираем AST и считаем векторно по колонкам.

Разрешено: имена колонок, числа, строки, списки констант; and / or / not; == != < <= > >= in, not in;
+ − * /; унарный минус; abs(x). NaN в сравнениях даёт False (строка не проходит фильтр).
Пример: "oi_change < -0.005 and regime_vol == 'high_vol' and abs(vwap_dist_atr) < 0.5"
"""
from __future__ import annotations

import ast
import operator
from functools import reduce

import numpy as np
import pandas as pd

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}


def names(expr: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name) and n.id != "abs"}


def evaluate(expr: str, df: pd.DataFrame) -> pd.Series:
    out = _ev(ast.parse(expr, mode="eval").body, df)
    if np.isscalar(out):
        return pd.Series(bool(out), index=df.index)
    return pd.Series(out, index=df.index).fillna(False).astype(bool)


def _const_list(node: ast.AST) -> list:
    if not isinstance(node, (ast.List, ast.Tuple)) or not all(isinstance(e, ast.Constant) for e in node.elts):
        raise ValueError("после `in` ожидается список констант")
    return [e.value for e in node.elts]


def _ev(node: ast.AST, df: pd.DataFrame):
    match node:
        case ast.BoolOp(op=ast.And(), values=vals):
            return reduce(operator.and_, (_bool(_ev(v, df)) for v in vals))
        case ast.BoolOp(op=ast.Or(), values=vals):
            return reduce(operator.or_, (_bool(_ev(v, df)) for v in vals))
        case ast.UnaryOp(op=ast.Not(), operand=v):
            return ~_bool(_ev(v, df))
        case ast.UnaryOp(op=ast.USub(), operand=v):
            return -_ev(v, df)
        case ast.BinOp(left=l, op=op, right=r) if type(op) in _BIN:
            return _BIN[type(op)](_ev(l, df), _ev(r, df))
        case ast.Compare(left=left, ops=ops, comparators=comps):
            result, cur = None, _ev(left, df)
            for op, comp in zip(ops, comps):
                if isinstance(op, (ast.In, ast.NotIn)):
                    part = pd.Series(cur, index=df.index).isin(_const_list(comp))
                    part = ~part if isinstance(op, ast.NotIn) else part
                    nxt = None
                elif type(op) in _CMP:
                    nxt = _ev(comp, df)
                    part = _CMP[type(op)](cur, nxt)
                else:
                    raise ValueError(f"оператор сравнения {type(op).__name__} не разрешён")
                result = _bool(part) if result is None else result & _bool(part)
                cur = nxt
            return result
        case ast.Call(func=ast.Name(id="abs"), args=[arg], keywords=[]):
            return abs(_ev(arg, df))
        case ast.Name(id=name):
            if name not in df.columns:
                raise ValueError(f"неизвестный признак {name!r}")
            return df[name]
        case ast.Constant(value=v) if isinstance(v, (int, float, str, bool)):
            return v
        case _:
            raise ValueError(f"конструкция не разрешена: {ast.dump(node)[:80]}")


def _bool(x) -> pd.Series:
    return pd.Series(x).fillna(False).astype(bool) if not np.isscalar(x) else x
