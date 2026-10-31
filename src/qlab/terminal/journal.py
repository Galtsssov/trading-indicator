"""Журнал сделок и персональная аналитика.

Сделка хранит всё, что было известно в момент входа: факты панели, признаки последнего закрытого бара,
события, режим, скриншот графика. Результат считается в R: 1R = |вход − стоп|; издержки вычитаются.
Аналитика честно показывает размер выборки и неопределённость: «улучшает» только то, что выдерживает
бутстреп-интервал; остальное — «наблюдать».
"""
from __future__ import annotations

import base64
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('long', 'short')),
    opened_at TEXT NOT NULL,
    entry REAL NOT NULL,
    stop REAL NOT NULL,
    target REAL,
    setup TEXT,
    note TEXT,
    context TEXT NOT NULL,
    screenshot TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    closed_at TEXT,
    exit REAL,
    exit_reason TEXT,
    result_r REAL,
    mfe_r REAL DEFAULT 0,
    mae_r REAL DEFAULT 0
);
"""


def _now(ts: pd.Timestamp | None = None) -> str:
    return (ts or pd.Timestamp(datetime.now(timezone.utc))).isoformat()


class Journal:
    def __init__(self, path: Path, shots_dir: Path, cost_pct_round_trip: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        shots_dir.mkdir(parents=True, exist_ok=True)
        self.shots = shots_dir
        self.cost = cost_pct_round_trip
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    # ---------- запись ----------
    def open(self, *, mode: str, symbol: str, tf: str, side: str, entry: float, stop: float, target: float | None,
             setup: str, note: str, context: dict, screenshot_data_url: str | None, opened_at: pd.Timestamp) -> int:
        if side not in ("long", "short"):
            raise ValueError("side должен быть long или short")
        risk = (entry - stop) if side == "long" else (stop - entry)
        if risk <= 0:
            raise ValueError("стоп должен быть ниже входа для лонга и выше входа для шорта")
        if target is not None and ((target - entry) if side == "long" else (entry - target)) <= 0:
            raise ValueError("цель должна быть по направлению сделки")
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO trades (mode, symbol, tf, side, opened_at, entry, stop, target, setup, note, context) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (mode, symbol, tf, side, _now(opened_at), entry, stop, target, setup.strip() or None, note.strip() or None,
                 json.dumps(context, ensure_ascii=False, default=str)))
            tid = cur.lastrowid
            if screenshot_data_url and screenshot_data_url.startswith("data:image/png;base64,"):
                p = self.shots / f"{tid}.png"
                p.write_bytes(base64.b64decode(screenshot_data_url.split(",", 1)[1]))
                self._db.execute("UPDATE trades SET screenshot=? WHERE id=?", (p.name, tid))
            self._db.commit()
        return int(tid)

    def _result_r(self, t: sqlite3.Row, exit_price: float) -> float:
        risk = abs(t["entry"] - t["stop"])
        sign = 1 if t["side"] == "long" else -1
        cost_r = self.cost * t["entry"] / risk
        return sign * (exit_price - t["entry"]) / risk - cost_r

    def close(self, tid: int, exit_price: float, reason: str, at: pd.Timestamp | None = None) -> None:
        with self._lock:
            t = self._db.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
            if t is None or t["status"] != "open":
                raise ValueError(f"сделка {tid} не открыта")
            self._db.execute("UPDATE trades SET status='closed', closed_at=?, exit=?, exit_reason=?, result_r=? WHERE id=?",
                             (_now(at), exit_price, reason, self._result_r(t, exit_price), tid))
            self._db.commit()

    def update_open(self, symbol: str, m1: pd.DataFrame, now: pd.Timestamp) -> list[int]:
        """Сопровождение открытых сделок по минуткам: MFE/MAE, авто-закрытие по стопу или цели.
        Если стоп и цель в одной минуте — засчитывается стоп (консервативно)."""
        closed = []
        for t in self._db.execute("SELECT * FROM trades WHERE status='open' AND symbol=?", (symbol,)).fetchall():
            path = m1[m1["ts"] >= pd.Timestamp(t["opened_at"]).floor("min")]
            if path.empty:
                continue
            risk = abs(t["entry"] - t["stop"])
            long = t["side"] == "long"
            fav = ((path["high"] - t["entry"]) if long else (t["entry"] - path["low"])) / risk
            adv = ((t["entry"] - path["low"]) if long else (path["high"] - t["entry"])) / risk
            with self._lock:
                self._db.execute("UPDATE trades SET mfe_r=?, mae_r=? WHERE id=?", (float(fav.max()), float(adv.max()), t["id"]))
                self._db.commit()
            stop_hit = (path["low"] <= t["stop"]) if long else (path["high"] >= t["stop"])
            tgt_hit = ((path["high"] >= t["target"]) if long else (path["low"] <= t["target"])) if t["target"] else stop_hit & False
            first_stop = stop_hit.idxmax() if stop_hit.any() else None
            first_tgt = tgt_hit.idxmax() if tgt_hit.any() else None
            if first_stop is not None and (first_tgt is None or first_stop <= first_tgt):
                self.close(t["id"], t["stop"], "stop", path.loc[first_stop, "ts"])
                closed.append(t["id"])
            elif first_tgt is not None:
                self.close(t["id"], t["target"], "target", path.loc[first_tgt, "ts"])
                closed.append(t["id"])
        return closed

    def update_note(self, tid: int, setup: str | None, note: str | None) -> None:
        with self._lock:
            self._db.execute("UPDATE trades SET setup=COALESCE(?, setup), note=COALESCE(?, note) WHERE id=?", (setup, note, tid))
            self._db.commit()

    def delete(self, tid: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM trades WHERE id=?", (tid,))
            self._db.commit()

    # ---------- чтение ----------
    def list(self, mode: str | None = None) -> list[dict]:
        q = "SELECT * FROM trades" + (" WHERE mode=?" if mode else "") + " ORDER BY id DESC"
        rows = self._db.execute(q, (mode,) if mode else ()).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d["context"])
            out.append(d)
        return out

    def frame(self, mode: str | None = None) -> pd.DataFrame:
        """Закрытые сделки + плоский контекст (условия на момент входа) для аналитики."""
        rows = [t for t in self.list(mode) if t["status"] == "closed" and t["result_r"] is not None]
        if not rows:
            return pd.DataFrame()
        recs = []
        for t in rows:
            c = t["context"].get("conditions", {})
            recs.append({"id": t["id"], "symbol": t["symbol"], "tf": t["tf"], "side": t["side"],
                         "setup": t["setup"] or "без метки", "r": t["result_r"], "mfe": t["mfe_r"], "mae": t["mae_r"],
                         "exit_reason": t["exit_reason"], "opened_at": t["opened_at"], **c})
        return pd.DataFrame(recs)


# ---------- аналитика ----------

def _stats(r: np.ndarray) -> dict:
    wins, losses = r[r > 0], r[r <= 0]
    return {"n": int(len(r)), "win_rate": float((r > 0).mean()) if len(r) else None,
            "expectancy": float(r.mean()) if len(r) else None, "total_r": float(r.sum()),
            "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None}


def _boot_diff(a: np.ndarray, b: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    lo, hi = np.percentile(da - db, [5, 95])
    return float(lo), float(hi)


GROUP_COLUMNS = {"setup": "Сетап", "side": "Направление", "symbol": "Монета", "tf": "Таймфрейм", "session": "Сессия",
                 "regime": "Режим", "volatility": "Волатильность", "btc": "BTC", "event": "Событие перед входом",
                 "hour": "Час входа (UTC)"}


def analyze(df: pd.DataFrame, min_group: int = 5, min_filter: int = 10) -> dict:
    """Что работает лично у меня: группы, ошибки, фильтры. Всё — с размером выборки."""
    if df.empty:
        return {"n": 0, "message": "Пока нет закрытых сделок."}
    rng = np.random.default_rng(0)
    r = df["r"].to_numpy(float)
    out: dict = {"n": len(df), "overall": _stats(r), "equity": np.round(np.cumsum(r[::-1]), 3).tolist(),
                 "groups": {}, "weak_spots": [], "filters": []}
    for col, title in GROUP_COLUMNS.items():
        if col not in df:
            continue
        rows = []
        for val, g in df.groupby(col, dropna=True):
            s = _stats(g["r"].to_numpy(float))
            rows.append({"value": VALUE_RU.get(str(val), str(val)), **s, "enough": s["n"] >= min_group})
        out["groups"][title] = sorted(rows, key=lambda x: -(x["expectancy"] or 0))
        if len(rows) < 2:                    # все сделки в одной группе — сравнивать не с чем
            continue
        for row in rows:
            if row["enough"] and row["expectancy"] is not None and row["expectancy"] < 0:
                out["weak_spots"].append({"where": f"{title}: {row['value']}", **row})
    out["weak_spots"].sort(key=lambda x: x["expectancy"])
    flags = [c for c in df.columns if c.startswith("is_")]
    for c in flags:
        mask = df[c].astype(bool).to_numpy()
        a, b = r[mask], r[~mask]
        if len(a) < 3 or len(b) < 3:
            continue
        lo, hi = _boot_diff(a, b, rng)
        verdict = ("улучшает" if lo > 0 and len(a) >= min_filter else "ухудшает" if hi < 0 and len(a) >= min_filter
                   else "наблюдать")
        out["filters"].append({"condition": CONDITION_NAMES.get(c, c), "n_with": len(a), "n_without": len(b),
                               "exp_with": float(a.mean()), "exp_without": float(b.mean()),
                               "delta": float(a.mean() - b.mean()), "ci90": [lo, hi], "verdict": verdict})
    out["filters"].sort(key=lambda x: -x["delta"])
    out["note"] = ("Интервалы — 90% бутстреп. «Улучшает» — только если интервал разницы выше нуля и сделок с условием "
                   f"не меньше {min_filter}. При малом числе сделок почти всё будет «наблюдать» — это честно.")
    return out


VALUE_RU = {"trend": "тренд", "range": "флэт", "high_vol": "высокая волатильность", "low_vol": "низкая волатильность",
            "btc_strong": "BTC вырос за сутки", "btc_weak": "BTC упал за сутки", "long": "лонг", "short": "шорт"}

CONDITION_NAMES = {
    "is_trend": "Режим тренда", "is_high_vol": "Высокая волатильность", "is_btc_strong": "BTC вырос за сутки",
    "is_risk_on": "Risk on (BTC выше средней 20 дней)", "is_with_trend": "Сделка по направлению тренда",
    "is_above_vwap": "Цена выше VWAP", "is_with_vwap": "Сделка по стороне VWAP (лонг выше / шорт ниже)",
    "is_after_sweep": "Был свип за последние 6 баров", "is_us_session": "Сессия США",
    "is_with_aggressors": "Агрессоры на стороне сделки (6 баров)", "is_funding_crowded": "Funding |>0.05%| (перекос толпы)",
    "is_oi_rising": "OI растёт за час", "is_rr_2": "Цель ≥ 2R",
}
