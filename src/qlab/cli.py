"""Командная строка: `python -m qlab all` — весь конвейер без ручного вмешательства.

Шаги: fetch (скачать архив и собрать Parquet) → quality (отчёт о качестве) → study (исследование и отчёт).
Каждый прогон исследования дописывается в журнал reports/ledger.jsonl (хранится в git).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from qlab import report, study
from qlab.config import Config
from qlab.data import binance, quality

log = logging.getLogger("qlab")
DATASETS = ("klines_1m", "metrics", "funding")


def _paths(cfg: Config) -> tuple[Path, Path]:
    return cfg.data_dir / "raw", cfg.data_dir / "clean"


def cmd_fetch(cfg: Config) -> None:
    raw, clean = _paths(cfg)
    d = cfg.section("data")
    files = [f for s in cfg.symbols for f in binance.plan(d["base_url"], raw, s, d["start"], date.today())]
    log.info("файлов в плане: %d", len(files))
    log.info("скачивание: %s", binance.fetch(files, d["workers"]))
    dups = {f"{ds}/{s}": binance.build_clean(raw, clean, ds, s) for ds in DATASETS for s in cfg.symbols}
    (clean / "build_stats.json").write_text(json.dumps(dups, indent=1))


def _load(clean: Path, ds: str, symbol: str) -> pd.DataFrame | None:
    p = clean / ds / f"{symbol}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def cmd_quality(cfg: Config) -> list[dict]:
    _, clean = _paths(cfg)
    stats_path = clean / "build_stats.json"
    dups = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    out = [quality.check(ds, s, _load(clean, ds, s), dups.get(f"{ds}/{s}") or 0).as_dict()
           for s in cfg.symbols for ds in DATASETS]
    (cfg.data_dir / "quality.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for q in out:
        log.info("%-8s %-9s %s → %s  покрытие=%s  %s", q["symbol"], q["dataset"], q["first"], q["last"],
                 q["coverage_pct"], q["verdict"])
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_study(cfg: Config, quality_rows: list[dict]) -> Path:
    raw, clean = _paths(cfg)
    m1 = {s: _load(clean, "klines_1m", s) for s in cfg.symbols}
    missing = [s for s, df in m1.items() if df is None]
    if missing:
        raise SystemExit(f"нет свечей для {missing}: сначала `python -m qlab fetch`")
    inp = study.Inputs(m1=m1, oi={s: _load(clean, "metrics", s) for s in cfg.symbols},
                       funding={s: _load(clean, "funding", s) for s in cfg.symbols})
    results, events = study.run(cfg, inp)
    summary = study.overall(results, cfg)
    meta = {"config_hash": cfg.hash(), "data_hash": binance.raw_manifest_hash(raw), "git_commit": _git_commit()}
    name = cfg.section("study")["name"]
    out_dir = Path("reports") / name
    path = report.build(cfg, quality_rows, results, events, summary, meta, out_dir)
    (out_dir / "results.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "runs": [dataclasses.asdict(r) for r in results]},
        indent=1, ensure_ascii=False, default=str))
    (out_dir / "quality.json").write_text(json.dumps(quality_rows, indent=1, ensure_ascii=False))
    with (out_dir.parent / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "study": name, **meta,
                            "n_runs": len(results), "supported": summary["supported"]}, ensure_ascii=False) + "\n")
    log.info("отчёт: %s", path)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qlab", description=__doc__)
    ap.add_argument("command", choices=["fetch", "quality", "study", "all"])
    ap.add_argument("--config", default="configs/mvp.toml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if args.command in ("fetch", "all"):
        cmd_fetch(cfg)
    if args.command in ("quality", "study", "all"):
        q = cmd_quality(cfg)
        if args.command != "quality":
            cmd_study(cfg, q)
