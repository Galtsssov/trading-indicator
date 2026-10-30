"""Командная строка платформы.

  fetch                 скачать архив Binance и собрать Parquet
  quality               отчёт о качестве данных
  cache                 посчитать кэш признаков для всех монет и ТФ
  registry              список гипотез и статусов
  register FILE...      зарегистрировать YAML-гипотезы (до любой проверки)
  test ID [--split S]   проверить гипотезу (validation по умолчанию; holdout — только с --confirm-holdout)
  discover              поиск аномалий на training, регистрация кандидатов
  pending               проверить на validation все гипотезы Testing без проверки
  all                   fetch → quality → cache → discover → pending (holdout не трогается)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

from qlab import discovery, engine
from qlab.cache import Data
from qlab.config import Config
from qlab.data import binance, quality
from qlab.hypothesis import Hypothesis
from qlab.registry import Registry

log = logging.getLogger("qlab")
DATASETS = ("klines_1m", "metrics", "funding")
LEDGER = Path("reports/ledger.jsonl")


def ledger(event: str, **kw) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event,
                            "git_commit": _git_commit(), **kw}, ensure_ascii=False, default=str) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_fetch(cfg: Config) -> None:
    raw, clean = cfg.data_dir / "raw", cfg.data_dir / "clean"
    d = cfg.section("data")
    files = [f for s in cfg.symbols for f in binance.plan(d["base_url"], raw, s, d["start"], date.today())]
    log.info("файлов в плане: %d; скачивание: %s", len(files), binance.fetch(files, d["workers"]))
    dups = {f"{ds}/{s}": binance.build_clean(raw, clean, ds, s) for ds in DATASETS for s in cfg.symbols}
    (clean / "build_stats.json").write_text(json.dumps(dups, indent=1))


def cmd_quality(cfg: Config) -> None:
    clean, data = cfg.data_dir / "clean", Data(cfg)
    p = clean / "build_stats.json"
    dups = json.loads(p.read_text()) if p.exists() else {}
    rows = [quality.check(ds, s, data.table(ds, s), dups.get(f"{ds}/{s}") or 0).as_dict()
            for s in cfg.symbols for ds in DATASETS]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/quality.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for q in rows:
        log.info("%-8s %-9s %s → %s  покрытие=%s  %s", q["symbol"], q["dataset"], q["first"], q["last"],
                 q["coverage_pct"], q["verdict"])


def cmd_cache(cfg: Config, data: Data) -> None:
    for s in cfg.symbols:
        for tf in cfg.timeframes:
            data.bars(s, tf)


def cmd_registry(reg: Registry) -> None:
    for hid, e in sorted(reg.items.items()):
        print(f"{hid}  {e['status']:<9} {e['name']}" + (f"\n        причина: {e['reason']}" if e["reason"] else ""))


def cmd_register(reg: Registry, files: list[str]) -> None:
    for f in files:
        h = Hypothesis.load(f)
        reg.register(h)
        ledger("register", id=h.id, fingerprint=h.fingerprint(), lock=h.lock())
        log.info("зарегистрирована %s: %s", h.id, h.name)


def cmd_test(cfg: Config, data: Data, reg: Registry, hid: str, split: str, confirm_holdout: bool) -> bool:
    h = Hypothesis.load(reg.items[hid]["path"])
    passed, report = engine.run(cfg, data, h, reg, split, confirm_holdout=confirm_holdout)
    ledger("test", id=hid, split=split, passed=passed, report=str(report), status=reg.items[hid]["status"])
    return passed


def cmd_discover(cfg: Config, data: Data, reg: Registry) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    df = discovery.scan(cfg, data)
    rob = discovery.robust(df) if not df.empty else df
    cand = discovery.candidates(rob, cfg.section("discovery")["max_candidates"])
    registered = discovery.register_candidates(cfg, reg, cand, stamp)
    path = discovery.write_report(cfg, df, rob, cand, registered, Path("reports/discovery") / stamp)
    ledger("discover", tests=len(df), significant=int(df["significant"].sum()) if not df.empty else 0,
           robust=len(rob), registered=[h.id for h in registered], report=str(path))
    log.info("discovery: %s", path)


def cmd_pending(cfg: Config, data: Data, reg: Registry) -> None:
    for hid, e in sorted(reg.items.items()):
        if e["status"] == "Testing" and not any(c["split"] == "validation" for c in e["checks"]):
            cmd_test(cfg, data, reg, hid, "validation", False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "quality", "cache", "registry", "register", "test", "discover",
                                        "pending", "all"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--config", default="configs/platform.toml")
    ap.add_argument("--registry", default="hypotheses/registry.json")
    ap.add_argument("--split", default="validation", choices=["validation", "holdout"])
    ap.add_argument("--confirm-holdout", action="store_true", help="holdout одноразовый: подтвердить осознанно")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, reg = Config.load(a.config), Registry(a.registry)
    data = Data(cfg)
    match a.command:
        case "fetch":
            cmd_fetch(cfg)
        case "quality":
            cmd_quality(cfg)
        case "cache":
            cmd_cache(cfg, data)
        case "registry":
            cmd_registry(reg)
        case "register":
            cmd_register(reg, a.args)
        case "test":
            for hid in a.args:
                cmd_test(cfg, data, reg, hid, a.split, a.confirm_holdout)
        case "discover":
            cmd_discover(cfg, data, reg)
        case "pending":
            cmd_pending(cfg, data, reg)
        case "all":
            cmd_fetch(cfg)
            cmd_quality(cfg)
            cmd_cache(cfg, data)
            cmd_discover(cfg, data, reg)
            cmd_pending(cfg, data, reg)
