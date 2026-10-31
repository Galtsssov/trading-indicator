"""Запуск терминала.

  qlab-terminal                       живой рынок (Binance), откроется браузер
  qlab-terminal --replay 2024-03-01   повтор истории из архива (тренажёр), --speed 60 = минута за секунду
  qlab-terminal history               построить базу похожих ситуаций из архива (после `python -m qlab fetch`)
"""
from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from pathlib import Path

from qlab.config import Config


def build_history(platform_cfg: Config, out_dir: Path, symbols: list[str], timeframes: list[str]) -> None:
    from qlab.cache import Data
    from qlab.terminal.similar import build_history as build
    data = Data(platform_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        for tf in timeframes:
            h = build(data.bars(s, tf), s, tf)
            h.to_parquet(out_dir / f"{s}_{tf}.parquet", index=False)
            logging.info("база похожих ситуаций %s %s: %d событий", s, tf, len(h))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qlab-terminal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "history"])
    ap.add_argument("--config", default="configs/terminal.toml")
    ap.add_argument("--platform-config", default="configs/platform.toml")
    ap.add_argument("--replay", metavar="ДАТА", help="повтор истории с этой даты (YYYY-MM-DD) вместо живого рынка")
    ap.add_argument("--speed", type=float, default=60.0, help="ускорение повтора: секунд рынка за секунду")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, pcfg = Config.load(a.config), Config.load(a.platform_config)
    t = cfg.section("terminal")
    data_dir = Path(t["data_dir"])
    if a.command == "history":
        build_history(pcfg, data_dir / "history", t["symbols"], t["timeframes"])
        return

    import uvicorn

    from qlab.terminal.app import create_app
    from qlab.terminal.journal import Journal
    from qlab.terminal.market import Market
    from qlab.terminal.sources import LiveSource, ReplaySource

    if a.replay:
        source = ReplaySource(t["symbols"], pcfg.data_dir / "clean", a.replay, a.speed, t["history_days"])
        journal_name = "journal_replay.sqlite"
    else:
        source = LiveSource(t["symbols"], data_dir / "live", t["history_days"], t["poll_seconds"], t["metrics_seconds"])
        journal_name = "journal.sqlite"
    costs = cfg.section("costs")
    journal = Journal(data_dir / journal_name, data_dir / "screenshots", 2 * (costs["taker_fee"] + costs["slippage"]))
    market = Market(source, pcfg.section("features"), pcfg.section("availability"))
    app = create_app(source, market, journal, data_dir / "history", t["symbols"], t["timeframes"], t["similar_k"])
    url = f"http://{t['host']}:{t['port']}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    logging.info("Терминал: %s (Ctrl+C — остановить)", url)
    uvicorn.run(app, host=t["host"], port=t["port"], log_level="warning")


if __name__ == "__main__":
    main()
