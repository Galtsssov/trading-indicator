"""HTTP API терминала (FastAPI) + раздача интерфейса. Работает только на 127.0.0.1."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qlab.config import TF_MINUTES
from qlab.terminal.journal import Journal, analyze
from qlab.terminal.market import Market, Snapshot
from qlab.terminal.similar import History
from qlab.terminal.sources import Source

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class TradeIn(BaseModel):
    symbol: str
    tf: str
    side: str = Field(pattern="^(long|short)$")
    stop: float
    target: float | None = None
    entry: float | None = None          # по умолчанию — текущая цена
    setup: str = ""
    note: str = ""
    screenshot: str | None = None       # data:image/png;base64,...


class CloseIn(BaseModel):
    exit: float | None = None


class NoteIn(BaseModel):
    setup: str | None = None
    note: str | None = None


def conditions(snap: Snapshot, side: str, entry: float, stop: float, target: float | None) -> dict:
    """Плоские условия на момент входа — по ним строится персональная аналитика."""
    last = snap.closed.iloc[-1]
    sign = 1 if side == "long" else -1
    c = snap.closed["close"]
    n = min(96, len(c) - 1)
    trend_dir = np.sign(c.iloc[-1] - c.iloc[-1 - n])
    tb = (snap.closed["taker_buy_volume"] / snap.closed["volume"].replace(0, np.nan)).tail(6).mean()
    oi = snap.closed["oi"]
    k = 60 // TF_MINUTES[snap.tf]
    sweep_recent = any(e["type"] == "liquidity_sweep" and e["bars_ago"] <= 6 for e in snap.payload["events"])
    last_event = snap.payload["events"][0] if snap.payload["events"] else None
    risk = abs(entry - stop)
    session = {0: "Азия", 1: "Европа", 2: "США", 3: "Поздняя"}[int(np.searchsorted([0, 7, 13, 20], last["ts"].hour, side="right") - 1)]
    return {
        "session": session, "regime": last["regime_trend"], "volatility": last["regime_vol"], "btc": last["regime_btc"],
        "event": f"{last_event['name']} {'↑' if last_event['dir'] > 0 else '↓'}" if last_event and last_event["bars_ago"] <= 6 else "нет",
        "hour": int(pd.Timestamp(snap.payload["now"], unit="s").hour),
        "is_trend": last["regime_trend"] == "trend", "is_high_vol": last["regime_vol"] == "high_vol",
        "is_btc_strong": last["regime_btc"] == "btc_strong", "is_risk_on": last["regime_risk"] == "risk_on",
        "is_with_trend": bool(trend_dir == sign), "is_above_vwap": bool(entry > last["vwap"]),
        "is_with_vwap": bool(np.sign(entry - last["vwap"]) == sign), "is_after_sweep": sweep_recent,
        "is_us_session": session == "США", "is_with_aggressors": bool(np.isfinite(tb) and np.sign(tb - 0.5) == sign),
        "is_funding_crowded": bool(np.isfinite(last["funding_rate"]) and abs(last["funding_rate"]) > 0.0005),
        "is_oi_rising": bool(len(oi) > k and np.isfinite(oi.iloc[-1]) and np.isfinite(oi.iloc[-1 - k]) and oi.iloc[-1] > oi.iloc[-1 - k]),
        "is_rr_2": bool(target is not None and round(abs(target - entry) / risk, 6) >= 2),
    }


def create_app(source: Source, market: Market, journal: Journal, history_dir: Path, symbols: list[str],
               timeframes: list[str], similar_k: int = 150) -> FastAPI:
    histories: dict[tuple[str, str], History] = {}

    def history(symbol: str, tf: str) -> History:
        if (symbol, tf) not in histories:
            histories[(symbol, tf)] = History(history_dir, symbol, tf)
        return histories[(symbol, tf)]

    def snap(symbol: str, tf: str) -> Snapshot:
        if symbol not in symbols or tf not in timeframes:
            raise HTTPException(404, "неизвестная монета или ТФ")
        s = market.snapshot(symbol, tf)
        if s is None:
            raise HTTPException(503, source.status.progress.get(symbol) or source.status.message or "данные загружаются")
        return s

    async def follow() -> None:                 # сопровождение открытых сделок: стоп, цель, MFE/MAE
        while True:
            for s in symbols:
                try:
                    m1 = source.minutes(s)
                    if m1 is not None and len(m1):
                        journal.update_open(s, m1, source.now())
                except Exception:
                    log.exception("сопровождение сделок %s", s)
            await asyncio.sleep(5)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        source.start()
        task = asyncio.create_task(follow())
        yield
        task.cancel()

    app = FastAPI(title="Торговый терминал", docs_url="/api/docs", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config() -> dict:
        return {"symbols": symbols, "timeframes": timeframes, "mode": source.mode,
                "history": {f"{s}_{tf}": history(s, tf).available for s in symbols for tf in timeframes}}

    @app.get("/api/status")
    def status() -> dict:
        st = source.status
        return {"ok": st.ok, "message": st.message, "progress": st.progress, "mode": source.mode,
                "now": int(source.now().timestamp()), "version": source.version}

    @app.get("/api/state")
    def state(symbol: str, tf: str, since: int | None = None) -> JSONResponse:
        p = dict(snap(symbol, tf).payload)
        if since is not None:                 # инкрементальное обновление: только хвост свечей
            for k in ("candles", "volume", "vwap", "atr"):
                p[k] = [x for x in p[k] if x["time"] >= since]
            p["incremental"] = True
        return JSONResponse(p)

    @app.get("/api/similar")
    def similar_view(symbol: str, tf: str, event_id: str | None = None) -> dict:
        s = snap(symbol, tf)
        evs = s.payload["events"]
        ev = next((e for e in evs if e["id"] == event_id), evs[0] if evs else None)
        if ev is None:
            return {"event": None}
        res = history(symbol, tf).query(ev["type"], ev["dir"], ev["context"], source.now(), k=similar_k,
                                        tf_minutes=TF_MINUTES[tf])
        return {"event": ev, **res}

    @app.post("/api/trades")
    def open_trade(t: TradeIn) -> dict:
        s = snap(t.symbol, t.tf)
        entry = t.entry or s.payload["price"]
        ctx = {"facts": s.payload["facts"]["groups"], "events": s.payload["events"][:5], "regime": s.payload["regime"],
               "features": {k: (None if not np.isfinite(v) else float(v)) for k, v in s.closed.iloc[-1].items()
                            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)},
               "conditions": conditions(s, t.side, entry, t.stop, t.target)}
        try:
            tid = journal.open(mode=source.mode, symbol=t.symbol, tf=t.tf, side=t.side, entry=entry, stop=t.stop,
                               target=t.target, setup=t.setup, note=t.note, context=ctx,
                               screenshot_data_url=t.screenshot, opened_at=source.now())
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": tid, "entry": entry}

    @app.post("/api/trades/{tid}/close")
    def close_trade(tid: int, body: CloseIn) -> dict:
        t = next((x for x in journal.list() if x["id"] == tid), None)
        if t is None:
            raise HTTPException(404, "нет такой сделки")
        price = body.exit or snap(t["symbol"], t["tf"]).payload["price"]
        try:
            journal.close(tid, price, "manual", source.now())
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.patch("/api/trades/{tid}")
    def edit_trade(tid: int, body: NoteIn) -> dict:
        journal.update_note(tid, body.setup, body.note)
        return {"ok": True}

    @app.delete("/api/trades/{tid}")
    def delete_trade(tid: int) -> dict:
        journal.delete(tid)
        return {"ok": True}

    @app.get("/api/trades")
    def trades() -> list[dict]:
        out = []
        for t in journal.list(source.mode):
            t = {k: v for k, v in t.items() if k != "context"} | {"conditions": t["context"].get("conditions", {}),
                                                                   "regime": t["context"].get("regime")}
            if t["status"] == "open":
                try:
                    price = market.snapshot(t["symbol"], t["tf"]).payload["price"]
                    risk = abs(t["entry"] - t["stop"])
                    t["live_r"] = (price - t["entry"]) / risk * (1 if t["side"] == "long" else -1)
                except Exception:
                    t["live_r"] = None
            out.append(t)
        return out

    @app.get("/api/analytics")
    def analytics() -> dict:
        return analyze(journal.frame(source.mode))

    @app.get("/api/screenshot/{name}")
    def screenshot(name: str) -> FileResponse:
        p = journal.shots / Path(name).name
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
