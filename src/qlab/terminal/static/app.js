// Точка входа интерфейса: выбор рынка, опрос сервера, связка графика, панели, журнала и аналитики.
import { createChart } from "./chart.js";
import { renderEvents, renderFacts, renderOpenTrades, renderPreview, renderSimilar, tradePreview } from "./panel.js";
import { renderAnalytics, renderJournal } from "./journal.js";

const $ = (id) => document.getElementById(id);
const TF_MIN = { "5m": 5, "15m": 15 };
const saved = (() => { try { return JSON.parse(localStorage.getItem("terminal") || "{}"); } catch { return {}; } })();
const ui = { symbol: saved.symbol || "ETHUSDT", tf: saved.tf || "5m", view: "chart", state: null, event: null, trades: [] };
const remember = () => { try { localStorage.setItem("terminal", JSON.stringify({ symbol: ui.symbol, tf: ui.tf })); } catch {} };

async function api(path, opts = {}) {
  const res = await fetch(path, { method: opts.method || "GET", headers: { "Content-Type": "application/json" },
    body: opts.body ? JSON.stringify(opts.body) : undefined });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

const chart = createChart($("chart"));

function seg(el, items, current, onPick) {
  el.innerHTML = items.map((x) => `<button data-v="${x}" class="${x === current ? "on" : ""}">${x.replace("USDT", "")}</button>`).join("");
  el.querySelectorAll("button").forEach((b) => (b.onclick = () => onPick(b.dataset.v)));
}

async function loadState(full) {
  const since = full ? null : chart.lastTime();
  const q = `/api/state?symbol=${ui.symbol}&tf=${ui.tf}${since ? `&since=${since}` : ""}`;
  try {
    const state = await api(q);
    if (state.symbol !== ui.symbol || state.tf !== ui.tf) return;
    ui.state = { ...(ui.state || {}), ...state };
    chart.render(state, full || !since);
    $("chart-msg").textContent = "";
    renderFacts(state);
    renderPreview(state);
    const top = state.events[0]?.id;
    if (!ui.event || !state.events.some((e) => e.id === ui.event) || top !== ui.topEvent) {
      ui.topEvent = top;
      ui.event = null;
    }
    selectEvent(ui.event);
  } catch (e) {
    $("chart-msg").textContent = e.message;
  }
}

let similarFor = null;
async function selectEvent(id) {
  const chosen = renderEvents(ui.state, id, (nid) => { ui.event = nid; selectEvent(nid); });
  ui.event = chosen;
  if (!chosen) return;
  const key = `${ui.symbol}:${ui.tf}:${chosen}`;
  if (similarFor && similarFor.key === key) { renderSimilar(similarFor.res, TF_MIN[ui.tf]); return; }
  try {
    const res = await api(`/api/similar?symbol=${ui.symbol}&tf=${ui.tf}&event_id=${encodeURIComponent(chosen)}`);
    similarFor = { key, res };
    if (ui.event === chosen) renderSimilar(res, TF_MIN[ui.tf]);
  } catch (e) { $("similar") && ($("similar").textContent = e.message); }
}

function pickMarket(symbol, tf) {
  ui.symbol = symbol; ui.tf = tf; ui.event = null; ui.state = null; similarFor = null;
  remember();
  seg($("symbols"), ui.config.symbols, ui.symbol, (s) => pickMarket(s, ui.tf));
  seg($("tfs"), ui.config.timeframes, ui.tf, (t) => pickMarket(ui.symbol, t));
  chart.reset();
  $("chart-msg").textContent = "Загрузка данных…";
  loadState(true);
}

async function openTrade(side) {
  if (!ui.state) return;
  const { stop, target } = tradePreview(ui.state, side);
  try {
    const r = await api("/api/trades", { method: "POST", body: {
      symbol: ui.symbol, tf: ui.tf, side, stop, target, setup: $("setup").value, note: $("note").value, screenshot: chart.screenshot() } });
    toast(`Сделка #${r.id} записана: ${side.toUpperCase()} от ${Number(r.entry).toPrecision(6)}. Рынок, условия и скриншот сохранены.`);
    $("note").value = "";
    refreshTrades();
  } catch (e) { toast(`Не записано: ${e.message}`); }
}

async function closeTrade(id) {
  if (!confirm(`Закрыть сделку #${id} по текущей цене?`)) return;
  try { await api(`/api/trades/${id}/close`, { method: "POST", body: {} }); refreshTrades(); } catch (e) { toast(e.message); }
}

async function refreshTrades() {
  try {
    ui.trades = await api("/api/trades");
    renderOpenTrades(ui.trades, closeTrade);
    $("setups").innerHTML = [...new Set(ui.trades.map((t) => t.setup).filter(Boolean))].map((s) => `<option value="${s}">`).join("");
    if (ui.view === "journal") renderJournal($("view-journal"), ui.trades, api, refreshTrades);
  } catch {}
}

async function showView(v) {
  ui.view = v;
  document.querySelectorAll("#views button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  for (const name of ["chart", "journal", "analytics"]) $(`view-${name}`).hidden = name !== v;
  if (v === "journal") renderJournal($("view-journal"), ui.trades, api, refreshTrades);
  if (v === "analytics") renderAnalytics($("view-analytics"), await api("/api/analytics"));
}

async function pollStatus() {
  try {
    const s = await api("/api/status");
    $("dot").className = `dot ${s.ok ? "ok" : "bad"}`;
    const prog = Object.values(s.progress || {}).find((p) => p !== "готово");
    $("status").textContent = prog || (s.mode === "replay" ? `${s.message}` : s.message);
    $("clock").textContent = new Date(s.now * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: s.mode === "replay" ? undefined : "2-digit" })
      + (s.mode === "replay" ? " · " + new Date(s.now * 1000).toLocaleDateString("ru-RU") : "");
  } catch { $("dot").className = "dot bad"; $("status").textContent = "сервер недоступен"; }
}

async function main() {
  ui.config = await api("/api/config");
  if (!ui.config.symbols.includes(ui.symbol)) ui.symbol = ui.config.symbols[0];
  document.querySelectorAll("#views button").forEach((b) => (b.onclick = () => showView(b.dataset.view)));
  $("btn-long").onclick = () => openTrade("long");
  $("btn-short").onclick = () => openTrade("short");
  for (const id of ["stop-atr", "target-r"]) $(id).oninput = () => ui.state && renderPreview(ui.state);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea")) return;
    const i = Number(e.key) - 1;
    if (i >= 0 && i < ui.config.symbols.length) pickMarket(ui.config.symbols[i], ui.tf);
    if (e.key === "t") pickMarket(ui.symbol, ui.tf === "5m" ? "15m" : "5m");
  });
  pickMarket(ui.symbol, ui.tf);
  pollStatus(); refreshTrades();
  setInterval(() => ui.view === "chart" && loadState(false), 3000);
  setInterval(pollStatus, 2000);
  setInterval(refreshTrades, 5000);
}

main();
