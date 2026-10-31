// Правая панель: факты, события с похожими случаями, форма сделки.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const fx = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));

export function renderFacts(state) {
  $("pair").textContent = `${state.symbol} · ${state.tf}`;
  $("px").textContent = Number(state.price).toPrecision(6);
  $("regime").textContent = state.regime;
  $("facts").innerHTML = state.facts.groups.map((g) => `
    <div class="card"><h3>${esc(g.title)}</h3>
      ${g.items.map((f) => `<div class="fact" title="${esc(f.hint)}"><span class="k">${esc(f.label)}</span>
        <span class="v num ${f.tone === "up" ? "up" : f.tone === "down" ? "down" : f.tone === "warn" ? "warn" : ""}">${esc(f.value)}</span></div>`).join("")}
    </div>`).join("");
}

export function renderEvents(state, selected, onSelect) {
  const evs = state.events.slice(0, 6);
  if (!evs.length) {
    $("event").innerHTML = `<div class="muted">За последние 60 баров событий нет.</div>`;
    return null;
  }
  const cur = evs.find((e) => e.id === selected) || evs[0];
  $("event").innerHTML = `<div class="chips">${evs.map((e) => `<span class="chip ${e.id === cur.id ? "on" : ""}" data-id="${e.id}">
      ${esc(e.name)} ${e.dir > 0 ? "↑" : "↓"} · ${e.bars_ago} бар.</span>`).join("")}</div><div id="similar" class="muted">Ищу похожие случаи…</div>`;
  $("event").querySelectorAll(".chip").forEach((c) => (c.onclick = () => onSelect(c.dataset.id)));
  return cur.id;
}

function pathSvg(sim, all) {
  const w = 320, h = 110, n = sim.mean_path.length;
  const vals = [...sim.q25_path, ...sim.q75_path, ...all.mean_path, 0];
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i) => (i / (n - 1)) * (w - 8) + 4;
  const y = (v) => h - 6 - ((v - lo) / (hi - lo || 1)) * (h - 12);
  const line = (arr) => arr.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const band = `${line(sim.q75_path)} ${sim.q25_path.map((v, i) => `${x(n - 1 - i).toFixed(1)},${y(sim.q25_path[n - 1 - i]).toFixed(1)}`).join(" ")}`;
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <line x1="0" x2="${w}" y1="${y(0)}" y2="${y(0)}" stroke="#2a3140" stroke-dasharray="3 3"/>
    <polygon points="${band}" fill="rgba(91,141,239,0.14)"/>
    <polyline points="${line(all.mean_path)}" fill="none" stroke="#5d6678" stroke-width="1.2"/>
    <polyline points="${line(sim.mean_path)}" fill="none" stroke="#5b8def" stroke-width="2"/>
  </svg>`;
}

export function renderSimilar(res, tfMinutes) {
  const el = $("similar");
  if (!el) return;
  if (!res || !res.event) { el.textContent = ""; return; }
  if (!res.available) {
    el.innerHTML = `Базы похожих ситуаций нет. Постройте её один раз: <span class="num">python -m qlab fetch</span>, затем <span class="num">qlab-terminal history</span>.`;
    return;
  }
  if (res.too_few) { el.textContent = `В истории слишком мало таких событий (${res.n}).`; return; }
  const s = res.similar, a = res.all, dir = res.event.dir > 0 ? "вверх" : "вниз";
  const hrs = (s.horizon_minutes / 60).toFixed(s.horizon_minutes % 60 ? 1 : 0);
  const row = (label, v, va, fmt = pct) => `<span>${label}</span><span class="num">${fmt(v)}</span><span class="num vs">${fmt(va)}</span>`;
  el.className = "";
  el.innerHTML = `
    <div class="muted" style="margin-bottom:6px">Похожих случаев: <b class="num">${s.n}</b> из ${res.n_pool} таких же событий в истории.
      Направление события — ${dir}; «продолжение» = движение ${dir}.</div>
    <div class="stats">
      <span class="h"></span><span class="h">похожие</span><span class="h">все</span>
      ${row(`Продолжение через ${hrs} ч`, s.continued, a.continued)}
      ${row("+1 ATR раньше −1 ATR", s.p_hit1, a.p_hit1)}
      ${row("+2 ATR раньше −1 ATR", s.p_hit2, a.p_hit2)}
      ${row("Касание VWAP", s.p_touch_vwap, a.p_touch_vwap)}
      ${row("Новый экстремум за точкой события", s.p_new_extreme, a.p_new_extreme)}
      ${row("Макс. продолжение (медиана), ATR", s.mfe_median, a.mfe_median, (v) => fx(v))}
      ${row("Макс. продолжение (90%), ATR", s.mfe_p90, a.mfe_p90, (v) => fx(v))}
      ${row("Макс. откат (медиана), ATR", s.mae_median, a.mae_median, (v) => fx(v))}
      ${row("Время до макс. продолжения, мин", s.t_mfe_minutes, a.t_mfe_minutes, (v) => fx(v, 0))}
    </div>
    <div class="path">${pathSvg(s, a)}</div>
    <div class="note">Синяя линия — средний путь похожих случаев (полоса — 25–75%), серая — всех таких событий. Ось — ATR по направлению события, ${s.mean_path.length} баров. Это статистика прошлого, а не сигнал.</div>`;
}

export function tradePreview(state, side) {
  const atr = state.facts.atr, price = state.price;
  const k = parseFloat($("stop-atr").value) || 1;
  const rr = parseFloat($("target-r").value);
  const sign = side === "long" ? 1 : -1;
  const stop = price - sign * k * atr;
  const target = Number.isFinite(rr) ? price + sign * rr * k * atr : null;
  return { stop, target, price };
}

export function renderPreview(state) {
  const l = tradePreview(state, "long"), s = tradePreview(state, "short");
  const p = (x) => Number(x).toPrecision(6);
  $("preview").innerHTML = `LONG: стоп ${p(l.stop)}${l.target ? ` · цель ${p(l.target)}` : ""}<br>SHORT: стоп ${p(s.stop)}${s.target ? ` · цель ${p(s.target)}` : ""}`;
}

export function renderOpenTrades(trades, onClose) {
  const open = trades.filter((t) => t.status === "open");
  $("open-trades").innerHTML = open.length ? `<div style="margin-top:10px">${open.map((t) => `
    <div class="open-trade"><span>${t.symbol} ${t.tf} <b class="${t.side === "long" ? "up" : "down"}">${t.side.toUpperCase()}</b>
      <span class="muted num">от ${Number(t.entry).toPrecision(6)}</span></span>
      <span class="num ${t.live_r > 0 ? "up" : "down"}">${t.live_r == null ? "—" : `${t.live_r > 0 ? "+" : ""}${t.live_r.toFixed(2)}R`}</span>
      <button class="btn small" data-close="${t.id}">Закрыть</button></div>`).join("")}</div>` : "";
  $("open-trades").querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => onClose(+b.dataset.close)));
}
