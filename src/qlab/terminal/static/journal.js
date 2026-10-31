// Журнал сделок и персональная аналитика.
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const r2 = (x) => (x == null ? "—" : `${x > 0 ? "+" : ""}${Number(x).toFixed(2)}R`);
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const dt = (s) => (s ? new Date(s).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—");
const p6 = (x) => (x == null ? "—" : Number(x).toPrecision(6));
const COND = {
  is_trend: "тренд", is_high_vol: "высокая вол.", is_btc_strong: "BTC вырос", is_risk_on: "risk on", is_with_trend: "по тренду",
  is_above_vwap: "выше VWAP", is_with_vwap: "по стороне VWAP", is_after_sweep: "после свипа", is_us_session: "сессия США",
  is_with_aggressors: "агрессоры за нас", is_funding_crowded: "перекос funding", is_oi_rising: "OI растёт", is_rr_2: "цель ≥ 2R",
};

export function renderJournal(el, trades, api, refresh) {
  if (!trades.length) {
    el.innerHTML = `<h2>Журнал</h2><div class="empty">Сделок пока нет. Нажмите LONG или SHORT на графике — терминал запомнит рынок, условия и скриншот.</div>`;
    return;
  }
  el.innerHTML = `<h2>Журнал · сделок: ${trades.length}</h2>
    <div class="journal-grid"><div><table><thead><tr>
      <th>Открыта</th><th>Рынок</th><th>Сторона</th><th>Сетап</th><th>Вход</th><th>Стоп</th><th>Цель</th><th>Выход</th><th>Причина</th>
      <th>Результат</th><th>MFE</th><th>MAE</th></tr></thead><tbody>
      ${trades.map((t) => `<tr data-id="${t.id}"><td class="num">${dt(t.opened_at)}</td><td>${t.symbol} ${t.tf}</td>
        <td class="${t.side === "long" ? "up" : "down"}">${t.side.toUpperCase()}</td><td>${esc(t.setup || "—")}</td>
        <td class="num">${p6(t.entry)}</td><td class="num">${p6(t.stop)}</td><td class="num">${p6(t.target)}</td>
        <td class="num">${t.status === "open" ? "открыта" : p6(t.exit)}</td><td>${{ stop: "стоп", target: "цель", manual: "вручную" }[t.exit_reason] || "—"}</td>
        <td class="num ${(t.result_r ?? t.live_r) > 0 ? "up" : "down"}">${r2(t.result_r ?? t.live_r)}</td>
        <td class="num">${Number(t.mfe_r || 0).toFixed(2)}</td><td class="num">${Number(t.mae_r || 0).toFixed(2)}</td></tr>`).join("")}
      </tbody></table></div><div class="detail" id="detail"><div class="muted">Выберите сделку.</div></div></div>`;
  el.querySelectorAll("tbody tr").forEach((tr) => (tr.onclick = () => showDetail(trades.find((t) => t.id === +tr.dataset.id))));

  function showDetail(t) {
    const conds = Object.entries(t.conditions || {}).filter(([k]) => k.startsWith("is_"));
    document.getElementById("detail").innerHTML = `
      <div class="card"><h3>Сделка #${t.id} · ${t.symbol} ${t.tf} · ${t.side.toUpperCase()}</h3>
        <div class="muted">${dt(t.opened_at)} → ${dt(t.closed_at)} · ${esc(t.regime || "")}</div>
        ${t.screenshot ? `<img src="/api/screenshot/${t.screenshot}" alt="скриншот" style="margin:8px 0">` : ""}
        <div class="muted" style="margin:6px 0">Условия на входе:</div>
        <div class="chips">${conds.map(([k, v]) => `<span class="chip ${v ? "on" : ""}">${COND[k] || k}: ${v ? "да" : "нет"}</span>`).join("")}</div>
        <div class="muted">Сессия: ${esc(t.conditions?.session)} · событие перед входом: ${esc(t.conditions?.event)}</div>
        <div class="trade" style="margin-top:10px">
          <label>Сетап<input id="ed-setup" value="${esc(t.setup || "")}"></label>
          <label style="margin-top:6px">Заметка / разбор<textarea id="ed-note" rows="3">${esc(t.note || "")}</textarea></label>
          <div class="btns" style="margin-top:8px"><button class="btn small" id="ed-save">Сохранить</button>
            <button class="btn small" id="ed-del">Удалить сделку</button></div>
        </div></div>`;
    document.getElementById("ed-save").onclick = async () => {
      await api(`/api/trades/${t.id}`, { method: "PATCH", body: { setup: document.getElementById("ed-setup").value, note: document.getElementById("ed-note").value } });
      refresh();
    };
    document.getElementById("ed-del").onclick = async () => {
      if (confirm(`Удалить сделку #${t.id}? Это нельзя отменить.`)) { await api(`/api/trades/${t.id}`, { method: "DELETE" }); refresh(); }
    };
  }
}

function equitySvg(eq) {
  if (eq.length < 2) return "";
  const w = 600, h = 140, lo = Math.min(0, ...eq), hi = Math.max(0, ...eq);
  const x = (i) => (i / (eq.length - 1)) * (w - 8) + 4, y = (v) => h - 6 - ((v - lo) / (hi - lo || 1)) * (h - 12);
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:140px" preserveAspectRatio="none">
    <line x1="0" x2="${w}" y1="${y(0)}" y2="${y(0)}" stroke="#2a3140" stroke-dasharray="3 3"/>
    <polyline points="${eq.map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="#5b8def" stroke-width="2"/></svg>`;
}

export function renderAnalytics(el, a) {
  if (!a.n) {
    el.innerHTML = `<h2>Аналитика</h2><div class="empty">${esc(a.message)} Аналитика появится после первых закрытых сделок; выводы — после 20–30.</div>`;
    return;
  }
  const o = a.overall;
  const group = (title, rows) => `<div class="card"><h3>${esc(title)}</h3><table><thead><tr><th></th><th>Сделок</th><th>WR</th><th>Ожидание</th><th>Итого</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td>${esc(r.value)}${r.enough ? "" : ' <span class="muted">(мало)</span>'}</td><td class="num">${r.n}</td><td class="num">${pct(r.win_rate)}</td>
      <td class="num ${r.expectancy > 0 ? "up" : "down"}">${r2(r.expectancy)}</td><td class="num">${r2(r.total_r)}</td></tr>`).join("")}</tbody></table></div>`;
  el.innerHTML = `<h2>Аналитика · закрытых сделок: ${a.n}</h2>
    <div class="kpis">
      <div class="kpi"><div class="k">Win rate</div><div class="v num">${pct(o.win_rate)}</div></div>
      <div class="kpi"><div class="k">Ожидание</div><div class="v num ${o.expectancy > 0 ? "up" : "down"}">${r2(o.expectancy)}</div></div>
      <div class="kpi"><div class="k">Итого</div><div class="v num">${r2(o.total_r)}</div></div>
      <div class="kpi"><div class="k">Profit factor</div><div class="v num">${o.profit_factor == null ? "—" : o.profit_factor.toFixed(2)}</div></div>
      <div class="kpi"><div class="k">Средний плюс / минус</div><div class="v num">${r2(o.avg_win)} / ${r2(o.avg_loss)}</div></div>
    </div>
    <div class="card"><h3>Кривая результата, R</h3>${equitySvg(a.equity)}</div>
    <div class="card"><h3>Какие условия улучшают мои результаты</h3>
      <table><thead><tr><th>Условие на входе</th><th>С ним</th><th>Без него</th><th>Разница</th><th>90% интервал</th><th></th></tr></thead><tbody>
      ${a.filters.map((f) => `<tr><td>${esc(f.condition)}</td><td class="num">${r2(f.exp_with)} (${f.n_with})</td><td class="num">${r2(f.exp_without)} (${f.n_without})</td>
        <td class="num ${f.delta > 0 ? "up" : "down"}">${r2(f.delta)}</td><td class="num muted">${r2(f.ci90[0])} … ${r2(f.ci90[1])}</td>
        <td><span class="tag ${{ "улучшает": "good", "ухудшает": "bad" }[f.verdict] || "watch"}">${f.verdict}</span></td></tr>`).join("")}</tbody></table>
      <div class="note">${esc(a.note)}</div></div>
    <div class="card"><h3>Где я чаще ошибаюсь</h3>${a.weak_spots.length ? `<table><tbody>${a.weak_spots.slice(0, 8).map((w) => `<tr><td>${esc(w.where)}</td>
      <td class="num">${w.n} сделок</td><td class="num">WR ${pct(w.win_rate)}</td><td class="num down">${r2(w.expectancy)}</td></tr>`).join("")}</tbody></table>`
      : '<div class="muted">Групп с отрицательным ожиданием (от 5 сделок) нет.</div>'}</div>
    <div class="grid2">${Object.entries(a.groups).map(([t, rows]) => group(t, rows)).join("")}</div>`;
}
