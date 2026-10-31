// График: свечи, объём, VWAP, ATR, маркеры событий и структуры, линии уровней.
// Время с сервера — UTC (секунды); на графике показываем местное время.
const TZ = -new Date().getTimezoneOffset() * 60;
const local = (t) => t + TZ;

export function createChart(el) {
  const chart = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { type: "solid", color: "#0b0e13" }, textColor: "#8a93a6", fontSize: 11 },
    grid: { vertLines: { color: "#141920" }, horzLines: { color: "#141920" } },
    rightPriceScale: { borderColor: "#1f2632", scaleMargins: { top: 0.06, bottom: 0.3 } },
    timeScale: { borderColor: "#1f2632", timeVisible: true, secondsVisible: false, rightOffset: 8 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    watermark: { visible: true, text: "", color: "rgba(150,160,180,0.06)", fontSize: 44 },
  });
  const candles = chart.addCandlestickSeries({
    upColor: "#3ccf91", downColor: "#ef5a5a", wickUpColor: "#3ccf91", wickDownColor: "#ef5a5a", borderVisible: false,
  });
  const volume = chart.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
  chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.88, bottom: 0 } });
  const atr = chart.addLineSeries({ priceScaleId: "atr", color: "#7a6cf0", lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
  chart.priceScale("atr").applyOptions({ scaleMargins: { top: 0.74, bottom: 0.14 } });
  const vwap = chart.addLineSeries({ color: "#c9a227", lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });

  let priceLines = [];
  let lastTime = null;

  const shift = (arr) => arr.map((x) => ({ ...x, time: local(x.time) }));

  function render(state, full) {
    if (full) {
      candles.setData(shift(state.candles));
      volume.setData(shift(state.volume));
      vwap.setData(shift(state.vwap));
      atr.setData(shift(state.atr));
      chart.timeScale().scrollToRealTime();
    } else {
      for (const c of shift(state.candles)) candles.update(c);
      for (const c of shift(state.volume)) volume.update(c);
      for (const c of shift(state.vwap)) vwap.update(c);
      for (const c of shift(state.atr)) atr.update(c);
    }
    if (state.candles.length) lastTime = state.candles[state.candles.length - 1].time;
    candles.setMarkers(shift(state.markers));
    for (const pl of priceLines) candles.removePriceLine(pl);
    priceLines = state.lines.map((l) => candles.createPriceLine({
      price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style, axisLabelVisible: true, title: l.title,
    }));
    chart.applyOptions({ watermark: { text: `${state.symbol} ${state.tf} · ${state.regime}` } });
  }

  return {
    render,
    lastTime: () => lastTime,
    reset: () => { lastTime = null; },
    screenshot: () => chart.takeScreenshot().toDataURL("image/png"),
  };
}
