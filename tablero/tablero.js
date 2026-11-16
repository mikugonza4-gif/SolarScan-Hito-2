// Tablero de SolarScan: combina el historial de la serie temporal (/serie/historial) con el
// registro de recomendaciones que el flujo de n8n escribe en Google Sheets (/api/registros).
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => Number(v).toLocaleString('es-CL', { minimumFractionDigits: d, maximumFractionDigits: d });
const COLOR = { s1: '#2a78d6', s2: '#eb6834', umbral: '#8a948f', ALTA: '#d03b3b', MEDIA: '#fab219', BAJA: '#0ca30c' };
const NOMBRES = { 'Bird-drop': 'Excrementos de aves', 'Clean': 'Limpio', 'Dusty': 'Polvo',
  'Electrical-damage': 'Daño eléctrico', 'Physical-Damage': 'Daño físico', 'Snow-Covered': 'Nieve' };
let SERIE = [], REG = [], UMBRAL = 0.9, graficos = {};

Chart.defaults.font.family = 'system-ui,-apple-system,"Segoe UI",sans-serif';
Chart.defaults.color = '#5c7067';
Chart.defaults.borderColor = '#e6ece6';
Chart.defaults.locale = 'es-CL';

async function cargar() {
  const [hist, info, reg] = await Promise.all([
    fetch('/serie/historial').then(r => r.json()),
    fetch('/serie/info').then(r => r.json()),
    fetch('/api/registros').then(r => r.json()).catch(() => ({ configurado: false, registros: [] })),
  ]);
  SERIE = hist; UMBRAL = info.umbral; REG = reg.registros || [];
  info.inversores.forEach(i => $('f-inversor').add(new Option(i, i)));
  Object.entries(NOMBRES).forEach(([k, v]) => $('f-clase').add(new Option(v, k)));
  $('f-desde').value = $('f-desde').min = $('f-hasta').min = info.fecha_inicio;
  $('f-hasta').value = $('f-desde').max = $('f-hasta').max = info.fecha_fin;
  if (!reg.configurado) mostrarAviso('El registro de recomendaciones aún no está conectado (falta TABLERO_CSV_URL en Render).');
  else if (reg.error) mostrarAviso(reg.error);
  ['f-desde', 'f-hasta', 'f-inversor', 'f-prioridad', 'f-clase'].forEach(id => $(id).addEventListener('change', dibujar));
  $('f-reset').addEventListener('click', () => {
    $('f-desde').value = info.fecha_inicio; $('f-hasta').value = info.fecha_fin;
    ['f-inversor', 'f-prioridad', 'f-clase'].forEach(id => $(id).value = ''); dibujar();
  });
  dibujar();
}

function mostrarAviso(t) { const a = $('aviso-registros'); a.textContent = t; a.hidden = false; }

function filtros() {
  return { desde: $('f-desde').value, hasta: $('f-hasta').value, inv: $('f-inversor').value,
           prio: $('f-prioridad').value, clase: $('f-clase').value };
}

function dibujar() {
  const f = filtros();
  const serie = SERIE.filter(r => r.fecha >= f.desde && r.fecha <= f.hasta && (!f.inv || r.inversor === f.inv));
  const reg = REG.filter(r => (r.fecha_operacion || '') >= f.desde && (r.fecha_operacion || '') <= f.hasta &&
    (!f.inv || r.inversor === f.inv) && (!f.prio || r.prioridad === f.prio) && (!f.clase || r.clase_imagen === f.clase));
  kpis(serie, reg); graficoIndice(serie, f); graficoPerdida(serie); graficoPrioridad(reg); mapa(f); tabla(reg);
}

function kpis(serie, reg) {
  const anom = serie.filter(r => r.anomalia);
  const perdida = anom.reduce((s, r) => s + Math.max(r.energia_esperada_kwh - r.energia_real_kwh, 0), 0);
  $('k-perdida').textContent = fmt(perdida);
  $('k-anom').textContent = fmt(anom.length);
  $('k-anom-nota').textContent = `de ${fmt(serie.length)} días-inversor (umbral ${fmt(UMBRAL * 100)} %)`;
  $('k-insp').textContent = fmt(reg.length);
  const ok = reg.filter(r => String(r.controles_ok).toLowerCase() === 'true').length;
  $('k-insp-nota').textContent = reg.length ? `${fmt(100 * ok / reg.length)} % pasó los controles del LLM` : 'desde el flujo n8n';
  const ap = reg.filter(r => r.estado_aprobacion === 'APROBADA').length;
  const pend = reg.filter(r => ['RECHAZADA', 'SIN RESPUESTA'].includes(r.estado_aprobacion)).length;
  $('k-ot').textContent = fmt(ap);
  $('k-ot-nota').textContent = `aprobadas · ${fmt(pend)} rechazadas o sin respuesta`;
}

function nuevo(id, config) {
  if (graficos[id]) graficos[id].destroy();
  graficos[id] = new Chart($(id), config);
}

const ejeTiempo = { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 10 } };

function graficoIndice(serie, f) {
  const fechas = [...new Set(serie.map(r => r.fecha))].sort();
  let datasets;
  if (f.inv) {
    const por = Object.fromEntries(serie.map(r => [r.fecha, r]));
    $('sub-indice').textContent = `${f.inv}: regresión (decide) y LSTM (segunda opinión). Bajo ${fmt(UMBRAL * 100)} % se marca anomalía.`;
    datasets = [
      { label: 'Regresión lineal (decide)', data: fechas.map(d => por[d] ? por[d].indice * 100 : null), borderColor: COLOR.s1, backgroundColor: COLOR.s1 },
      { label: 'LSTM (segunda opinión)', data: fechas.map(d => por[d] && por[d].indice_lstm ? por[d].indice_lstm * 100 : null), borderColor: COLOR.s2, backgroundColor: COLOR.s2 }];
  } else {
    const med = fechas.map(d => { const v = serie.filter(r => r.fecha === d).map(r => r.indice).sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] * 100 : null; });
    const min = fechas.map(d => { const v = serie.filter(r => r.fecha === d).map(r => r.indice); return v.length ? Math.min(...v) * 100 : null; });
    $('sub-indice').textContent = `Planta completa: mediana de los 22 inversores y el inversor más bajo de cada día. Bajo ${fmt(UMBRAL * 100)} % se marca anomalía.`;
    datasets = [
      { label: 'Mediana de la planta', data: med, borderColor: COLOR.s1, backgroundColor: COLOR.s1 },
      { label: 'Inversor más bajo del día', data: min, borderColor: COLOR.s2, backgroundColor: COLOR.s2 }];
  }
  datasets.forEach(d => Object.assign(d, { borderWidth: 2, pointRadius: 2.5, pointHoverRadius: 5, tension: 0.2, spanGaps: true }));
  datasets.push({ label: `Umbral ${fmt(UMBRAL * 100)} %`, data: fechas.map(() => UMBRAL * 100), borderColor: COLOR.umbral,
                  borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 0 });
  nuevo('g-indice', { type: 'line', data: { labels: fechas, datasets },
    options: { maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top', align: 'start', labels: { boxWidth: 12, usePointStyle: true } },
                 tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y == null ? 's/d' : fmt(c.parsed.y, 1) + ' %'}` } } },
      scales: { x: ejeTiempo, y: { title: { display: true, text: 'Índice de rendimiento (%)' }, suggestedMin: 0, suggestedMax: 120 } } } });
}

function graficoPerdida(serie) {
  const fechas = [...new Set(serie.map(r => r.fecha))].sort();
  const val = fechas.map(d => serie.filter(r => r.fecha === d && r.anomalia)
    .reduce((s, r) => s + Math.max(r.energia_esperada_kwh - r.energia_real_kwh, 0), 0));
  nuevo('g-perdida', { type: 'bar', data: { labels: fechas, datasets: [{ label: 'Energía no generada', data: val,
      backgroundColor: COLOR.s1, borderRadius: 4, maxBarThickness: 18 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => `${fmt(c.parsed.y)} kWh` } } },
      scales: { x: ejeTiempo, y: { title: { display: true, text: 'kWh' }, beginAtZero: true } } } });
}

function graficoPrioridad(reg) {
  const fechas = [...new Set(reg.map(r => r.fecha_operacion))].sort();
  const datasets = ['ALTA', 'MEDIA', 'BAJA'].map(p => ({ label: p, data: fechas.map(d => reg.filter(r => r.fecha_operacion === d && r.prioridad === p).length),
    backgroundColor: COLOR[p], borderColor: '#ffffff', borderWidth: 2, borderRadius: 4, maxBarThickness: 28 }));
  nuevo('g-prioridad', { type: 'bar', data: { labels: fechas, datasets },
    options: { maintainAspectRatio: false, plugins: { legend: { position: 'top', align: 'start', labels: { boxWidth: 12 } } },
      scales: { x: { ...ejeTiempo, stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 },
        title: { display: true, text: 'Recomendaciones' } } } } });
}

function colorCelda(v) {
  if (v >= UMBRAL) return '#eef2ee';
  if (v >= 0.75) return '#f6c9b8';
  if (v >= 0.6) return '#ec835a';
  if (v >= 0.4) return '#d03b3b';
  return '#8f1f1f';
}

function mapa(f) {
  const datos = SERIE.filter(r => r.fecha >= f.desde && r.fecha <= f.hasta);
  const fechas = [...new Set(datos.map(r => r.fecha))].sort();
  const invs = [...new Set(SERIE.map(r => r.inversor))].sort().filter(i => !f.inv || i === f.inv);
  const idx = Object.fromEntries(datos.map(r => [r.inversor + r.fecha, r]));
  let h = '<thead><tr><th></th>' + fechas.map((d, i) => `<th>${i % 3 === 0 ? d.slice(5) : ''}</th>`).join('') + '</tr></thead><tbody>';
  for (const inv of invs) {
    h += `<tr><th class="fila">${inv}</th>` + fechas.map(d => {
      const r = idx[inv + d];
      if (!r) return `<td class="sin" data-t="${inv} · ${d}: sin datos"></td>`;
      const t = `${inv} · ${d}: ${fmt(r.indice * 100, 1)} % de lo esperado` + (r.anomalia ? ` · ${fmt(Math.max(r.energia_esperada_kwh - r.energia_real_kwh, 0))} kWh no generados` : '');
      return `<td style="background:${colorCelda(r.indice)}" data-t="${t}"></td>`;
    }).join('') + '</tr>';
  }
  $('mapa').innerHTML = h + '</tbody>';
}

function tabla(reg) {
  const filas = [...reg].sort((a, b) => (b.fecha_registro_utc || '').localeCompare(a.fecha_registro_utc || '')).slice(0, 60);
  if (!filas.length) { $('t-registro').innerHTML = '<tr><td colspan="9">Sin recomendaciones para los filtros elegidos.</td></tr>'; return; }
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  $('t-registro').innerHTML = filas.map(r => `<tr>
    <td>${esc(r.fecha_operacion)}</td><td>${esc(r.inversor)}</td><td>${esc(NOMBRES[r.clase_imagen] || r.clase_imagen)}</td>
    <td>${r.indice_rendimiento ? fmt(Number(r.indice_rendimiento) * 100, 1) + ' %' : 's/d'}</td>
    <td><span class="badge ${esc(r.prioridad)}">${esc(r.prioridad)}</span></td><td>${esc(r.accion)}</td>
    <td>${esc(r.estado_aprobacion)}${r.ot_id ? '<br><small>' + esc(r.ot_id) + '</small>' : ''}</td>
    <td class="fuentes">${esc(r.fuentes) || '—'}</td><td>${String(r.controles_ok).toLowerCase() === 'true' ? 'Aprobados' : 'Respaldo'}</td></tr>`).join('');
}

// Tooltip del mapa de calor.
const tip = $('tooltip');
document.addEventListener('mousemove', e => {
  const t = e.target.closest && e.target.closest('td[data-t]');
  if (!t) { tip.hidden = true; return; }
  tip.textContent = t.dataset.t; tip.hidden = false;
  tip.style.left = Math.min(e.clientX + 12, innerWidth - 250) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
});

cargar().catch(err => { $('t-registro').innerHTML = `<tr><td colspan="9">No se pudo cargar el tablero: ${err.message}</td></tr>`; });
