// Test console for the Edge-AI camera trap. Talks to python/main.py through the WebUI brick.
const ui = new WebUI();
const $ = id => document.getElementById(id);
const pct = v => `${Math.round(v * 100)}%`;
const MAX_RESULTS = 8;

let state = null;

// ---- connection
ui.on_connect(() => { $('conn').textContent = 'Connected'; $('conn').className = 'pill on'; });
ui.on_disconnect(() => { $('conn').textContent = 'Disconnected'; $('conn').className = 'pill off'; });

// ---- state: mode, thresholds, models, statistics
ui.on_message('state', s => {
  state = s;
  document.querySelectorAll('#mode button').forEach(b => b.classList.toggle('active', b.dataset.mode === s.mode));
  $('mode-hint').textContent = {
    pir: 'Waiting for the PIR sensor.',
    timer: `Capturing every ${s.timer_s} s.`,
    manual: 'Only the button triggers a capture.',
  }[s.mode];

  if (document.activeElement !== $('det-thr')) $('det-thr').value = s.det_thr;
  if (document.activeElement !== $('cls-thr')) $('cls-thr').value = s.cls_thr;
  $('det-val').textContent = pct(s.det_thr);
  $('cls-val').textContent = pct(s.cls_thr);

  $('m-det').textContent = describe(s.detector);
  $('m-cls').textContent = describe(s.classifier);

  const st = s.stats || {};
  $('st-triggers').textContent = st.triggers || 0;
  $('st-frames').textContent = st.frames || 0;
  $('st-empty').textContent = st.empty || 0;
  $('st-animal').textContent = st.animal || 0;
  $('st-person').textContent = st.person || 0;
  $('st-vehicle').textContent = st.vehicle || 0;
  $('species-chips').innerHTML = (s.species || [])
    .map(([name, n]) => `<span class="chip">${esc(name)} × ${n}</span>`).join('');

  setBusy(s.busy);
  if (s.stream && !$('stream').src) $('stream').src = '/camera';
  $('no-stream').classList.toggle('hidden', s.stream);
  $('stream').classList.toggle('hidden', !s.stream);
  if (s.sensors) showSensors(s.sensors);
});

function describe(m) {
  if (!m) return '–';
  const labels = m.labels || [];
  const shown = labels.length > 6 ? `${labels.slice(0, 6).join(', ')} … (${labels.length})` : labels.join(', ');
  return `${m.name} (project ${m.project}) · ${m.input} · ${shown}`;
}

// ---- sensors
ui.on_message('sensors', showSensors);

function showSensors(s) {
  $('s-light').textContent = s.light;
  $('s-light-bar').style.width = `${Math.min(100, (s.light / 1023) * 100)}%`;
  $('s-night').textContent = s.night ? '🌙 Night' : '☀️ Day';
  $('s-pir').classList.toggle('active', s.pir);
  $('s-pir-text').textContent = s.pir ? 'Motion' : 'Still';
  $('s-ir').classList.toggle('active', s.ir);
  $('s-ir-text').textContent = s.ir ? 'On' : 'Off';
}

// ---- results
ui.on_message('busy', () => setBusy(true));

ui.on_message('result', r => {
  setBusy(false);
  $('no-results')?.remove();
  const el = document.createElement('div');
  el.className = 'result';
  el.innerHTML = `
    <div class="result-head">
      <b>${esc(r.time)}</b>
      <span class="tag">${esc(r.trigger)}</span>
      ${r.night ? '<span class="tag night">night</span>' : ''}
      ${r.ir ? '<span class="tag night">IR on</span>' : ''}
      ${r.light != null ? `<span class="tag">light ${r.light}</span>` : ''}
    </div>
    <div class="frames">${r.frames.map(frameHtml).join('')}</div>`;
  const list = $('results');
  list.prepend(el);
  while (list.children.length > MAX_RESULTS) list.lastElementChild.remove();
});

function frameHtml(f) {
  const dets = f.boxes.length
    ? f.boxes.map(b => `
        <div class="det">
          <span class="cls ${esc(b.label)}">${esc(b.label)}</span> ${pct(b.conf)}
          ${b.species ? `→ <b>${esc(b.species)}</b> ${pct(b.species_conf)}` : ''}
          ${b.top3 && b.top3.length ? `<span class="top3">top 3: ${b.top3.map(([s, c]) => `${esc(s)} ${pct(c)}`).join(', ')}</span>` : ''}
        </div>`).join('')
    : '<span class="empty">Nothing detected: frame discarded</span>';
  return `
    <div class="frame">
      ${f.preview ? `<img src="${f.preview}" alt="Frame ${f.n}" />` : ''}
      <div class="meta">${dets}<span class="hint">frame ${f.n} · ${f.took}s${f.saved ? ' · saved' : ''}</span></div>
    </div>`;
}

function setBusy(b) {
  $('busy').classList.toggle('hidden', !b);
  $('capture').disabled = b;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---- controls
$('capture').onclick = () => {
  ui.send_message('capture', { ir: $('capture-ir').checked });
  setBusy(true);
};

document.querySelectorAll('#mode button').forEach(b => {
  b.onclick = () => ui.send_message('set_mode', { mode: b.dataset.mode });
});

document.querySelectorAll('[data-ir]').forEach(b => {
  b.onclick = () => ui.send_message('ir', { on: b.dataset.ir === '1' });
});

for (const id of ['det-thr', 'cls-thr']) {
  $(id).oninput = () => {
    $('det-val').textContent = pct(+$('det-thr').value);
    $('cls-val').textContent = pct(+$('cls-thr').value);
  };
  $(id).onchange = () => ui.send_message('set_thresholds', {
    det_thr: +$('det-thr').value,
    cls_thr: +$('cls-thr').value,
  });
}
