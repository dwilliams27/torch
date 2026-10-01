// DOM HUD: title screen status, zone title cards, tiny stats, settings panel, toasts.
const $ = (id) => document.getElementById(id);

export class Hud {
  constructor(settings, actions) {
    this.settings = settings;
    this.actions = actions;
    this.statsEl = $('stats');
    this.panelEl = $('panel');
    this.cardEl = $('zonecard');
    this.toastEl = $('toast');
    this.visible = true;
    this.panelOpen = false;
    this._cardTimer = null;
    this._toastTimer = null;
    this._buildPanel();
    $('gear')?.addEventListener('click', (e) => { e.stopPropagation(); this.togglePanel(); });
  }

  _buildPanel() {
    const S = this.settings;
    const sliders = [
      ['strength', 'dream strength', 0.1, 0.95, 0.01, (v) => v.toFixed(2)],
      ['feedback', 'feedback', 0, 0.95, 0.01, (v) => v.toFixed(2)],
      ['paintRate', 'paint rate', 0.05, 0.9, 0.01, (v) => v.toFixed(2)],
      ['captureRate', 'captures / s', 1, 30, 1, (v) => v.toFixed(0)],
      ['relaxHalfLife', 'memory (s)', 30, 1200, 10, (v) => v.toFixed(0)],
    ];
    const body = this.panelEl.querySelector('.panel-body');
    body.innerHTML = '';
    // looks: one tap switches a whole set of settings (looks.js)
    const looks = this.actions.looks || [];
    if (looks.length) {
      const lrow = document.createElement('div');
      lrow.className = 'row looks';
      lrow.innerHTML = looks.map((l, i) => `<button data-look="${l.id}" title="${l.hint} (${i + 1})">${l.id}</button>`).join('');
      lrow.addEventListener('click', (e) => { const id = e.target?.dataset?.look; if (id) this.actions.look?.(id); });
      lrow._sync = () => { for (const b of lrow.querySelectorAll('button')) b.classList.toggle('on', b.dataset.look === S.look); };
      body.appendChild(lrow);
    }
    for (const [key, label, min, max, step, fmt] of sliders) {
      const row = document.createElement('label');
      row.className = 'row';
      row.innerHTML = `<span class="lbl">${label}</span><input type="range" min="${min}" max="${max}" step="${step}"><span class="val"></span>`;
      const input = row.querySelector('input'), val = row.querySelector('.val');
      input.value = S[key];
      val.textContent = fmt(S[key]);
      input.addEventListener('input', () => { S[key] = parseFloat(input.value); val.textContent = fmt(S[key]); this.actions.onSetting?.(key); });
      row._sync = () => { input.value = S[key]; val.textContent = fmt(S[key]); };
      body.appendChild(row);
    }
    const prow = document.createElement('div');
    prow.className = 'row prompt';
    prow.innerHTML = `<span class="lbl">prompt override</span><textarea rows="3" placeholder="leave empty to use the zone's own dream"></textarea>`;
    const ta = prow.querySelector('textarea');
    ta.value = S.promptOverride || '';
    ta.addEventListener('input', () => { S.promptOverride = ta.value; });
    ta.addEventListener('keydown', (e) => e.stopPropagation());
    body.appendChild(prow);
    const brow = document.createElement('div');
    brow.className = 'row buttons';
    brow.innerHTML = `<button data-a="blink" style="grid-column: span 2">close your eyes</button><button data-a="forget">forget the dream</button><button data-a="compare">raw / dreamt</button><button data-a="audio">sound</button><button data-a="stats">stats</button><button data-a="close" style="grid-column: span 2">close</button>`;
    brow.addEventListener('click', (e) => {
      const a = e.target?.dataset?.a;
      if (a) e.target.blur?.();   // (a focused button would take the next Enter)
      if (a === 'blink') this.actions.blink?.();
      else if (a === 'forget') this.actions.forget?.();
      else if (a === 'compare') this.actions.compare?.();
      else if (a === 'audio') this.actions.audio?.();
      else if (a === 'stats') document.documentElement.classList.toggle('stats-off');
      else if (a === 'close') this.togglePanel(false);
    });
    body.appendChild(brow);
  }

  syncPanel() {
    for (const row of this.panelEl.querySelectorAll('.row')) row._sync?.();
  }

  togglePanel(force) {
    this.panelOpen = force ?? !this.panelOpen;
    this.panelEl.classList.toggle('open', this.panelOpen);
    if (this.panelOpen) { this.syncPanel(); document.exitPointerLock?.(); }
    this.actions.onPanel?.(this.panelOpen);
  }

  toggleVisible() {
    this.visible = !this.visible;
    document.body.classList.toggle('hud-hidden', !this.visible);
  }

  zoneCard(zone) {
    if (!zone) return;
    const el = this.cardEl;
    el.querySelector('.name').textContent = zone.name || '';
    el.querySelector('.sub').textContent = zone.subtitle || '';
    el.classList.remove('show');
    void el.offsetWidth;
    el.classList.add('show');
    clearTimeout(this._cardTimer);
    this._cardTimer = setTimeout(() => el.classList.remove('show'), 6500);
  }

  toast(msg) {
    const el = this.toastEl;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => el.classList.remove('show'), 1800);
  }

  setStats(lines) {
    if (!this.visible) return;
    this.statsEl.textContent = lines.join('\n');
  }
}
