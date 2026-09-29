// Unified desktop + touch input for the first-person controller.
// Desktop: WASD/arrows, Shift run, Space jump, pointer-lock mouse look (click the canvas).
// Touch: left half = floating virtual stick, right half = drag to look, quick tap on right = jump.

const MOVE_CODES = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space', 'ShiftLeft', 'ShiftRight', 'KeyQ', 'KeyE', 'KeyC']);

export class Input {
  constructor(dom, opts = {}) {
    this.dom = dom;
    this.enabled = true;
    this.keys = new Set();
    this.lookDX = 0; this.lookDY = 0;
    this.stick = { id: null, ox: 0, oy: 0, x: 0, y: 0 };
    this.look = { id: null, x: 0, y: 0, t0: 0, moved: 0 };
    this.jumpQueued = false;
    this.activity = 0;           // bumps on any user input (autopilot cancel)
    this.pressed = new Set();    // edge-triggered keys this frame
    this.isTouch = typeof window !== 'undefined' && ('ontouchstart' in window || navigator.maxTouchPoints > 0);
    this.mouseSensitivity = opts.mouseSensitivity || 0.0022;
    this.touchSensitivity = opts.touchSensitivity || 0.0055;
    this._h = [];
    const on = (t, ev, fn, o) => { t.addEventListener(ev, fn, o); this._h.push([t, ev, fn, o]); };

    on(window, 'keydown', (e) => {
      if (!this.enabled || e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (MOVE_CODES.has(e.code)) e.preventDefault();
      if (!e.repeat) this.pressed.add(e.code);
      if (e.code === 'Space' && !e.repeat) this.jumpQueued = true;
      this.keys.add(e.code);
      if (MOVE_CODES.has(e.code)) this.activity++;
    });
    on(window, 'keyup', (e) => { this.keys.delete(e.code); });
    on(window, 'blur', () => this.keys.clear());
    if (opts.pointerLock !== false) {
      on(dom, 'click', () => {
        if (!this.enabled || this.isTouch) return;
        if (document.pointerLockElement !== dom && dom.requestPointerLock) {
          try { const p = dom.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch (_) { /* ignore */ }
        }
      });
    }
    on(document, 'mousemove', (e) => {
      if (!this.enabled || document.pointerLockElement !== dom) return;
      // guard against the occasional huge spike some browsers emit on lock
      if (Math.abs(e.movementX) > 400 || Math.abs(e.movementY) > 400) return;
      this.lookDX += e.movementX * this.mouseSensitivity;
      this.lookDY += e.movementY * this.mouseSensitivity;
      this.activity++;
    });
    on(dom, 'mousedown', () => { this.activity++; });

    // --- touch
    if (this.isTouch) {
      this._stickEl = document.createElement('div');
      this._knobEl = document.createElement('div');
      Object.assign(this._stickEl.style, {
        position: 'fixed', width: '116px', height: '116px', marginLeft: '-58px', marginTop: '-58px', borderRadius: '50%',
        border: '1.5px solid rgba(255,255,255,0.28)', background: 'rgba(255,255,255,0.05)', pointerEvents: 'none',
        display: 'none', zIndex: 20, backdropFilter: 'blur(2px)',
      });
      Object.assign(this._knobEl.style, {
        position: 'absolute', left: '38px', top: '38px', width: '40px', height: '40px', borderRadius: '50%',
        background: 'rgba(255,255,255,0.32)', pointerEvents: 'none',
      });
      this._stickEl.appendChild(this._knobEl);
      document.body.appendChild(this._stickEl);
      const opt = { passive: false };
      on(dom, 'touchstart', (e) => this._touchStart(e), opt);
      on(dom, 'touchmove', (e) => this._touchMove(e), opt);
      on(dom, 'touchend', (e) => this._touchEnd(e), opt);
      on(dom, 'touchcancel', (e) => this._touchEnd(e), opt);
    }
  }

  _touchStart(e) {
    if (!this.enabled) return;
    e.preventDefault();
    this.activity++;
    const w = window.innerWidth;
    for (const t of e.changedTouches) {
      if (t.clientX < w * 0.45 && this.stick.id === null) {
        this.stick = { id: t.identifier, ox: t.clientX, oy: t.clientY, x: 0, y: 0 };
        this._stickEl.style.display = 'block';
        this._stickEl.style.left = t.clientX + 'px';
        this._stickEl.style.top = t.clientY + 'px';
        this._knobEl.style.transform = 'translate(0px,0px)';
      } else if (this.look.id === null) {
        this.look = { id: t.identifier, x: t.clientX, y: t.clientY, t0: performance.now(), moved: 0 };
      }
    }
  }
  _touchMove(e) {
    if (!this.enabled) return;
    e.preventDefault();
    for (const t of e.changedTouches) {
      if (t.identifier === this.stick.id) {
        let dx = t.clientX - this.stick.ox, dy = t.clientY - this.stick.oy;
        const R = 50, l = Math.hypot(dx, dy);
        if (l > R) { dx *= R / l; dy *= R / l; }
        this.stick.x = dx / R; this.stick.y = dy / R;
        this._knobEl.style.transform = `translate(${dx}px,${dy}px)`;
      } else if (t.identifier === this.look.id) {
        const dx = t.clientX - this.look.x, dy = t.clientY - this.look.y;
        this.look.x = t.clientX; this.look.y = t.clientY;
        this.look.moved += Math.abs(dx) + Math.abs(dy);
        this.lookDX += dx * this.touchSensitivity;
        this.lookDY += dy * this.touchSensitivity;
      }
    }
  }
  _touchEnd(e) {
    for (const t of e.changedTouches) {
      if (t.identifier === this.stick.id) {
        this.stick = { id: null, ox: 0, oy: 0, x: 0, y: 0 };
        this._stickEl.style.display = 'none';
      } else if (t.identifier === this.look.id) {
        if (performance.now() - this.look.t0 < 220 && this.look.moved < 12) this.jumpQueued = true;
        this.look.id = null;
      }
    }
  }

  // movement axes: x = strafe right, y = forward; run flag; turn = keyboard yaw rate (-1..1)
  axes() {
    const k = this.keys;
    let x = 0, y = 0, turn = 0;
    if (k.has('KeyW') || k.has('ArrowUp')) y += 1;
    if (k.has('KeyS') || k.has('ArrowDown')) y -= 1;
    if (k.has('KeyD')) x += 1;
    if (k.has('KeyA')) x -= 1;
    if (k.has('ArrowRight')) turn -= 1;
    if (k.has('ArrowLeft')) turn += 1;
    let run = k.has('ShiftLeft') || k.has('ShiftRight');
    if (this.stick.id !== null) {
      x += this.stick.x; y -= this.stick.y;
      if (Math.hypot(this.stick.x, this.stick.y) > 0.92) run = true;
    }
    const l = Math.hypot(x, y);
    if (l > 1) { x /= l; y /= l; }
    return { x, y, run, turn, up: k.has('Space') ? 1 : 0, down: k.has('KeyC') || k.has('ControlLeft') ? 1 : 0 };
  }
  consumeLook() { const r = [this.lookDX, this.lookDY]; this.lookDX = 0; this.lookDY = 0; return r; }
  consumeJump() { const j = this.jumpQueued; this.jumpQueued = false; return j; }
  consumePressed(code) { const h = this.pressed.has(code); this.pressed.delete(code); return h; }
  endFrame() { this.pressed.clear(); }

  dispose() {
    for (const [t, ev, fn, o] of this._h) t.removeEventListener(ev, fn, o);
    if (this._stickEl) this._stickEl.remove();
  }
}
