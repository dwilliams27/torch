// Minimal fly camera (used with the test level / when the world Player is unavailable).
// Same surface as Player: update(dt), position, yaw, pitch, setPose().
import * as THREE from 'three';

export class FlyCam {
  constructor(level, camera, dom) {
    this.camera = camera;
    this.dom = dom;
    this.pos = new THREE.Vector3(...(level.spawn?.position || [0, 1.6, 0]));
    this._yaw = level.spawn?.yaw || 0;
    this._pitch = 0;
    this.keys = new Set();
    this.enabled = true;
    this.vel = new THREE.Vector3();
    this.touch = { move: null, look: null };
    window.addEventListener('keydown', (e) => { if (!e.target.closest?.('textarea,input')) this.keys.add(e.code); });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
    document.addEventListener('mousemove', (e) => {
      if (document.pointerLockElement !== dom || !this.enabled) return;
      this._yaw -= e.movementX * 0.0022;
      this._pitch = THREE.MathUtils.clamp(this._pitch - e.movementY * 0.0022, -1.45, 1.45);
    });
    dom.addEventListener('touchstart', (e) => this._touch(e), { passive: false });
    dom.addEventListener('touchmove', (e) => this._touch(e), { passive: false });
    dom.addEventListener('touchend', (e) => this._touch(e), { passive: false });
    this._apply();
  }
  _touch(e) {
    e.preventDefault();
    const w = window.innerWidth;
    const seen = new Set();
    for (const t of e.touches) {
      seen.add(t.identifier);
      const left = t.clientX < w / 2;
      const slot = left ? 'move' : 'look';
      const s = this.touch[slot];
      if (!s || s.id !== t.identifier) { if (!s) this.touch[slot] = { id: t.identifier, x0: t.clientX, y0: t.clientY, x: t.clientX, y: t.clientY }; continue; }
      if (slot === 'look') {
        this._yaw -= (t.clientX - s.x) * 0.005;
        this._pitch = THREE.MathUtils.clamp(this._pitch - (t.clientY - s.y) * 0.005, -1.4, 1.4);
      }
      s.x = t.clientX; s.y = t.clientY;
    }
    for (const k of ['move', 'look']) if (this.touch[k] && !seen.has(this.touch[k].id)) this.touch[k] = null;
  }
  get position() { return this.pos; }
  get yaw() { return this._yaw; }
  get pitch() { return this._pitch; }
  setPose(p, yaw, pitch) { this.pos.copy(p); this._yaw = yaw; this._pitch = pitch ?? 0; this._apply(); }
  update(dt) {
    const k = this.keys;
    const f = new THREE.Vector3(-Math.sin(this._yaw) * Math.cos(this._pitch), Math.sin(this._pitch), -Math.cos(this._yaw) * Math.cos(this._pitch));
    const r = new THREE.Vector3(Math.cos(this._yaw), 0, -Math.sin(this._yaw));
    const want = new THREE.Vector3();
    if (this.enabled) {
      if (k.has('KeyW') || k.has('ArrowUp')) want.add(f);
      if (k.has('KeyS') || k.has('ArrowDown')) want.sub(f);
      if (k.has('KeyD')) want.add(r);
      if (k.has('KeyA')) want.sub(r);
      if (k.has('ArrowLeft')) this._yaw += dt * 1.8;
      if (k.has('ArrowRight')) this._yaw -= dt * 1.8;
      if (k.has('KeyE') || k.has('Space')) want.y += 1;
      if (k.has('KeyQ') || k.has('KeyC')) want.y -= 1;
      const m = this.touch.move;
      if (m) {
        const dx = (m.x - m.x0) / 60, dy = (m.y - m.y0) / 60;
        want.addScaledVector(r, THREE.MathUtils.clamp(dx, -1, 1));
        want.addScaledVector(f, THREE.MathUtils.clamp(-dy, -1, 1));
      }
    }
    const speed = (k.has('ShiftLeft') || k.has('ShiftRight')) ? 9 : 4;
    if (want.lengthSq() > 1) want.normalize();
    want.multiplyScalar(speed);
    this.vel.lerp(want, 1 - Math.exp(-dt * 8));
    this.pos.addScaledVector(this.vel, dt);
    this._apply();
  }
  _apply() {
    this.camera.position.copy(this.pos);
    this.camera.rotation.set(this._pitch, this._yaw, 0, 'YXZ');
  }
}
