// First-person controller. `position` is the EYE (camera) position; setPose takes the same.
// URL params honored: ?pos=x,y,z (eye) &yaw=r &pitch=r (camera frozen in place until you move),
// ?autopilot=1 (cinematic tour; &tour=k starts at waypoint k; any input takes over), ?fly=1.
// Keys: V toggles free-fly (noclip) for debugging (disable with opts.allowFly = false).
// level.spawn.position is the EYE position too (feet = y - CHAR.eye).
import * as THREE from 'three';
import { stepCharacter, CHAR } from './collision.js';
import { Input } from './input.js';

const WALK = 3.4, RUN = 6.4, JUMP_V = 5.7, TURN_RATE = 2.0;
const wrapPi = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };

export class Player {
  constructor(level, camera, domElement, opts = {}) {
    this.level = level;
    this.camera = camera;
    this.dom = domElement;
    this.col = level.collision;
    this.input = new Input(domElement, opts);
    camera.rotation.order = 'YXZ';
    const sp = level.spawn.position;
    this.st = { x: sp[0], y: sp[1] - CHAR.eye, z: sp[2], vx: 0, vy: 0, vz: 0, grounded: false };
    this.allowFly = opts.allowFly !== false;
    this._yaw = level.spawn.yaw || 0;
    this._pitch = 0;
    this._pos = new THREE.Vector3();
    this.eyeY = sp[1];
    this.bobPhase = 0; this.bobAmp = 0; this.dip = 0; this.dipV = 0; this.lean = 0;
    this.coyote = 0;
    this.safe = [];              // ring of recent safe positions
    this.safeT = 0;
    this.frozen = false;         // camera held at a URL pose until the user moves
    this.fly = false;
    this.autopilot = null;
    this.onRespawn = null;       // optional callback
    this.respawns = 0;

    const q = typeof location !== 'undefined' ? new URLSearchParams(location.search) : new URLSearchParams('');
    if (q.get('pos')) {
      const p = q.get('pos').split(',').map(Number);
      if (p.length === 3 && p.every(Number.isFinite)) {
        this.setPose(p, q.get('yaw') != null ? +q.get('yaw') : this._yaw, q.get('pitch') != null ? +q.get('pitch') : 0);
        this.frozen = true;
      }
    } else if (q.get('yaw') != null || q.get('pitch') != null) {
      this._yaw = +(q.get('yaw') || this._yaw); this._pitch = +(q.get('pitch') || 0);
    }
    if (q.get('fly') === '1') this.fly = true;
    if (q.get('autopilot') && q.get('autopilot') !== '0' && level.tour && level.tour.length > 1) {
      this.startAutopilot(+(q.get('tour') || 0));
    }
    this._baseActivity = this.input.activity;
    this._apply(0);
  }

  get position() { return this._pos; }
  get yaw() { return this._yaw; }
  get pitch() { return this._pitch; }
  get grounded() { return this.st.grounded; }
  get velocity() { return [this.st.vx, this.st.vy, this.st.vz]; }

  setPose(pos, yaw = this._yaw, pitch = this._pitch) {
    const x = pos.x != null ? pos.x : pos[0], y = pos.y != null ? pos.y : pos[1], z = pos.z != null ? pos.z : pos[2];
    this.st.x = x; this.st.y = y - CHAR.eye; this.st.z = z;
    this.st.vx = this.st.vy = this.st.vz = 0; this.st.grounded = false;
    this.eyeY = y; this._yaw = yaw; this._pitch = pitch;
    this._apply(0);
  }

  // ---------------------------------------------------------------- autopilot
  startAutopilot(startIndex = 0) {
    const pts = this.level.tour.map((w) => ({ p: new THREE.Vector3(w[0], w[1], w[2]), yaw: w[3], pitch: w[4] }));
    for (let i = 1; i < pts.length; i++) {           // unwrap yaw for smooth interpolation
      let d = pts[i].yaw - pts[i - 1].yaw;
      while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
      pts[i].yaw = pts[i - 1].yaw + d;
    }
    const segLen = [];
    for (let i = 0; i + 1 < pts.length; i++) segLen.push(Math.max(0.5, pts[i].p.distanceTo(pts[i + 1].p)));
    this.autopilot = { pts, segLen, seg: Math.min(Math.max(0, startIndex | 0), pts.length - 2), u: 0, t: 0, speed: 1.7 };
    this.frozen = false;
    this._baseActivity = this.input.activity;
  }
  stopAutopilot() {
    if (!this.autopilot) return;
    this.autopilot = null;
    const p = this.camera.position;
    this.setPose([p.x, p.y, p.z], this._yaw, this._pitch);
  }
  _autopilot(dt) {
    const A = this.autopilot, P = A.pts;
    A.t += dt;
    A.u += (A.speed * dt) / A.segLen[A.seg];
    while (A.u >= 1) { A.u -= 1; A.seg++; if (A.seg >= P.length - 1) { A.seg = 0; A.u = 0; } }
    const i = A.seg, u = A.u;
    const p0 = P[Math.max(0, i - 1)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(P.length - 1, i + 2)];
    const cr = (a, b, c, d) => {
      const u2 = u * u, u3 = u2 * u;
      return 0.5 * ((2 * b) + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2 + (-a + 3 * b - 3 * c + d) * u3);
    };
    const s = u * u * (3 - 2 * u);
    const x = cr(p0.p.x, p1.p.x, p2.p.x, p3.p.x), y = cr(p0.p.y, p1.p.y, p2.p.y, p3.p.y), z = cr(p0.p.z, p1.p.z, p2.p.z, p3.p.z);
    const yaw = p1.yaw + (p2.yaw - p1.yaw) * s + 0.05 * Math.sin(A.t * 0.31);
    const pitch = p1.pitch + (p2.pitch - p1.pitch) * s + 0.025 * Math.sin(A.t * 0.23 + 1);
    this._yaw = yaw; this._pitch = pitch;
    this.camera.position.set(x, y, z);
    this.camera.rotation.set(pitch, yaw, 0.012 * Math.sin(A.t * 0.17), 'YXZ');
    this._pos.copy(this.camera.position);
  }

  // ---------------------------------------------------------------- update
  update(dt) {
    dt = Math.min(Math.max(dt, 0), 0.05);
    const inp = this.input;
    if (inp.consumePressed('KeyV') && this.allowFly) { this.fly = !this.fly; this.frozen = false; if (this.autopilot) this.stopAutopilot(); }
    if (this.autopilot) {
      if (inp.activity !== this._baseActivity) this.stopAutopilot();
      else { this._autopilot(dt); inp.consumeLook(); inp.consumeJump(); inp.endFrame(); return; }
    }
    const [ldx, ldy] = inp.consumeLook();
    const ax = inp.axes();
    this._yaw -= ldx; this._pitch -= ldy;
    this._yaw += ax.turn * TURN_RATE * dt;
    this._yaw = wrapPi(this._yaw);
    this._pitch = Math.max(-1.48, Math.min(1.48, this._pitch));
    const jump = inp.consumeJump();
    inp.endFrame();
    if (this.frozen) {
      if (ax.x || ax.y || jump || ax.turn) this.frozen = false;
      else { this._apply(dt); return; }
    }
    const sy = Math.sin(this._yaw), cy = Math.cos(this._yaw);
    const fwd = [-sy, -cy], right = [cy, -sy];
    const st = this.st;

    if (this.fly) {
      const sp = ax.run ? 16 : 7;
      const cp = Math.cos(this._pitch), spch = Math.sin(this._pitch);
      st.x += (fwd[0] * cp * ax.y + right[0] * ax.x) * sp * dt;
      st.z += (fwd[1] * cp * ax.y + right[1] * ax.x) * sp * dt;
      st.y += (spch * ax.y + ax.up - ax.down) * sp * dt;
      st.vx = st.vy = st.vz = 0; st.grounded = false;
      this.eyeY = st.y + CHAR.eye;
      this._apply(dt);
      return;
    }

    const speed = ax.run ? RUN : WALK;
    const tx = (fwd[0] * ax.y + right[0] * ax.x) * speed, tz = (fwd[1] * ax.y + right[1] * ax.x) * speed;
    const k = st.grounded ? 11 : 2.2;
    const a = 1 - Math.exp(-k * dt);
    st.vx += (tx - st.vx) * a; st.vz += (tz - st.vz) * a;
    this.coyote = st.grounded ? 0.12 : Math.max(0, this.coyote - dt);
    if (jump && (st.grounded || this.coyote > 0)) { st.vy = JUMP_V; st.grounded = false; this.coyote = 0; }
    const vyBefore = st.vy, wasGrounded = st.grounded;
    stepCharacter(this.col, st, dt);
    if (!wasGrounded && st.grounded && vyBefore < -5) this.dipV -= Math.min(2.2, -vyBefore * 0.14);

    // safe-position memory + void respawn
    this.safeT += dt;
    if (st.grounded && this.safeT > 0.4) {
      this.safeT = 0;
      this.safe.push([st.x, st.y, st.z]);
      if (this.safe.length > 12) this.safe.shift();
    }
    if (st.y < this.col.killY) this.respawn();

    // eye smoothing (step-ups glide), landing dip, head bob, strafe lean
    const target = st.y + CHAR.eye;
    const d = target - this.eyeY;
    if (Math.abs(d) < 0.7 && st.grounded) this.eyeY += d * (1 - Math.exp(-(d > 0 ? 16 : 22) * dt));
    else this.eyeY = target;
    this.dipV += (-this.dip * 90 - this.dipV * 14) * dt;
    this.dip += this.dipV * dt;
    const hs = Math.hypot(st.vx, st.vz);
    this.bobAmp += ((st.grounded ? Math.min(1, hs / WALK) : 0) - this.bobAmp) * (1 - Math.exp(-6 * dt));
    this.bobPhase += dt * (1.6 + hs * 1.9);
    this.lean += ((-ax.x * 0.012) - this.lean) * (1 - Math.exp(-5 * dt));
    this._apply(dt);
  }

  respawn() {
    const sp = this.level.spawn.position;
    const s = this.safe.length > 3 ? this.safe[this.safe.length - 4] : this.safe[0] || [sp[0], sp[1] - CHAR.eye, sp[2]];
    this.st.x = s[0]; this.st.y = s[1] + 0.05; this.st.z = s[2];
    this.st.vx = this.st.vy = this.st.vz = 0; this.st.grounded = false;
    this.eyeY = this.st.y + CHAR.eye;
    this.safe.length = 0;
    this.respawns++;
    if (this.onRespawn) this.onRespawn();
  }

  _apply() {
    const sy = Math.sin(this._yaw), cy = Math.cos(this._yaw);
    const bob = this.bobAmp;
    const by = Math.sin(this.bobPhase * 2) * 0.032 * bob, bx = Math.cos(this.bobPhase) * 0.022 * bob;
    this.camera.position.set(this.st.x + cy * bx, this.eyeY + by + this.dip, this.st.z - sy * bx);
    this.camera.rotation.set(this._pitch + Math.sin(this.bobPhase * 2) * 0.004 * bob, this._yaw, this.lean + Math.cos(this.bobPhase) * 0.003 * bob, 'YXZ');
    this._pos.copy(this.camera.position);
  }

  dispose() { this.input.dispose(); }
}
