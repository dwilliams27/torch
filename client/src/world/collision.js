// Collision world built from the same primitives as the visible geometry.
// Pure JS (no three.js) so node tools can simulate the exact same character physics.
//
// Primitive types:
//   0 BOX   yaw-rotated box: center (cx,cz), half extents (hx,hz) in local frame, y0..y1
//   1 RAMP  yaw-rotated wedge: top rises linearly along local +x from yl (x=-hx) to yh (x=+hx), bottom y0
//   2 CYL   vertical cylinder: center (cx,cz), radius r, y0..y1
//
// Yaw convention (three.js rotation about +Y): world = c + R(yaw)·local
//   dx = lx*cos + lz*sin ; dz = -lx*sin + lz*cos

export const BOX = 0, RAMP = 1, CYL = 2;

export const CHAR = {
  radius: 0.3,     // body radius (m)
  height: 1.75,    // body height (m)
  eye: 1.6,        // eye height above feet
  step: 0.45,      // max step-up
  gravity: 18,     // m/s^2
  maxFall: 40,     // terminal velocity
};

export class CollisionWorld {
  constructor(cell = 4) {
    this.cell = cell;
    this.prims = [];
    this.grid = new Map();
    this.stamp = 0;
    this.killY = -60;
    this._tmp = [];
  }

  addBox(cx, cy, cz, hx, hy, hz, yaw = 0) {
    const p = { type: BOX, cx, cz, hx, hz, c: Math.cos(yaw), s: Math.sin(yaw), y0: cy - hy, y1: cy + hy, yl: 0, yh: 0, r: 0, q: 0 };
    return this._add(p);
  }
  addRamp(cx, cz, hx, hz, yaw, yLow, yHigh, yBase) {
    const p = { type: RAMP, cx, cz, hx, hz, c: Math.cos(yaw), s: Math.sin(yaw), y0: yBase, y1: Math.max(yLow, yHigh), yl: yLow, yh: yHigh, r: 0, q: 0 };
    return this._add(p);
  }
  addCyl(cx, cz, r, y0, y1) {
    const p = { type: CYL, cx, cz, hx: r, hz: r, c: 1, s: 0, y0, y1, yl: 0, yh: 0, r, q: 0 };
    return this._add(p);
  }

  _add(p) {
    if (p.type === CYL) {
      p.minx = p.cx - p.r; p.maxx = p.cx + p.r; p.minz = p.cz - p.r; p.maxz = p.cz + p.r;
    } else {
      const ex = Math.abs(p.hx * p.c) + Math.abs(p.hz * p.s);
      const ez = Math.abs(p.hx * p.s) + Math.abs(p.hz * p.c);
      p.minx = p.cx - ex; p.maxx = p.cx + ex; p.minz = p.cz - ez; p.maxz = p.cz + ez;
    }
    p.id = this.prims.length;
    this.prims.push(p);
    const C = this.cell;
    const i0 = Math.floor(p.minx / C), i1 = Math.floor(p.maxx / C);
    const k0 = Math.floor(p.minz / C), k1 = Math.floor(p.maxz / C);
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const key = i * 73856093 ^ k * 19349663;
      let arr = this.grid.get(key);
      if (!arr) { arr = []; this.grid.set(key, arr); }
      arr.push(p);
    }
    return p;
  }

  // prims whose XZ bounds overlap the disc (x,z,r). Returned array is reused.
  query(x, z, r) {
    const out = this._tmp; out.length = 0;
    const st = ++this.stamp;
    const C = this.cell;
    const i0 = Math.floor((x - r) / C), i1 = Math.floor((x + r) / C);
    const k0 = Math.floor((z - r) / C), k1 = Math.floor((z + r) / C);
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const arr = this.grid.get(i * 73856093 ^ k * 19349663);
      if (!arr) continue;
      for (let j = 0; j < arr.length; j++) {
        const p = arr[j];
        if (p.q === st) continue;
        p.q = st;
        if (p.maxx < x - r || p.minx > x + r || p.maxz < z - r || p.minz > z + r) continue;
        out.push(p);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- queries
  groundHeight(x, z, maxTop, r = CHAR.radius * 0.55) {
    const ps = this.query(x, z, r);
    let best = -Infinity;
    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      if (!overlap(p, x, z, r)) continue;
      const t = topAt(p, x, z);
      if (t <= maxTop && t > best) best = t;
    }
    return best;
  }

  ceilingHeight(x, z, minBottom, r = CHAR.radius * 0.8) {
    const ps = this.query(x, z, r);
    let best = Infinity;
    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      if (p.y0 < minBottom || p.y0 >= best) continue;
      if (overlap(p, x, z, r)) best = p.y0;
    }
    return best;
  }

  // true if a standing body at feet (x,y,z) intersects any solid
  blocked(x, y, z, r = CHAR.radius) {
    const ps = this.query(x, z, r);
    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      if (p.y0 >= y + CHAR.height) continue;
      if (topAt(p, x, z) <= y + 0.02) continue;
      if (overlap(p, x, z, r)) return true;
    }
    return false;
  }

  // Push the body out of walls (anything taller than a step at this height).
  resolveWalls(st) {
    const R = CHAR.radius;
    for (let iter = 0; iter < 4; iter++) {
      let moved = false;
      const ps = this.query(st.x, st.z, R);
      for (let i = 0; i < ps.length; i++) {
        const p = ps[i];
        if (p.y0 >= st.y + CHAR.height - 0.02) continue;           // entirely above head
        if (topAt(p, st.x, st.z) <= st.y + CHAR.step) continue;   // a step / floor, not a wall
        const d = pushOut(p, st.x, st.z, R);
        if (d) {
          st.x += d[0]; st.z += d[1]; moved = true;
          // kill velocity into the wall (slide along it)
          const len = Math.hypot(d[0], d[1]);
          if (len > 1e-9) {
            const nx = d[0] / len, nz = d[1] / len;
            const vn = st.vx * nx + st.vz * nz;
            if (vn < 0) { st.vx -= vn * nx; st.vz -= vn * nz; }
          }
        }
      }
      if (!moved) break;
    }
  }
}

// ------------------------------------------------------------------ helpers
export function topAt(p, x, z) {
  if (p.type !== RAMP) return p.y1;
  const dx = x - p.cx, dz = z - p.cz;
  let lx = dx * p.c - dz * p.s;
  if (lx < -p.hx) lx = -p.hx; else if (lx > p.hx) lx = p.hx;
  return p.yl + (p.yh - p.yl) * ((lx + p.hx) / (2 * p.hx));
}

export function overlap(p, x, z, r) {
  if (p.type === CYL) {
    const dx = x - p.cx, dz = z - p.cz, rr = p.r + r;
    return dx * dx + dz * dz < rr * rr;
  }
  const dx = x - p.cx, dz = z - p.cz;
  const lx = dx * p.c - dz * p.s, lz = dx * p.s + dz * p.c;
  const qx = lx < -p.hx ? -p.hx : lx > p.hx ? p.hx : lx;
  const qz = lz < -p.hz ? -p.hz : lz > p.hz ? p.hz : lz;
  const ex = lx - qx, ez = lz - qz;
  return ex * ex + ez * ez < r * r;
}

// minimal XZ displacement separating a disc from the primitive, or null
export function pushOut(p, x, z, r) {
  if (p.type === CYL) {
    const dx = x - p.cx, dz = z - p.cz, rr = p.r + r;
    const d2 = dx * dx + dz * dz;
    if (d2 >= rr * rr) return null;
    const d = Math.sqrt(d2);
    if (d < 1e-6) return [rr, 0];
    const k = (rr - d) / d;
    return [dx * k, dz * k];
  }
  const dx = x - p.cx, dz = z - p.cz;
  const lx = dx * p.c - dz * p.s, lz = dx * p.s + dz * p.c;
  const qx = lx < -p.hx ? -p.hx : lx > p.hx ? p.hx : lx;
  const qz = lz < -p.hz ? -p.hz : lz > p.hz ? p.hz : lz;
  let ex = lx - qx, ez = lz - qz;
  const d2 = ex * ex + ez * ez;
  let mx, mz;
  if (d2 > 1e-12) {
    if (d2 >= r * r) return null;
    const d = Math.sqrt(d2), k = (r - d) / d;
    mx = ex * k; mz = ez * k;
  } else {
    // center inside the rectangle: exit via the nearest side
    const px = p.hx - Math.abs(lx) + r, pz = p.hz - Math.abs(lz) + r;
    if (px < pz) { mx = lx >= 0 ? px : -px; mz = 0; }
    else { mx = 0; mz = lz >= 0 ? pz : -pz; }
  }
  // local -> world direction
  return [mx * p.c + mz * p.s, -mx * p.s + mz * p.c];
}

// One physics step of the character. st = {x,y,z,vx,vy,vz,grounded}. Horizontal velocity is
// whatever the controller wants; this integrates gravity, resolves walls, steps and ceilings.
export function stepCharacter(world, st, dt) {
  const H = CHAR.height, STEP = CHAR.step;
  const hs = Math.hypot(st.vx, st.vz) * dt, vs = Math.abs(st.vy) * dt + CHAR.gravity * dt * dt;
  const n = Math.min(12, Math.max(1, Math.ceil(Math.max(hs, vs) / 0.1)));
  const h = dt / n;
  for (let i = 0; i < n; i++) {
    const wasGrounded = st.grounded;
    st.x += st.vx * h; st.z += st.vz * h;
    world.resolveWalls(st);
    st.vy -= CHAR.gravity * h;
    if (st.vy < -CHAR.maxFall) st.vy = -CHAR.maxFall;
    let ny = st.y + st.vy * h;
    const g = world.groundHeight(st.x, st.z, st.y + STEP);
    if (st.vy <= 0 && ny <= g + 1e-4) {
      st.y = g; st.vy = 0; st.grounded = true;
    } else if (st.vy <= 0 && wasGrounded && g > -Infinity && st.y - g <= STEP) {
      st.y = g; st.vy = 0; st.grounded = true;   // glue to stairs / ramps going down
    } else {
      if (st.vy > 0) {
        const c = world.ceilingHeight(st.x, st.z, st.y + H - 0.15);
        if (ny + H > c) { ny = c - H; st.vy = 0; }
      }
      st.y = ny; st.grounded = false;
    }
  }
  return st;
}
