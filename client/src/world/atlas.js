// Dream-atlas chart packer (sorted shelf packing with 90° rotation) + attribute flattening.
// Pure JS. Charts come from geom.js Builder.

function dims(ch, tpm, size, pad) {
  let d = ch.abs ? ch.dens * (size / 4096) : tpm * ch.dens;
  let cw = Math.ceil(ch.w * d), chh = Math.ceil(ch.h * d);
  const lim = Math.floor(size * 0.5) - 2 * pad;
  const big = Math.max(cw, chh);
  if (big > lim) { d *= lim / big; cw = Math.ceil(ch.w * d); chh = Math.ceil(ch.h * d); }
  cw = Math.max(1, cw); chh = Math.max(1, chh);
  return { d, cw, chh };
}

function tryPack(charts, tpm, size, pad) {
  const items = charts.map((ch, i) => {
    const { d, cw, chh } = dims(ch, tpm, size, pad);
    const rot = chh > cw;
    const W = (rot ? chh : cw) + 2 * pad, H = (rot ? cw : chh) + 2 * pad;
    return { i, d, rot, W, H, content: cw * chh };
  });
  items.sort((a, b) => b.H - a.H || b.W - a.W);
  let x = 0, y = 0, shelfH = 0, content = 0;
  const rects = new Array(charts.length);
  for (const it of items) {
    if (x + it.W > size) { y += shelfH; x = 0; shelfH = 0; }
    if (y + it.H > size) return null;
    rects[it.i] = { x, y, W: it.W, H: it.H, d: it.d, rot: it.rot };
    x += it.W; shelfH = Math.max(shelfH, it.H);
    content += it.content;
  }
  return { rects, fill: content / (size * size), used: (y + shelfH) / size };
}

export function packAtlas(charts, size = 4096, pad = 4, target = 0.75) {
  let A = 0;
  for (const ch of charts) if (!ch.abs) A += ch.w * ch.h * ch.dens * ch.dens;
  let lo = 0.5, hi = Math.sqrt((size * size) / Math.max(A, 1)) * 1.2, best = null, bestT = lo;
  for (let it = 0; it < 22; it++) {
    const mid = (lo + hi) / 2;
    const r = tryPack(charts, mid, size, pad);
    if (r && r.fill <= target) { best = r; bestT = mid; lo = mid; } else hi = mid;
  }
  if (!best) { best = tryPack(charts, lo, size, pad); bestT = lo; }
  if (!best) throw new Error('atlas packing failed');
  return { tpm: bestT, rects: best.rects, fill: best.fill, used: best.used, size, pad };
}

// Flatten charts to typed arrays for a single BufferGeometry.
export function flatten(charts, pack) {
  let nv = 0, ni = 0;
  for (const ch of charts) { nv += ch.P.length; ni += ch.I.length; }
  const position = new Float32Array(nv * 3), normal = new Float32Array(nv * 3), uv = new Float32Array(nv * 2);
  const atlasUv = new Float32Array(nv * 2), surface = new Float32Array(nv), zone = new Float32Array(nv);
  const index = new Uint32Array(ni);
  const chartOf = new Uint32Array(nv);
  let v = 0, k = 0;
  const S = pack.size, pad = pack.pad;
  for (let c = 0; c < charts.length; c++) {
    const ch = charts[c], r = pack.rects[c];
    const base = v;
    for (let i = 0; i < ch.P.length; i++, v++) {
      const p = ch.P[i], n = ch.N[i], u = ch.U[i], cc = ch.C[i];
      position[v * 3] = p[0]; position[v * 3 + 1] = p[1]; position[v * 3 + 2] = p[2];
      normal[v * 3] = n[0]; normal[v * 3 + 1] = n[1]; normal[v * 3 + 2] = n[2];
      uv[v * 2] = u[0]; uv[v * 2 + 1] = u[1];
      let ax, ay;
      if (!r.rot) { ax = r.x + pad + cc[0] * r.d; ay = r.y + pad + cc[1] * r.d; }
      else { ax = r.x + pad + (ch.h - cc[1]) * r.d; ay = r.y + pad + cc[0] * r.d; }
      atlasUv[v * 2] = ax / S; atlasUv[v * 2 + 1] = ay / S;
      surface[v] = ch.surface; zone[v] = ch.zone; chartOf[v] = c;
    }
    for (let i = 0; i < ch.I.length; i++) index[k++] = base + ch.I[i];
  }
  return { position, normal, uv, atlasUv, surface, zone, index, chartOf, vertexCount: nv, triCount: ni / 3 };
}
