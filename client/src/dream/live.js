// Live views: the newest centre-view diffusion results, kept whole (image + capture depth +
// capture camera) and projected straight onto the world at display time, newest over
// older over the atlas. The atlas holds ~13 texels/m, so a wall 3 m away gets ~4.5x
// fewer texels than the capture that painted it had pixels (6x at 2 m); the live layer
// shows the model's own pixels there instead. The capture shader reads the same layer, so the model is shown a sharp
// reprojection of its previous result and refines it rather than re-inventing detail.
import * as THREE from 'three';

// Shader array size (2 samplers each: 13 of the 16 guaranteed units with the atlas). Fewer
// can be active; walking forward, the older views are what keep the screen edges sharp.
export const LIVE_SLOTS = 6;

// GLSL shared by the display and capture materials. Views are composited oldest first,
// so the newest wins wherever it saw the surface, unless it is coarser there than the
// paint below it (a wide capture over a narrow one, or seen from farther away): then it
// mostly lets the finer paint show through.
export const GLSL_LIVE = /* glsl */ `
uniform sampler2D uLiveImg[${LIVE_SLOTS}];
uniform sampler2D uLiveDepth[${LIVE_SLOTS}];
uniform mat4 uLiveVP[${LIVE_SLOTS}];
uniform vec3 uLivePos[${LIVE_SLOTS}];
uniform float uLiveA[${LIVE_SLOTS}];
uniform float uLiveFlip[${LIVE_SLOTS}];
uniform float uLivePixAngle[${LIVE_SLOTS}];   // capture image pixel angle (radians) without the warp, per view
uniform float uLiveNarrow[${LIVE_SLOTS}];     // 1 = a narrow (foveated) capture
uniform float uLiveWarp[${LIVE_SLOTS}];       // foveal warp gain of the capture image (1 = none)
uniform vec2 uLiveTexel;        // 1 / capture depth size (the perspective render)
uniform vec2 uLiveImgTexel;     // 1 / result image size
uniform vec2 uLiveNF;           // capture near / far
uniform float uLiveSharpen;     // contrast-adaptive sharpening of the live pixels (0 = off)
// Foveal warp (per axis, so vertical and horizontal lines stay straight): a capture's
// perspective NDC p lands on its image at q = m p / (1 + (m - 1)|p|). The middle of the
// image gets m times the pixels per degree, the border 1/m, and the border maps to itself.
vec2 liveWarp(vec2 p, float m){ return m * p / (1.0 + (m - 1.0) * abs(p)); }
// the warp's magnification at p along its weaker axis (dq/dp)
float liveWarpGain(vec2 p, float m){ vec2 d = 1.0 + (m - 1.0) * abs(p); return m / (max(d.x, d.y) * max(d.x, d.y)); }
float liveLin(float d){ return (uLiveNF.x * uLiveNF.y) / (uLiveNF.y - (uLiveNF.y - uLiveNF.x) * d); }
// Bilinear-filtered depth comparison (manual PCF): smooth silhouettes instead of
// capture-pixel stair steps.
float liveVis(sampler2D dep, vec2 suv, float z, float bias){
  vec2 tc = suv / uLiveTexel - 0.5;
  vec2 f = fract(tc);
  vec2 b = (floor(tc) + 0.5) * uLiveTexel;
  vec4 s = vec4(liveLin(texture(dep, b).r), liveLin(texture(dep, b + vec2(uLiveTexel.x, 0.0)).r),
                liveLin(texture(dep, b + vec2(0.0, uLiveTexel.y)).r), liveLin(texture(dep, b + uLiveTexel).r));
  vec4 v = vec4(1.0) - smoothstep(vec4(bias), vec4(bias * 2.0 + 0.04), vec4(z) - s);
  return mix(mix(v.x, v.y, f.x), mix(v.z, v.w, f.x), f.y);
}
// rgb = colour (as the model returned it), a = coverage in 0..1; fp = the size of one of
// the view's pixels on this surface (m)
vec4 liveView(sampler2D img, sampler2D dep, mat4 vp, vec3 cpos, float alpha, float flip, float pixA, float narrow, float warp, vec3 wp, vec3 n, out float fp){
  fp = 1e6;
  if (alpha <= 0.0) return vec4(0.0);
  vec4 clip = vp * vec4(wp, 1.0);
  if (clip.w <= 1e-4) return vec4(0.0);
  vec2 ndc = clip.xy / clip.w;
  if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0) return vec4(0.0);
  vec2 suv = ndc * 0.5 + 0.5;
  vec3 toCam = cpos - wp;
  float dist = length(toCam);
  float cosT = abs(dot(n, toCam / dist));
  // depth is the perspective render, warp times the image's density; fp is the size of an
  // image pixel here, which the warp shrinks toward the middle
  float bias = 0.04 + pixA / warp * 2.5 * clip.w / max(cosT, 0.1);
  fp = pixA * clip.w / max(cosT, 0.2) / liveWarpGain(ndc, warp);
  float vis = liveVis(dep, suv, clip.w, bias);
  vec2 e = min(suv, 1.0 - suv);
  float edge = smoothstep(0.0, 0.06, min(e.x, e.y));
  if (narrow > 0.5) {
    // a narrow view fades out over a wide band with rounded corners (a superellipse,
    // |x|^4 + |y|^4), so detail falls off toward the screen edges instead of stopping at a
    // visible border
    vec2 q = abs(suv * 2.0 - 1.0); q *= q; q *= q;
    edge = 1.0 - smoothstep(0.7, 0.98, sqrt(sqrt(q.x + q.y)));
  }
  // grazing surfaces are kept: the display sees them from about where the capture did,
  // so there is no smear, and dropping them leaves unpainted rims on every silhouette
  float face = smoothstep(0.0, 0.06, cosT);
  vec2 wuv = liveWarp(ndc, warp) * 0.5 + 0.5;
  vec2 iuv = vec2(wuv.x, flip > 0.5 ? 1.0 - wuv.y : wuv.y);
  float a = alpha * vis * edge * face;
  if (a <= 0.0) return vec4(0.0);
  vec3 c = texture(img, iuv).rgb;
  if (uLiveSharpen > 0.0) {
    // contrast-adaptive sharpening (after AMD FidelityFX CAS): a wide capture pixel spans
    // ~4 screen pixels across at 1080p, so it arrives soft; sharpen, less where contrast is high.
    // One weight from green for all channels, as CAS does, so chroma noise isn't boosted.
    vec3 tN = texture(img, iuv + vec2(0.0, uLiveImgTexel.y)).rgb, tS = texture(img, iuv - vec2(0.0, uLiveImgTexel.y)).rgb;
    vec3 tE = texture(img, iuv + vec2(uLiveImgTexel.x, 0.0)).rgb, tW = texture(img, iuv - vec2(uLiveImgTexel.x, 0.0)).rgb;
    float mn = min(min(min(tN.g, tS.g), min(tE.g, tW.g)), c.g), mx = max(max(max(tN.g, tS.g), max(tE.g, tW.g)), c.g);
    float amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, 1e-4), 0.0, 1.0));
    float wt = -amp / mix(8.0, 5.0, uLiveSharpen);
    c = clamp((c + wt * (tN + tS + tE + tW)) / (1.0 + 4.0 * wt), 0.0, 1.0);
  }
  return vec4(c, a);
}
vec4 liveComposite(vec3 wp, vec3 n){
  // premultiplied "over", newest last: rgb = sum of colour x weight, a = weight. fpAcc is
  // the pixel size accumulated the same way (fpAcc / acc.a = the mean size of what is
  // already there; no large sentinel: fp32 mix(1e6, fp, 1) can round to 0 on Metal).
  // Gating only reweights the live views among themselves; cov, their ungated coverage,
  // is how much of the atlas they hide.
  vec4 acc = vec4(0.0);
  vec4 v;
  float fp, a, fpAcc = 0.0, cov = 0.0;
${Array.from({ length: LIVE_SLOTS }, (_, i) => `  v = liveView(uLiveImg[${i}], uLiveDepth[${i}], uLiveVP[${i}], uLivePos[${i}], uLiveA[${i}], uLiveFlip[${i}], uLivePixAngle[${i}], uLiveNarrow[${i}], uLiveWarp[${i}], wp, n, fp);
  a = v.a * (1.0 - 0.9 * acc.a * smoothstep(1.15, 1.5, fp * acc.a / max(fpAcc, 1e-7)));
  acc = vec4(v.rgb * a, a) + acc * (1.0 - a);
  fpAcc = fp * a + fpAcc * (1.0 - a);
  cov = v.a + cov * (1.0 - v.a);`).join('\n')}
  return vec4(acc.rgb / max(acc.a, 1e-4), cov);   // straight colour + coverage
}
`;

let dummy = null;   // 1x1 white stand-in for unused sampler slots
function dummyTexture() {
  if (!dummy) {
    dummy = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    dummy.needsUpdate = true;
  }
  return dummy;
}

export function createLiveUniforms() {
  const img = dummyTexture(), dep = img;
  return {
    uLiveImg: { value: Array.from({ length: LIVE_SLOTS }, () => img) },
    uLiveDepth: { value: Array.from({ length: LIVE_SLOTS }, () => dep) },
    uLiveVP: { value: Array.from({ length: LIVE_SLOTS }, () => new THREE.Matrix4()) },
    uLivePos: { value: Array.from({ length: LIVE_SLOTS }, () => new THREE.Vector3()) },
    uLiveA: { value: new Array(LIVE_SLOTS).fill(0) },
    uLiveFlip: { value: new Array(LIVE_SLOTS).fill(0) },
    uLivePixAngle: { value: new Array(LIVE_SLOTS).fill(0.004) },
    uLiveNarrow: { value: new Array(LIVE_SLOTS).fill(0) },
    uLiveWarp: { value: new Array(LIVE_SLOTS).fill(1) },
    uLiveTexel: { value: new THREE.Vector2(1 / 384, 1 / 384) },
    uLiveImgTexel: { value: new THREE.Vector2(1 / 384, 1 / 384) },
    uLiveNF: { value: new THREE.Vector2(0.1, 800) },
    uLiveSharpen: { value: 0.6 },
  };
}

// A copy of a result into a view's own target (flip applied), or a blend toward it.
const COPY_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform float uFlip;
uniform float uAlpha;
varying vec2 vUv;
void main(){ gl_FragColor = vec4(texture(uSrc, vec2(vUv.x, uFlip > 0.5 ? 1.0 - vUv.y : vUv.y)).rgb, uAlpha); }
`;

export class LiveViews {
  // shared: the uniforms object from createSharedUniforms (holds the uLive* entries)
  // count: views kept (at most LIVE_SLOTS - 1: one shader slot stays free, so a dropped
  //   view fades out over fadeOut instead of vanishing in one frame).
  // tau: a held framing (the same pose and width: narrow or wide) keeps one view, and its
  //   picture follows the newest result of that framing with this time constant (s), frame
  //   by frame. Standing still, each result is a slightly different refinement of the same
  //   picture: replacing outright makes detail boil at the dream rate, and a stack of the last
  //   few results jumps whenever its oldest is dropped. A running average does neither, and
  //   is as calm at 12 dreams a second as at 7. 0 = each result is its own view.
  // maxAge: a view with no new result for this long (s) is dropped.
  constructor(shared, renderer, { count = LIVE_SLOTS - 1, fadeIn = 0.25, fadeOut = 0.25, maxAge = 2.5, tau = 0.35 } = {}) {
    this.U = shared;
    this.renderer = renderer;
    // Half-float targets where the GPU can render them: in 8 bits a blend step under half a
    // level rounds to nothing, and the picture would stall short of the results it follows.
    this.rtType = renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float')
      ? THREE.HalfFloatType : THREE.UnsignedByteType;
    this.count = Math.max(0, Math.min(LIVE_SLOTS - 1, count | 0));
    // (a zero fade would divide 0 by 0 on a view's first frame)
    this.fadeIn = Math.max(1e-3, fadeIn);
    this.fadeOut = Math.max(1e-3, fadeOut);
    this.maxAge = maxAge;
    this.tau = tau;
    this.views = [];     // oldest first: { slot, t, last, texture, flipY, rt, target, chase, release }
    this.leaving = [];   // dropped views fading out, oldest first (+ gone: time dropped)
    this.enabled = this.count > 0;
    this._last = 0;
    this._pool = [];     // spare render targets
    this.copyMat = new THREE.ShaderMaterial({
      uniforms: { uSrc: { value: null }, uFlip: { value: 0 }, uAlpha: { value: 1 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: COPY_FRAG, depthTest: false, depthWrite: false, transparent: true,
      blending: THREE.CustomBlending, blendSrc: THREE.SrcAlphaFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    this.quad = new THREE.Mesh(geo, this.copyMat);
    this.quad.frustumCulled = false;
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  _rt(w, h) {
    let rt = this._pool.pop();
    if (rt && (rt.width !== w || rt.height !== h)) { rt.dispose(); rt = null; }
    return rt || new THREE.WebGLRenderTarget(w, h, { depthBuffer: false, type: this.rtType });
  }

  // draw `texture` into rt at alpha (1 = copy)
  _blit(texture, flipY, rt, alpha) {
    const r = this.renderer, prev = r.getRenderTarget();
    const U = this.copyMat.uniforms;
    U.uSrc.value = texture; U.uFlip.value = flipY ? 1 : 0; U.uAlpha.value = alpha;
    r.setRenderTarget(rt);
    r.render(this.quad, this.cam);
    r.setRenderTarget(prev);
  }

  // Takes a finished result. `release()` is called once the live layer is done with it:
  // at once for a result that joins its framing's view, when the view is dropped otherwise
  // (the view keeps that first result for its picture until a second arrives, and its slot
  // for its depth and camera). Walking, every result is a new framing, so nothing is copied.
  add(slot, texture, flipY, now, release) {
    if (!this.enabled) { release(); return; }
    if (this.tau > 0) {
      let v = null;
      for (let j = this.views.length - 1; j >= 0 && !v; j--) {
        const u = this.views[j];
        if (u.slot.kfId === slot.kfId && !!u.slot.narrow === !!slot.narrow) v = u;
      }
      if (v) {
        if (!v.rt) {   // a second result of a held framing: the average starts from the first
          v.rt = this._rt(slot.imgW, slot.imgH);
          v.target = this._rt(slot.imgW, slot.imgH);
          this._blit(v.texture, v.flipY, v.rt, 1);
        }
        this._blit(texture, flipY, v.target, 1);
        v.last = now; v.chase = true;
        release();
        return;
      }
    }
    this.views.push({ slot, t: now, last: now, texture, flipY, rt: null, target: null, chase: false, release });
    while (this.views.length > this.count) this._drop(this.views.shift(), now);
  }

  _free(v) {
    v.release();
    if (v.rt) this._pool.push(v.rt, v.target);
    while (this._pool.length > 2 * LIVE_SLOTS) this._pool.shift().dispose();
  }

  _drop(v, now) {
    v.gone = now;
    this.leaving.push(v);
    while (this.leaving.length > LIVE_SLOTS - this.count) this._free(this.leaving.shift());
  }

  clear() {
    for (const v of this.leaving) this._free(v);
    for (const v of this.views) this._free(v);
    this.views = [];
    this.leaving = [];
    this.update(performance.now());
  }

  update(now) {
    const U = this.U;
    while (this.views.length && (now - this.views[0].last) / 1000 > this.maxAge) this._drop(this.views.shift(), now);
    this.leaving = this.leaving.filter((v) => {
      if ((now - v.gone) / 1000 < this.fadeOut) return true;
      this._free(v);
      return false;
    });
    // each view's picture chases its newest result (the blend is frame-rate independent);
    // four time constants after the last result it has arrived, and stops costing a pass
    const dt = Math.min(0.1, Math.max(0, (now - this._last) / 1000));
    this._last = now;
    if (this.tau > 0 && dt > 0) {
      const a = 1 - Math.exp(-dt / this.tau);
      for (const v of this.views) {
        if (!v.chase) continue;
        this._blit(v.target.texture, false, v.rt, a);
        if ((now - v.last) / 1000 > 4 * this.tau) v.chase = false;
      }
    }
    const list = this.leaving.concat(this.views);   // composited oldest first
    const n = list.length;
    const smooth = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
    for (let i = 0; i < LIVE_SLOTS; i++) {
      // pack the newest views into the highest indices (composited last)
      const v = list[n - LIVE_SLOTS + i];
      if (!v) {
        U.uLiveA.value[i] = 0;
        U.uLiveImg.value[i] = dummyTexture();
        U.uLiveDepth.value[i] = dummyTexture();
        continue;
      }
      const fade = v.gone != null ? 1 - smooth((now - v.gone) / 1000 / this.fadeOut) : smooth((now - v.t) / 1000 / this.fadeIn);
      U.uLiveA.value[i] = fade;
      U.uLiveImg.value[i] = v.rt ? v.rt.texture : v.texture;
      U.uLiveDepth.value[i] = v.slot.rt.depthTexture;
      U.uLiveVP.value[i].copy(v.slot.viewProj);
      U.uLivePos.value[i].copy(v.slot.pos);
      U.uLiveFlip.value[i] = v.rt ? 0 : (v.flipY ? 1 : 0);   // the view's own targets hold the picture upright
      U.uLiveTexel.value.set(1 / v.slot.rt.width, 1 / v.slot.rt.height);
      U.uLiveImgTexel.value.set(1 / v.slot.imgW, 1 / v.slot.imgH);
      U.uLivePixAngle.value[i] = 2 * Math.tan(THREE.MathUtils.degToRad(v.slot.fov) / 2) / v.slot.imgH;
      U.uLiveNarrow.value[i] = v.slot.narrow ? 1 : 0;
      U.uLiveWarp.value[i] = v.slot.warp || 1;
      U.uLiveNF.value.set(v.slot.near, v.slot.far);
    }
  }
}
