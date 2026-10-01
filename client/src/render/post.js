// Post chain: HDR scene target -> dual-filter bloom -> composite
// (depth-edge contour glow for undreamt areas, depth separation for dreamt ones,
// chromatic breathing, vignette, grain, soft shoulder tonemap). Tasteful by default.
import * as THREE from 'three';
import { GLSL_COMMON } from './shaders.js';

const VERT = /* glsl */ `
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const DOWN_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uPrefilter;
varying vec2 vUv;
vec3 pre(vec3 c){
  if (uPrefilter < 0.5) return c;
  float br = max(c.r, max(c.g, c.b));
  float knee = uThreshold * 0.6;
  float soft = clamp(br - uThreshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 1e-4);
  float w = max(soft, br - uThreshold) / max(br, 1e-4);
  return c * w;
}
void main(){
  vec2 t = uTexel;
  vec3 a = texture(uSrc, vUv + t * vec2(-2, -2)).rgb;
  vec3 b = texture(uSrc, vUv + t * vec2( 0, -2)).rgb;
  vec3 c = texture(uSrc, vUv + t * vec2( 2, -2)).rgb;
  vec3 d = texture(uSrc, vUv + t * vec2(-2,  0)).rgb;
  vec3 e = texture(uSrc, vUv).rgb;
  vec3 f = texture(uSrc, vUv + t * vec2( 2,  0)).rgb;
  vec3 g = texture(uSrc, vUv + t * vec2(-2,  2)).rgb;
  vec3 h = texture(uSrc, vUv + t * vec2( 0,  2)).rgb;
  vec3 i = texture(uSrc, vUv + t * vec2( 2,  2)).rgb;
  vec3 j = texture(uSrc, vUv + t * vec2(-1, -1)).rgb;
  vec3 k = texture(uSrc, vUv + t * vec2( 1, -1)).rgb;
  vec3 l = texture(uSrc, vUv + t * vec2(-1,  1)).rgb;
  vec3 m = texture(uSrc, vUv + t * vec2( 1,  1)).rgb;
  vec3 o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  gl_FragColor = vec4(pre(o), 1.0);
}
`;

// Screen-space fill chain: premultiplied "dreamt colour" (weight = how dreamt a pixel is,
// from the scene alpha) downsampled with the same 13-tap filter, so undreamt slivers can
// borrow the dream around them (silhouette disocclusions, rims, the edge of a turn).
const FILL_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uFirst;
varying vec2 vUv;
// first level: a 2x2 box of exact texels (half-texel offsets land on texel centres), each
// weighted by how dreamt it is before anything is averaged; the scene alpha mixes
// "undreamt" with fog, so only pixels that are (nearly) fully dreamt count
vec4 texel(vec2 o){
  vec4 c = texture(uSrc, vUv + uTexel * o);
  float w = 1.0 - smoothstep(0.05, 0.25, c.a);
  return vec4(c.rgb * w, w);
}
vec4 tap(vec2 o){ return texture(uSrc, vUv + uTexel * o); }
void main(){
  if (uFirst > 0.5) {
    gl_FragColor = (texel(vec2(-0.5, -0.5)) + texel(vec2(0.5, -0.5)) + texel(vec2(-0.5, 0.5)) + texel(vec2(0.5, 0.5))) * 0.25;
    return;
  }
  vec4 o = tap(vec2(0.0)) * 0.125
    + (tap(vec2(-2, -2)) + tap(vec2(2, -2)) + tap(vec2(-2, 2)) + tap(vec2(2, 2))) * 0.03125
    + (tap(vec2(0, -2)) + tap(vec2(-2, 0)) + tap(vec2(2, 0)) + tap(vec2(0, 2))) * 0.0625
    + (tap(vec2(-1, -1)) + tap(vec2(1, -1)) + tap(vec2(-1, 1)) + tap(vec2(1, 1))) * 0.125;
  gl_FragColor = o;
}
`;

// Depth separation chain, first level: scene depth -> inverse depth (1/m), a 2x2 average
// of exact texels. Inverse depth is linear across the screen on any plane, so blurring it
// leaves walls and floors exactly alone and only silhouettes differ. Later levels reuse
// DOWN_FRAG, so the last ones hold a pixel's neighbourhood (tens of pixels at full size).
const INVDEPTH_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uNear;
uniform float uFar;
varying vec2 vUv;
float iz(vec2 o){
  float d = texture(uSrc, vUv + uTexel * o).r;
  return (uFar - (uFar - uNear) * d) / (uNear * uFar);
}
void main(){
  float v = 0.25 * (iz(vec2(-0.5, -0.5)) + iz(vec2(0.5, -0.5)) + iz(vec2(-0.5, 0.5)) + iz(vec2(0.5, 0.5)));
  gl_FragColor = vec4(v, v, v, 1.0);
}
`;

const UP_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uRadius;
varying vec2 vUv;
void main(){
  vec2 t = uTexel * uRadius;
  vec3 s = texture(uSrc, vUv).rgb * 4.0;
  s += (texture(uSrc, vUv + vec2(-t.x, 0)).rgb + texture(uSrc, vUv + vec2(t.x, 0)).rgb
      + texture(uSrc, vUv + vec2(0, -t.y)).rgb + texture(uSrc, vUv + vec2(0, t.y)).rgb) * 2.0;
  s += texture(uSrc, vUv + vec2(-t.x, -t.y)).rgb + texture(uSrc, vUv + vec2(t.x, -t.y)).rgb
     + texture(uSrc, vUv + vec2(-t.x, t.y)).rgb + texture(uSrc, vUv + vec2(t.x, t.y)).rgb;
  gl_FragColor = vec4(s / 16.0, 1.0);
}
`;

const COMPOSITE_FRAG = /* glsl */ `
${GLSL_COMMON}
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform sampler2D uFill;
uniform float uFillAmount;
uniform sampler2D uDepth;
uniform vec2 uTexel;
uniform float uTime;
uniform float uNear;
uniform float uFar;
uniform float uBloomStrength;
uniform float uExposure;
uniform float uVignette;
uniform float uGrain;
uniform float uCA;
uniform float uLines;
uniform vec3 uLineColor;
uniform float uFade;
uniform float uLid;             // 0 = eyes open .. 1 = closed (main.js)
uniform float uContrast;
uniform float uSaturation;
uniform sampler2D uDepthNearA;   // mean inverse depth of the neighbourhood: rims (1/8 size)
uniform sampler2D uDepthNearB;   // ...and objects (1/64 size, a pillar's width and more)
uniform float uDepthSep;
varying vec2 vUv;
float lin(float d){ return (uNear * uFar) / (uFar - (uFar - uNear) * d); }
void main(){
  vec2 uv = vUv;
  vec2 c = uv - 0.5;
  float r2 = dot(c, c);
  // chromatic breathing: very slow, strongest at the rim
  float ca = uCA * (0.55 + 0.45 * sin(uTime * 0.31)) * r2;
  vec4 center = texture(uScene, uv);
  vec3 col;
  col.r = texture(uScene, uv - c * ca).r;
  col.g = center.g;
  col.b = texture(uScene, uv + c * ca).b;
  float undreamt = clamp(center.a, 0.0, 1.0);
  // undreamt pixels with dream around them take the surrounding dream's colour: thin
  // blueprint slivers at silhouettes vanish, wide undreamt regions keep the blueprint
  vec4 F = texture(uFill, uv);
  float fillAmt = uFillAmount * smoothstep(0.3, 0.75, undreamt) * smoothstep(0.1, 0.35, F.a);
  col = mix(col, F.rgb / max(F.a, 1e-4), fillAmt);
  undreamt *= 1.0 - fillAmt;
  // depth-edge contour lines (only where the world is not yet dreamt)
  if (uLines > 0.0 && undreamt > 0.35) {
    // ...and not next to (nearly) fully dreamt pixels: a silhouette whose rim alone is
    // unpainted (grazing angles, disocclusion slivers) would otherwise get a pale outline.
    // Fogged undreamt neighbours (alpha ~0.4-0.65) keep their lines.
    vec2 o = uTexel * 2.5;
    float nmin = min(min(texture(uScene, uv + vec2(o.x, 0)).a, texture(uScene, uv - vec2(o.x, 0)).a),
                     min(texture(uScene, uv + vec2(0, o.y)).a, texture(uScene, uv - vec2(0, o.y)).a));
    undreamt *= smoothstep(0.08, 0.25, nmin);
  }
  if (uLines > 0.0 && undreamt > 0.35) {
    float d = lin(texture(uDepth, uv).r);
    float dl = lin(texture(uDepth, uv - vec2(uTexel.x, 0)).r);
    float dr = lin(texture(uDepth, uv + vec2(uTexel.x, 0)).r);
    float dd = lin(texture(uDepth, uv - vec2(0, uTexel.y)).r);
    float du = lin(texture(uDepth, uv + vec2(0, uTexel.y)).r);
    float lap = abs(dl + dr - 2.0 * d) + abs(du + dd - 2.0 * d);
    float edge = smoothstep(0.004, 0.03, lap / d);
    float sil = smoothstep(0.03, 0.2, max(abs(dl - dr), abs(du - dd)) / d);
    float e = max(edge * 0.8, sil);
    // only genuinely undreamt surfaces get contours (not chart-border confidence dips)
    col += uLineColor * e * smoothstep(0.35, 0.85, undreamt) * uLines * (0.3 + 0.7 * exp(-d * 0.04));
  }
  // Depth separation (unsharp masking the depth buffer, after Luft, Colditz and Deussen
  // 2006): where the neighbourhood is nearer than this pixel (a wall beside a pillar) it
  // darkens, where it is farther (the pillar's rim) it lifts a little. The paint may blend
  // a pillar into the wall behind it; the real geometry still separates them, standing
  // still as well as moving. Dreamt pixels only; it fades out from 80 m and is gone past 300 m
  // (sky, far haze).
  // Two scales: a rim halo, and one wider than a pillar, so the whole pillar lifts against
  // a broad shade on the wall around it.
  if (uDepthSep > 0.0) {
    float z = lin(texture(uDepth, uv).r);
    float xr = log(max(z * texture(uDepthNearA, uv).r, 1e-3));   // > 0: the neighbourhood is nearer
    float xo = log(max(z * texture(uDepthNearB, uv).r, 1e-3));
    xr = xr / (1.0 + abs(xr) / 0.6);           // soft limits at about +-0.6
    xo = xo / (1.0 + abs(xo) / 0.6);
    float sep = exp(-uDepthSep * ((xr > 0.0 ? xr : 0.4 * xr) + 0.8 * (xo > 0.0 ? xo : 0.6 * xo)));
    sep = mix(sep, 1.0, smoothstep(80.0, 300.0, z));
    col *= mix(1.0, sep, 1.0 - undreamt);
  }
  col += texture(uBloom, uv).rgb * uBloomStrength;
  // closed eyes: light through the lids, a warm dark glow brightest in the middle and
  // breathing slowly; the scene fades into it
  if (uLid > 0.0) {
    vec3 lid = vec3(0.10, 0.028, 0.022) * (0.7 + 0.6 * (1.0 - smoothstep(0.0, 0.5, r2))) * (0.92 + 0.08 * sin(uTime * 0.8));
    col = mix(col, lid, uLid);
  }
  col *= uExposure;
  float vig = 1.0 - uVignette * smoothstep(0.1, 0.75, r2 * 1.6);
  col *= vig;
  col = softClip(col);
  col = linearToSrgb(col);
  // gentle filmic grade: a touch of contrast and colour
  float lg = luma(col);
  col = mix(vec3(lg), col, uSaturation);
  col = clamp((col - 0.5) * uContrast + 0.5, 0.0, 1.0);
  // grain + dither (luma-weighted: less in highlights)
  float n = hash12(gl_FragCoord.xy + fract(uTime * 7.13) * 431.0) - 0.5;
  col += n * uGrain * (1.0 - 0.6 * luma(col));
  col += (hash12(gl_FragCoord.xy * 1.37 + 11.0) - 0.5) / 255.0;
  col *= uFade;
  gl_FragColor = vec4(col, 1.0);
}
`;

export class Post {
  constructor(renderer, { halfFloat = true } = {}) {
    this.renderer = renderer;
    const type = halfFloat ? THREE.HalfFloatType : THREE.UnsignedByteType;
    this.type = type;
    const depthTexture = new THREE.DepthTexture(4, 4);
    depthTexture.type = THREE.UnsignedIntType;
    this.sceneRT = new THREE.WebGLRenderTarget(4, 4, {
      type, depthTexture, depthBuffer: true, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    });
    this.levels = 5;
    this.bloomRTs = [];
    for (let i = 0; i < this.levels; i++) {
      this.bloomRTs.push(new THREE.WebGLRenderTarget(4, 4, {
        type, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      }));
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    this.quad = new THREE.Mesh(geo);
    this.quad.frustumCulled = false;
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.downMat = new THREE.ShaderMaterial({
      uniforms: { uSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uThreshold: { value: 0.8 }, uPrefilter: { value: 0 } },
      vertexShader: VERT, fragmentShader: DOWN_FRAG, depthTest: false, depthWrite: false,
    });
    this.fillLevels = 3;   // 1/2, 1/4, 1/8: the last one reaches ~8-12 px at full resolution
    this.fillRTs = [];
    for (let i = 0; i < this.fillLevels; i++) {
      this.fillRTs.push(new THREE.WebGLRenderTarget(4, 4, {
        type, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      }));
    }
    // depth separation: inverse depth at 1/2, then 1/4 ... 1/64 (needs float targets:
    // without them the effect stays off)
    this.depthSepOK = type === THREE.HalfFloatType;
    this.depthRTs = [];
    for (let i = 0; i < 6; i++) {
      this.depthRTs.push(new THREE.WebGLRenderTarget(4, 4, {
        type, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      }));
    }
    this.invDepthMat = new THREE.ShaderMaterial({
      uniforms: { uSrc: { value: depthTexture }, uTexel: { value: new THREE.Vector2() }, uNear: { value: 0.05 }, uFar: { value: 1000 } },
      vertexShader: VERT, fragmentShader: INVDEPTH_FRAG, depthTest: false, depthWrite: false,
    });
    this.fillMat = new THREE.ShaderMaterial({
      uniforms: { uSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uFirst: { value: 1 } },
      vertexShader: VERT, fragmentShader: FILL_FRAG, depthTest: false, depthWrite: false,
    });
    this.upMat = new THREE.ShaderMaterial({
      uniforms: { uSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uRadius: { value: 1.0 } },
      vertexShader: VERT, fragmentShader: UP_FRAG, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor, transparent: true,
    });
    this.compMat = new THREE.ShaderMaterial({
      uniforms: {
        uScene: { value: this.sceneRT.texture }, uBloom: { value: this.bloomRTs[0].texture },
        uFill: { value: this.fillRTs[this.fillLevels - 1].texture }, uFillAmount: { value: 1 },
        uDepth: { value: depthTexture }, uTexel: { value: new THREE.Vector2() }, uTime: { value: 0 },
        uNear: { value: 0.05 }, uFar: { value: 1000 }, uBloomStrength: { value: 0.16 }, uExposure: { value: 1.0 },
        uVignette: { value: 0.42 }, uGrain: { value: 0.035 }, uCA: { value: 0.012 }, uLines: { value: 0.55 },
        uLineColor: { value: new THREE.Vector3(0.5, 0.7, 1.0) }, uFade: { value: 1 }, uLid: { value: 0 },
        uContrast: { value: 1.08 }, uSaturation: { value: 1.08 },
        uDepthNearA: { value: this.depthRTs[2].texture }, uDepthNearB: { value: this.depthRTs[5].texture }, uDepthSep: { value: 0 },
      },
      vertexShader: VERT, fragmentShader: COMPOSITE_FRAG, depthTest: false, depthWrite: false,
    });
    this.width = 0; this.height = 0;
  }

  setSize(w, h) {
    w = Math.max(4, Math.floor(w)); h = Math.max(4, Math.floor(h));
    if (w === this.width && h === this.height) return;
    this.width = w; this.height = h;
    this.sceneRT.setSize(w, h);
    let bw = w, bh = h;
    for (const rt of this.bloomRTs) {
      bw = Math.max(1, Math.floor(bw / 2)); bh = Math.max(1, Math.floor(bh / 2));
      rt.setSize(bw, bh);
    }
    bw = w; bh = h;
    for (const rt of this.fillRTs) {
      bw = Math.max(1, Math.floor(bw / 2)); bh = Math.max(1, Math.floor(bh / 2));
      rt.setSize(bw, bh);
    }
    bw = w; bh = h;
    for (const rt of this.depthRTs) {
      bw = Math.max(1, Math.floor(bw / 2)); bh = Math.max(1, Math.floor(bh / 2));
      rt.setSize(bw, bh);
    }
    this.compMat.uniforms.uTexel.value.set(1 / w, 1 / h);
  }

  _pass(mat, target) {
    this.quad.material = mat;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.quad, this.cam);
  }

  render(camera, time, opts = {}) {
    const r = this.renderer;
    const U = this.compMat.uniforms;
    // bloom down chain
    let src = this.sceneRT.texture, sw = this.width, sh = this.height;
    for (let i = 0; i < this.levels; i++) {
      const rt = this.bloomRTs[i];
      this.downMat.uniforms.uSrc.value = src;
      this.downMat.uniforms.uTexel.value.set(1 / sw, 1 / sh);
      this.downMat.uniforms.uPrefilter.value = i === 0 ? 1 : 0;
      this.downMat.uniforms.uThreshold.value = opts.bloomThreshold ?? 0.9;
      this._pass(this.downMat, rt);
      src = rt.texture; sw = rt.width; sh = rt.height;
    }
    // up chain (additive into the larger level)
    for (let i = this.levels - 2; i >= 0; i--) {
      const s = this.bloomRTs[i + 1];
      this.upMat.uniforms.uSrc.value = s.texture;
      this.upMat.uniforms.uTexel.value.set(1 / s.width, 1 / s.height);
      this._pass(this.upMat, this.bloomRTs[i]);
    }
    // fill chain
    src = this.sceneRT.texture; sw = this.width; sh = this.height;
    for (let i = 0; i < this.fillLevels; i++) {
      const rt = this.fillRTs[i];
      this.fillMat.uniforms.uSrc.value = src;
      this.fillMat.uniforms.uTexel.value.set(1 / sw, 1 / sh);
      this.fillMat.uniforms.uFirst.value = i === 0 ? 1 : 0;
      this._pass(this.fillMat, rt);
      src = rt.texture; sw = rt.width; sh = rt.height;
    }
    // depth separation chain
    U.uDepthSep.value = this.depthSepOK ? (opts.depthSep ?? 0) : 0;
    if (U.uDepthSep.value > 0) {
      const L = this.invDepthMat.uniforms;
      L.uTexel.value.set(1 / this.width, 1 / this.height);
      L.uNear.value = camera.near; L.uFar.value = camera.far;
      this._pass(this.invDepthMat, this.depthRTs[0]);
      this.downMat.uniforms.uPrefilter.value = 0;
      for (let i = 1; i < this.depthRTs.length; i++) {
        const s = this.depthRTs[i - 1];
        this.downMat.uniforms.uSrc.value = s.texture;
        this.downMat.uniforms.uTexel.value.set(1 / s.width, 1 / s.height);
        this._pass(this.downMat, this.depthRTs[i]);
      }
    }
    U.uTime.value = time;
    U.uNear.value = camera.near; U.uFar.value = camera.far;
    if (opts.lineColor) U.uLineColor.value.copy(opts.lineColor);
    U.uLid.value = opts.lid ?? 0;
    this._pass(this.compMat, null);
  }
}
