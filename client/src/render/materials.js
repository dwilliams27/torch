// Level materials: DISPLAY (undreamt blueprint <-> painted dream composite),
// CAPTURE (the strongly-structured image the model sees, mixed with the
// already-painted dream for feedback) and PAINT (atlas-space projection of a
// diffusion result back onto the world). All share one uniforms object.
import * as THREE from 'three';
import {
  GLSL_COMMON, GLSL_PATTERNS, GLSL_LIGHTING, LEVEL_VERT,
  MAX_LIGHTS, MAX_SURF, MAX_ZONES, PATTERN_IDS,
} from './shaders.js';
import { GLSL_LIVE, createLiveUniforms } from '../dream/live.js';

const DISPLAY_FRAG = /* glsl */ `
${GLSL_COMMON}
${GLSL_PATTERNS}
${GLSL_LIGHTING}
${GLSL_LIVE}
uniform sampler2D uAtlas;
uniform float uLiveMix;
uniform float uDreamMix;
uniform float uPaintGain;
uniform float uFrontier;
uniform float uLivingLight;
uniform float uDetail;
uniform float uLocalContrast;
uniform float uRelight;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUv;
varying vec2 vAtlasUv;
flat varying int vSurf;
flat varying int vZone;
varying float vViewZ;
void main(){
  vec3 n = normalize(vNormal);
  int pat = int(uSurfPat[vSurf] + 0.5);
  vec3 V = cameraPosition - vWorld;
  float dist = length(V);
  vec3 vdir = V / dist;
  if (!gl_FrontFacing) n = -n;
  vec4 P = surfacePattern(pat, vUv, vWorld, n, uTime);
  vec3 steady;
  vec3 lit = accumulateLights(vWorld, n, steady);
  vec3 amb = uZoneAmb[vZone];
  vec3 zl = uZoneLight[vZone];
  vec4 surf = uSurf[vSurf];
  float hemi = 0.6 + 0.4 * n.y;

  // ---- undreamt: dark architectural blueprint, faint glowing contours
  float g = 0.075 * (0.75 + 0.25 * P.x);
  vec3 base = vec3(g) * mix(vec3(1.0), surf.rgb, 0.3);
  vec3 undreamt = base * (amb * hemi * 2.2 + lit * 0.9);
  float fres = pow(1.0 - saturate(abs(dot(n, vdir))), 4.0);
  float near = 0.3 + 0.7 * exp(-dist * 0.05);
  // contour lines breathe with a slow wave rolling outward from the viewer
  float wave = 0.75 + 0.25 * sin(uTime * 0.6 - dist * 0.35);
  undreamt += zl * (P.y * 0.11 * near * wave + fres * 0.03);
  undreamt += surf.rgb * (surf.a + P.z) * 0.18;
  if (pat == 8) undreamt = skyShade(-vdir, uTime) * 0.18;

  // ---- painted dream (premultiplied atlas: rgb = sum w*c, a = confidence)
  vec4 A = texture(uAtlas, vAtlasUv);
  vec4 Ab = texture(uAtlas, vAtlasUv, 2.5);
  vec3 blurred = Ab.rgb / max(Ab.a, 1e-4);
  // hole fill: thin never-visible slivers (behind silhouettes, chart borders)
  // borrow paint from their atlas neighbourhood via the premultiplied mips
  float fill = (1.0 - smoothstep(0.05, 0.45, A.a)) * smoothstep(0.12, 0.45, Ab.a);
  // a filled sliver inside painted surroundings counts as dreamt (at Ab.a-weighted
  // confidence it stayed under the wash-in threshold: a jagged zone-lit crack); a sparse
  // neighbourhood, e.g. mip blur reaching across chart padding from an unrelated chart,
  // still doesn't
  float conf = max(A.a, fill * 0.75 * smoothstep(0.2, 0.5, Ab.a));
  vec3 painted = mix(A.rgb / max(A.a, 1e-4), blurred, fill);
  painted = max(painted + (painted - blurred) * uLocalContrast, 0.0); // unsharp: crisper brushwork
  // live layer: the newest results projected at their own resolution (dream/live.js)
  vec4 L = liveComposite(vWorld, n);
  float la = L.a * uLiveMix;
  painted = mix(painted, L.rgb, la);
  conf = max(conf, la);
  painted = srgbToLinear(painted);
  // detail map: the procedural pattern keeps close atlas paint crisp (live paint has its own)
  if (pat != 8) painted *= mix(1.0, (0.55 + 0.5 * P.x) * (1.0 - 0.35 * P.y), uDetail * near * (1.0 - 0.8 * la));
  float lm = luma(lit + amb) / max(luma(steady + amb), 1e-3);
  painted *= mix(1.0, clamp(lm, 0.6, 1.5), uLivingLight) * uPaintGain;
  // relight from the real geometry: surfaces turning away from the eye darken, so a round
  // pillar shades toward its edges even where the paint made it one with the wall behind
  if (pat != 8) painted *= mix(1.0, 0.45 + 0.55 * sqrt(saturate(abs(dot(n, vdir)))), uRelight);
  // emissive surfaces glow; their procedural pattern (window spokes, glow bands) only
  // modulates atlas paint, not the model's own picture in the live views
  if (pat != 8) painted += painted * (surf.a + P.z * 0.5 * (1.0 - la)) * 0.6;

  // organic wash-in: confidence crosses a world-space noise threshold, so the
  // dream arrives like ink bleeding into wet paper rather than a uniform fade
  float nz = fbm3(vWorld * 0.55 + vec3(0.0, uTime * 0.02, 0.0));
  float c2 = conf * 1.35;
  float k = smoothstep(nz - 0.1, nz + 0.1, c2) * uDreamMix;
  float front = (1.0 - smoothstep(0.0, 0.1, abs(c2 - nz))) * smoothstep(0.01, 0.05, conf);
  // ...but only where a region is being dreamt for the first time: slivers inside painted
  // surroundings (behind silhouettes, chart borders) stay quiet instead of outlining edges
  front *= (1.0 - smoothstep(0.3, 0.7, Ab.a)) * (1.0 - la);
  // ...and not along a static visibility edge (confidence jumping within a few pixels):
  // a wash-in is smooth across the screen, an edge would sit there as a pale seam
  front *= 1.0 - smoothstep(0.02, 0.08, fwidth(conf));
  float sh = vnoise3(vWorld * 1.7 + vec3(0.0, uTime * 0.25, uTime * 0.11));
  vec3 col = mix(undreamt, painted, k);
  col += zl * front * uFrontier * (0.35 + 0.65 * sh) * uDreamMix * near * 0.5;
  // fresh paint shimmers gently until it settles
  col *= 1.0 + 0.1 * (sh - 0.5) * (1.0 - smoothstep(0.35, 0.85, conf)) * k;

  float fog = 1.0 - exp(-uFogDensity * dist);
  if (pat == 8) fog *= 0.35;
  fog *= mix(1.0, 0.35, k);               // the dream already carries its own atmosphere
  col = mix(col, uFogColor, fog);
  gl_FragColor = vec4(col, (1.0 - k) * (1.0 - fog));
}
`;

const CAPTURE_FRAG = /* glsl */ `
${GLSL_COMMON}
${GLSL_PATTERNS}
${GLSL_LIGHTING}
${GLSL_LIVE}
uniform sampler2D uAtlas;
uniform float uLiveFeedback;
uniform float uFeedback;
uniform float uFeedbackBlur;
uniform float uFeedbackAnchor;
uniform float uFeedbackSat;
uniform float uCapExposure;
uniform float uCapContrast;
uniform sampler2D uCapNear;     // this capture's inverse depth at quarter resolution, mipmapped
uniform vec2 uCapRes;           // this capture's render size (px)
uniform float uCapDepthSep;
uniform float uCapEdgeKeep;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUv;
varying vec2 vAtlasUv;
flat varying int vSurf;
flat varying int vZone;
varying float vViewZ;
void main(){
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;
  int pat = int(uSurfPat[vSurf] + 0.5);
  vec3 V = cameraPosition - vWorld;
  float dist = length(V);
  vec3 vdir = V / dist;
  vec4 P = surfacePattern(pat, vUv, vWorld, n, uTime);
  vec3 steady;
  vec3 lit = accumulateLights(vWorld, n, steady);
  vec4 surf = uSurf[vSurf];
  vec3 amb = uZoneAmb[vZone];
  float hemi = 0.55 + 0.45 * n.y;
  vec3 albedo = surf.rgb * P.x;
  albedo *= 1.0 - 0.6 * P.y;                 // dark grooves: strong readable structure
  vec3 col = albedo * (amb * hemi * 1.1 + lit);
  float fres = pow(1.0 - saturate(dot(n, vdir)), 3.0);
  col += albedo * fres * amb * 0.8;          // rim separation
  col += surf.rgb * (surf.a + P.z) * 1.1;    // emissive
  if (pat == 8) col = skyShade(-vdir, uTime);
  float fog = 1.0 - exp(-uFogDensity * 0.4 * dist);
  if (pat == 8) fog *= 0.3;
  col = mix(col, uFogColor, fog);
  // depth separation, as on the display (post.js): the surface just behind a nearer
  // silhouette darkens, the silhouette's rim lifts, so a pillar reads apart from the wall
  // behind it in what the model sees. On the raw render only: the feedback below carries
  // the model's own picture, and a cue added there would compound pass after pass.
  // Two scales, as on the display: rims (mip 1, ~8 px) and objects (mip 3, ~32 px and its
  // bilinear spread), so a whole pillar lifts against a shade on the wall around it.
  float sepX = 0.0;   // strongest of the two, soft-limited: how far from a silhouette's depth
  if (uCapDepthSep > 0.0 || uCapEdgeKeep > 0.0) {
    vec2 cuv = gl_FragCoord.xy / uCapRes;
    float xr = log(max(vViewZ * textureLod(uCapNear, cuv, 1.0).r, 1e-3));
    float xo = log(max(vViewZ * textureLod(uCapNear, cuv, 3.0).r, 1e-3));
    xr = xr / (1.0 + abs(xr) / 0.6);
    xo = xo / (1.0 + abs(xo) / 0.6);
    float far = smoothstep(80.0, 300.0, vViewZ);
    col *= mix(exp(-uCapDepthSep * ((xr > 0.0 ? xr : 0.4 * xr) + 0.8 * (xo > 0.0 ? xo : 0.6 * xo))), 1.0, far);
    sepX = max(abs(xr), abs(xo)) * (1.0 - far);
  }
  vec3 srgb = linearToSrgb(softClip(col * uCapExposure));
  srgb = saturate3((srgb - 0.45) * uCapContrast + 0.43);   // punchy input: the model mirrors contrast
  // feedback: show the model its own prior dream so it refines rather than restarts.
  // The atlas part is low-passed (mip-biased) so the model inherits the dream's palette
  // and composition but not its high-frequency artifacts; where the live views reach, the
  // model sees its previous results at full resolution (a 90 s hold stayed stable).
  // Both are luminance-anchored to the raw render so tone can't run away in the loop.
  vec4 A = texture(uAtlas, vAtlasUv, uFeedbackBlur);
  float conf = A.a;
  vec3 painted = A.rgb / max(conf, 1e-4);
  // where the previous results saw this surface, show the model those pixels (sharp,
  // reprojected) rather than the soft atlas: it refines its last dream instead of
  // re-inventing detail for every new viewpoint
  vec4 L = liveComposite(vWorld, n);
  float la = L.a * uLiveFeedback;
  painted = mix(painted, L.rgb, la);
  conf = max(conf, la);
  // leak: pull saturation back a little each trip round the loop so colour
  // can't run away into flat neon blobs, and anchor luminance to the raw render
  float lp0 = luma(painted);
  painted = mix(vec3(lp0), painted, uFeedbackSat);
  float lr = luma(srgb) + 0.02, lp = lp0 + 0.02;
  painted *= mix(1.0, clamp(lr / lp, 0.5, 2.0), uFeedbackAnchor);
  float fb = uFeedback * smoothstep(0.05, 0.5, conf);
  // at silhouettes the model sees more of the real geometry and less of its last dream, so
  // a pillar it once painted into the wall can come back out
  fb *= 1.0 - uCapEdgeKeep * smoothstep(0.08, 0.3, sepX);
  gl_FragColor = vec4(saturate3(mix(srgb, painted, fb)), 1.0);
}
`;

const PAINT_VERT = /* glsl */ `
attribute vec2 atlasUv;
attribute float surface;
varying vec3 vWorld;
varying vec3 vNormal;
flat varying int vSurf;
void main(){
  vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
  vNormal = normalize(mat3(modelMatrix) * normal);
  vSurf = int(surface + 0.5);
  gl_Position = vec4(atlasUv * 2.0 - 1.0, 0.0, 1.0);
}
`;

const PAINT_FRAG = /* glsl */ `
uniform sampler2D uImage;
uniform sampler2D uDepth;
uniform mat4 uCapViewProj;
uniform vec3 uCapPos;
uniform float uNear;
uniform float uFar;
uniform float uRate;
uniform float uFlipY;
uniform vec2 uDepthTexel;
uniform float uPixelAngle;
uniform float uDistRef;
uniform float uSide;
uniform float uWarp;            // the capture image's foveal warp (live.js liveWarp; 1 = none)
uniform float uSurfPat[${MAX_SURF}];
varying vec3 vWorld;
varying vec3 vNormal;
flat varying int vSurf;
float linDepth(float d){ return (uNear * uFar) / (uFar - (uFar - uNear) * d); }
void main(){
  vec4 clip = uCapViewProj * vec4(vWorld, 1.0);
  if (clip.w <= 1e-4) discard;
  vec2 ndc = clip.xy / clip.w;
  if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0) discard;
  vec2 suv = ndc * 0.5 + 0.5;
  vec3 toCam = uCapPos - vWorld;
  float dist = length(toCam);
  vec3 n = normalize(vNormal);
  float cosT = dot(n, toCam / dist);
  bool sky = int(uSurfPat[vSurf] + 0.5) == 8;
  if (sky) cosT = abs(cosT);
  if (cosT < 0.03) discard;
  float z = clip.w;
  // soft visibility: fraction of 5 depth taps that don't occlude this point
  // (the centre counts double). Silhouette-adjacent texels get partial paint
  // instead of a hard jagged gap.
  vec2 t = uDepthTexel * 1.25;
  float bias = 0.04 + uPixelAngle * 2.5 * z / max(cosT, 0.1);
  float s0 = linDepth(texture(uDepth, suv).r);
  float s1 = linDepth(texture(uDepth, suv + vec2( t.x,  t.y)).r);
  float s2 = linDepth(texture(uDepth, suv + vec2(-t.x,  t.y)).r);
  float s3 = linDepth(texture(uDepth, suv + vec2( t.x, -t.y)).r);
  float s4 = linDepth(texture(uDepth, suv + vec2(-t.x, -t.y)).r);
  vec4 sn = vec4(s1, s2, s3, s4);
  vec4 vn = vec4(1.0) - smoothstep(vec4(bias), vec4(bias * 2.0 + 0.05), vec4(z) - sn);
  float v0 = 1.0 - smoothstep(bias, bias * 2.0 + 0.05, z - s0);
  float vis = v0 * (2.0 + dot(vn, vec4(1.0))) / 6.0;
  vis *= vis;
  if (vis <= 0.01) discard;
  vec2 e = min(suv, 1.0 - suv);
  float edge = smoothstep(0.0, 0.12, min(e.x, e.y));
  float face = smoothstep(0.05, 0.5, cosT);
  float distW = sky ? 0.6 : 1.0 / (1.0 + (dist / uDistRef) * (dist / uDistRef));
  // side glances mostly paint the part of their image the centre view can't see
  float sideW = uSide == 0.0 ? 1.0 : mix(0.2, 1.0, smoothstep(0.3, 0.8, uSide > 0.0 ? 1.0 - suv.x : suv.x));
  float a = uRate * vis * edge * face * distW * sideW;
  if (a < 0.002) discard;
  vec2 wuv = uWarp * ndc / (1.0 + (uWarp - 1.0) * abs(ndc)) * 0.5 + 0.5;
  vec2 iuv = vec2(wuv.x, uFlipY > 0.5 ? 1.0 - wuv.y : wuv.y);
  vec3 c = texture(uImage, iuv).rgb;
  gl_FragColor = vec4(c, a);
}
`;

const QUAD_VERT = /* glsl */ `
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

export function createSharedUniforms(level) {
  const surf = [], pat = new Array(MAX_SURF).fill(0);
  for (let i = 0; i < MAX_SURF; i++) {
    const s = level.surfaceTypes[i];
    surf.push(s ? new THREE.Vector4(s.color[0], s.color[1], s.color[2], s.emissive || 0) : new THREE.Vector4(0.5, 0.5, 0.5, 0));
    pat[i] = s ? (PATTERN_IDS[s.pattern] ?? 0) : 0;
  }
  const amb = [], zl = [];
  for (let i = 0; i < MAX_ZONES; i++) {
    const z = level.zones[i];
    amb.push(z ? new THREE.Vector3(...z.ambient) : new THREE.Vector3(0.1, 0.1, 0.1));
    zl.push(z ? new THREE.Vector3(...z.light) : new THREE.Vector3(0.5, 0.6, 0.8));
  }
  return {
    uTime: { value: 0 },
    uLightPos: { value: Array.from({ length: MAX_LIGHTS }, () => new THREE.Vector4(0, -1000, 0, 1)) },
    uLightCol: { value: Array.from({ length: MAX_LIGHTS }, () => new THREE.Vector4(0, 0, 0, 1)) },
    uLightCount: { value: 0 },
    uSurf: { value: surf },
    uSurfPat: { value: pat },
    uZoneAmb: { value: amb },
    uZoneLight: { value: zl },
    uFogColor: { value: new THREE.Vector3(0.05, 0.06, 0.08) },
    uFogDensity: { value: 0.02 },
    uSkyColor: { value: new THREE.Vector3(0.3, 0.35, 0.6) },
    uSkyColor2: { value: new THREE.Vector3(0.05, 0.05, 0.1) },
    uAtlas: { value: null },
    ...createLiveUniforms(),
  };
}

export function createLevelMaterials(shared) {
  const display = new THREE.ShaderMaterial({
    name: 'dream-display',
    uniforms: {
      ...shared,
      uDreamMix: { value: 1 },
      uLiveMix: { value: 1 },
      uPaintGain: { value: 1.0 },
      uFrontier: { value: 1.0 },
      uLivingLight: { value: 0.55 },
      uDetail: { value: 0.22 },
      uLocalContrast: { value: 0.45 },
      uRelight: { value: 0 },
    },
    vertexShader: LEVEL_VERT,
    fragmentShader: DISPLAY_FRAG,
    side: THREE.DoubleSide,
  });
  const capture = new THREE.ShaderMaterial({
    name: 'dream-capture',
    // the capture reads the live views unsharpened: sharpening inside the feedback loop
    // would compound pass after pass over a long hold
    uniforms: { ...shared, uLiveSharpen: { value: 0 }, uLiveFeedback: { value: 1 }, uFeedback: { value: 0.45 }, uFeedbackBlur: { value: 0.5 }, uFeedbackAnchor: { value: 0.35 }, uFeedbackSat: { value: 0.8 }, uCapExposure: { value: 1.0 }, uCapContrast: { value: 1.15 },
      uCapNear: { value: null }, uCapRes: { value: new THREE.Vector2(1, 1) }, uCapDepthSep: { value: 0 }, uCapEdgeKeep: { value: 0 } },
    vertexShader: LEVEL_VERT,
    fragmentShader: CAPTURE_FRAG,
    side: THREE.DoubleSide,
  });
  const paint = new THREE.ShaderMaterial({
    name: 'dream-paint',
    uniforms: {
      uImage: { value: null },
      uDepth: { value: null },
      uCapViewProj: { value: new THREE.Matrix4() },
      uCapPos: { value: new THREE.Vector3() },
      uNear: { value: 0.1 },
      uFar: { value: 800 },
      uRate: { value: 0.35 },
      uFlipY: { value: 1 },
      uDepthTexel: { value: new THREE.Vector2(1 / 512, 1 / 512) },
      uPixelAngle: { value: 0.003 },
      uDistRef: { value: 12 },
      uSide: { value: 0 },
      uWarp: { value: 1 },
      uSurfPat: shared.uSurfPat,
    },
    vertexShader: PAINT_VERT,
    fragmentShader: PAINT_FRAG,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.SrcAlphaFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    blendEquationAlpha: THREE.AddEquation,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
  });
  // multiplies the whole atlas (rgb and confidence) by uFactor: the dream slowly relaxes
  const relax = new THREE.ShaderMaterial({
    name: 'dream-relax',
    uniforms: { uFactor: { value: 0.99 } },
    vertexShader: QUAD_VERT,
    fragmentShader: `uniform float uFactor; void main(){ gl_FragColor = vec4(uFactor); }`,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.ZeroFactor,
    blendDst: THREE.SrcColorFactor,
    blendEquationAlpha: THREE.AddEquation,
    blendSrcAlpha: THREE.ZeroFactor,
    blendDstAlpha: THREE.SrcAlphaFactor,
  });
  // a capture's inverse depth (1/m), for the capture's depth separation (dream.js)
  const capNear = new THREE.ShaderMaterial({
    name: 'dream-capture-depth',
    vertexShader: LEVEL_VERT,
    fragmentShader: `varying float vViewZ; void main(){ float v = 1.0 / max(vViewZ, 0.05); gl_FragColor = vec4(v, v, v, 1.0); }`,
    side: THREE.DoubleSide,
  });
  return { display, capture, paint, relax, capNear };
}

// Picks the nearest lights to the camera, flickers them, fades the cut-off ones.
export class LightDriver {
  constructor(level, shared) {
    this.shared = shared;
    this.lights = (level.lights || []).map((l, i) => ({
      p: new THREE.Vector3(...l.position),
      c: new THREE.Vector3(...l.color).multiplyScalar(l.intensity ?? 1),
      r: l.radius ?? 10,
      flicker: !!l.flicker,
      seed: (i * 7.31) % 6.283,
      score: 0,
    }));
    this.sorted = this.lights.slice();
  }
  update(camPos, t) {
    const L = this.sorted;
    for (const l of L) l.score = l.p.distanceTo(camPos) - l.r;
    L.sort((a, b) => a.score - b.score);
    const n = Math.min(MAX_LIGHTS, L.length);
    const cutoff = L.length > MAX_LIGHTS ? L[MAX_LIGHTS].score : Infinity;
    const P = this.shared.uLightPos.value, C = this.shared.uLightCol.value;
    for (let i = 0; i < n; i++) {
      const l = L[i];
      let f = 1;
      if (l.flicker) {
        const s = l.seed;
        f = 1 + 0.11 * Math.sin(t * 8.3 + s) + 0.07 * Math.sin(t * 13.1 + s * 2.1) + 0.05 * Math.sin(t * 21.7 + s * 3.3)
          + 0.06 * Math.sin(t * 1.3 + s * 5.0);
      }
      const fade = cutoff === Infinity ? 1 : Math.min(1, Math.max(0, (cutoff - l.score) / 4));
      P[i].set(l.p.x, l.p.y, l.p.z, l.r);
      C[i].set(l.c.x * f * fade, l.c.y * f * fade, l.c.z * f * fade, Math.max(f, 0.05));
    }
    this.shared.uLightCount.value = n;
  }
}

// Smoothly blends fog/sky toward the current zone.
export class ZoneAtmos {
  constructor(level, shared) {
    this.level = level; this.shared = shared;
    this.fog = new THREE.Vector3(); this.sky = new THREE.Vector3(); this.dens = 0.02;
    this.initialized = false;
  }
  update(zoneIndex, dt) {
    const z = this.level.zones[zoneIndex] || this.level.zones[0];
    const tf = new THREE.Vector3(...z.fog), ts = new THREE.Vector3(...z.sky);
    const k = this.initialized ? 1 - Math.exp(-dt / 1.6) : 1;
    this.initialized = true;
    this.fog.lerp(tf, k); this.sky.lerp(ts, k);
    this.dens += ((z.fogDensity ?? 0.02) - this.dens) * k;
    const S = this.shared;
    S.uFogColor.value.copy(this.fog);
    S.uFogDensity.value = this.dens;
    S.uSkyColor.value.copy(this.sky);
    S.uSkyColor2.value.copy(this.fog).multiplyScalar(0.8);
  }
}

// Split one big static geometry into spatial chunks sharing the same GPU
// attribute buffers (index-range sub-geometries). Gives frustum culling for
// the main view, the capture and - most importantly - the paint pass.
export function buildChunks(geometry, cell = 12) {
  const index = geometry.index;
  const pos = geometry.attributes.position;
  const triCount = index ? index.count / 3 : pos.count / 3;
  const getV = index ? (i) => index.getX(i) : (i) => i;
  const keys = new Float64Array(triCount);
  const order = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    const a = getV(t * 3), b = getV(t * 3 + 1), c = getV(t * 3 + 2);
    const cx = (pos.getX(a) + pos.getX(b) + pos.getX(c)) / 3;
    const cy = (pos.getY(a) + pos.getY(b) + pos.getY(c)) / 3;
    const cz = (pos.getZ(a) + pos.getZ(b) + pos.getZ(c)) / 3;
    // big triangles (sky domes, far backdrops) are bucketed by direction instead
    // of position, otherwise every dome triangle becomes its own draw call
    const e = Math.max(
      Math.hypot(pos.getX(a) - pos.getX(b), pos.getY(a) - pos.getY(b), pos.getZ(a) - pos.getZ(b)),
      Math.hypot(pos.getX(a) - pos.getX(c), pos.getY(a) - pos.getY(c), pos.getZ(a) - pos.getZ(c)),
      Math.hypot(pos.getX(b) - pos.getX(c), pos.getY(b) - pos.getY(c), pos.getZ(b) - pos.getZ(c)));
    if (e > cell * 1.5) {
      const oct = Math.floor((Math.atan2(cz, cx) + Math.PI) / (Math.PI / 4)) % 8;
      keys[t] = -1 - (oct + 8 * (cy > 0 ? 1 : 0));
    } else {
      const ix = Math.floor(cx / cell) + 512, iy = Math.floor(cy / (cell * 2)) + 64, iz = Math.floor(cz / cell) + 512;
      keys[t] = (ix * 1024 + iz) * 128 + iy;
    }
    order[t] = t;
  }
  order.sort((x, y) => keys[x] - keys[y]);
  const newIndex = new Uint32Array(triCount * 3);
  const chunks = [];
  let start = 0;
  for (let i = 0; i < triCount; i++) {
    const t = order[i];
    newIndex[i * 3] = getV(t * 3); newIndex[i * 3 + 1] = getV(t * 3 + 1); newIndex[i * 3 + 2] = getV(t * 3 + 2);
    const last = i === triCount - 1 || keys[order[i + 1]] !== keys[t];
    if (last) { chunks.push({ start: start * 3, count: (i + 1 - start) * 3 }); start = i + 1; }
  }
  const sharedIndex = new THREE.BufferAttribute(newIndex, 1);
  const attrs = geometry.attributes;
  const box = new THREE.Box3(), v = new THREE.Vector3();
  return chunks.map(({ start, count }) => {
    const g = new THREE.BufferGeometry();
    for (const name in attrs) g.setAttribute(name, attrs[name]);
    g.setIndex(sharedIndex);
    g.setDrawRange(start, count);
    box.makeEmpty();
    for (let i = start; i < start + count; i++) { v.fromBufferAttribute(pos, newIndex[i]); box.expandByPoint(v); }
    g.boundingBox = box.clone();
    g.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
    return g;
  });
}
