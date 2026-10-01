// GLSL building blocks shared by the level materials (display / capture / paint)
// and the post chain. Everything here is plain GLSL ES 3.0 via three's
// ShaderMaterial prefixes (attribute/varying/gl_FragColor are remapped).

export const MAX_LIGHTS = 8;
export const MAX_SURF = 32;
export const MAX_ZONES = 12;

export const PATTERN_IDS = {
  stone: 0, tile: 1, brick: 2, metal: 3, wood: 4, crystal: 5,
  plaster: 6, water: 7, sky: 8, glow: 9, formwork: 10,
};

export const GLSL_COMMON = /* glsl */ `
#define PI 3.14159265
float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3 saturate3(vec3 x){ return clamp(x, 0.0, 1.0); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float hash13(vec3 p3){ p3 = fract(p3 * .1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
vec3 hash33(vec3 p3){ p3 = fract(p3 * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yxz+33.33); return fract((p3.xxy + p3.yxx)*p3.zyx); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
float vnoise3(vec3 p){ vec3 i = floor(p), f = fract(p); vec3 u = f*f*(3.0-2.0*f);
  float a = mix(mix(hash13(i), hash13(i+vec3(1,0,0)), u.x), mix(hash13(i+vec3(0,1,0)), hash13(i+vec3(1,1,0)), u.x), u.y);
  float b = mix(mix(hash13(i+vec3(0,0,1)), hash13(i+vec3(1,0,1)), u.x), mix(hash13(i+vec3(0,1,1)), hash13(i+vec3(1,1,1)), u.x), u.y);
  return mix(a, b, u.z); }
float fbm2(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a*vnoise(p); p = p*2.03 + 17.1; a *= 0.5; } return s; }
float fbm3(vec3 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 3; i++){ s += a*vnoise3(p); p = p*2.07 + 11.3; a *= 0.5; } return s; }
vec3 srgbToLinear(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }
vec3 linearToSrgb(vec3 c){ return pow(max(c, 0.0), vec3(1.0/2.2)); }
// gentle shoulder: identity below 0.75, smooth roll-off to 1
vec3 softClip(vec3 x){ const float a = 0.75; vec3 over = max(x - a, 0.0); return min(x, vec3(a)) + (1.0 - a) * (1.0 - exp(-over / (1.0 - a))); }
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// anti-aliased "distance to line" -> coverage
float aaLine(float d, float w, float fw){ return 1.0 - smoothstep(w - fw, w + fw, d); }
`;

// Procedural surface patterns in world-scale meters.
// returns vec4(albedoMul, lineMask, emissive, gloss)
export const GLSL_PATTERNS = /* glsl */ `
vec4 surfacePattern(int pat, vec2 uv, vec3 wp, vec3 n, float t){
  vec2 fw2 = fwidth(uv);
  float fw = max(max(fw2.x, fw2.y), 1e-4);
  float alb = 1.0, line = 0.0, emis = 0.0, gloss = 0.1;
  if (pat == 0) { // stone: running bond of irregular blocks
    float rh = 0.62;
    float row = floor(uv.y / rh);
    float x = uv.x + hash12(vec2(row, 3.1)) * 5.0;
    float bw = 0.9 + 0.5 * hash12(vec2(row, 7.7));
    float col = floor(x / bw);
    vec2 l = vec2(fract(x / bw) * bw, fract(uv.y / rh) * rh);
    float d = min(min(l.x, bw - l.x), min(l.y, rh - l.y));
    line = aaLine(d, 0.018, fw);
    float h = hash12(vec2(row, col));
    alb = 0.72 + 0.4 * h + 0.25 * (fbm2(uv * 3.0) - 0.5);
    alb *= 1.0 - 0.18 * smoothstep(0.08, 0.0, d);
  } else if (pat == 1) { // tile
    float s = 0.6;
    vec2 g = abs(fract(uv / s - 0.5) - 0.5) * s;
    float d = min(g.x, g.y);
    line = aaLine(d, 0.01, fw);
    vec2 id = floor(uv / s);
    float chk = mod(id.x + id.y, 2.0);
    alb = 0.8 + 0.18 * chk + 0.12 * hash12(id) + 0.08 * (vnoise(uv * 9.0) - 0.5);
    gloss = 0.5;
  } else if (pat == 2) { // brick
    vec2 b = vec2(0.42, 0.16);
    float row = floor(uv.y / b.y);
    float x = uv.x + mod(row, 2.0) * b.x * 0.5;
    vec2 l = vec2(fract(x / b.x) * b.x, fract(uv.y / b.y) * b.y);
    float d = min(min(l.x, b.x - l.x), min(l.y, b.y - l.y));
    line = aaLine(d, 0.012, fw);
    alb = 0.75 + 0.35 * hash12(vec2(floor(x / b.x), row)) + 0.15 * (vnoise(uv * 14.0) - 0.5);
  } else if (pat == 3) { // metal panels with rivets
    vec2 s = vec2(1.2, 2.4);
    vec2 l = fract(uv / s) * s;
    float d = min(min(l.x, s.x - l.x), min(l.y, s.y - l.y));
    line = aaLine(d, 0.012, fw);
    vec2 c = min(l, s - l);
    float rv2 = length(c - 0.08);
    line = max(line, aaLine(abs(rv2 - 0.02), 0.006, fw));
    alb = 0.7 + 0.1 * hash12(floor(uv / s)) + 0.08 * vnoise(vec2(uv.x * 80.0, uv.y * 2.0));
    gloss = 0.8;
  } else if (pat == 4) { // wood planks
    float pw = 0.21;
    float row = floor(uv.y / pw);
    float x = uv.x + hash12(vec2(row, 1.3)) * 7.0;
    float plen = 2.4;
    vec2 l = vec2(fract(x / plen) * plen, fract(uv.y / pw) * pw);
    float d = min(min(l.x, plen - l.x), min(l.y, pw - l.y));
    line = aaLine(d, 0.006, fw);
    float grain = vnoise(vec2(x * 1.5, uv.y * 40.0 + fbm2(vec2(x * 0.7, row)) * 6.0));
    alb = 0.62 + 0.3 * hash12(vec2(row, floor(x / plen))) + 0.25 * grain;
  } else if (pat == 5) { // crystal facets (3D cells)
    vec3 p = wp * 1.4;
    vec3 ip = floor(p), fp = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int k = 0; k < 8; k++) {
      vec3 o = vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
      vec3 r = o + hash33(ip + o) - fp;
      float dd = dot(r, r);
      if (dd < d1) { d2 = d1; d1 = dd; } else if (dd < d2) { d2 = dd; }
    }
    float e = sqrt(d2) - sqrt(d1);
    line = 1.0 - smoothstep(0.02, 0.06 + fw * 2.0, e);
    alb = 0.7 + 0.5 * hash13(floor(p + 0.5));
    emis = 0.25 * line;
    gloss = 0.9;
  } else if (pat == 6) { // plaster
    float st = fbm2(uv * 0.7) ;
    alb = 0.85 + 0.2 * (st - 0.5) + 0.06 * (vnoise(uv * 12.0) - 0.5);
    vec2 g = abs(fract(uv / vec2(3.0, 3.0) - 0.5) - 0.5) * 3.0;
    line = aaLine(g.y, 0.008, fw) * 0.6;
  } else if (pat == 7) { // water: moving caustics
    vec2 q = uv * 0.9;
    float c = 0.0;
    for (int i = 0; i < 3; i++) {
      float fi = float(i);
      q += vec2(sin(q.y * 1.7 + t * (0.4 + 0.1 * fi)), cos(q.x * 1.3 - t * (0.35 + 0.12 * fi))) * 0.45;
      c += abs(sin(q.x * 2.1 + q.y * 1.6));
    }
    c = pow(saturate(1.0 - c / 3.0), 3.0);
    alb = 0.55 + 0.25 * fbm2(uv * 0.5 + t * 0.03);
    emis = c * 0.5;
    line = c;
    gloss = 1.0;
  } else if (pat == 8) { // sky: shaded in material
    alb = 1.0;
  } else if (pat == 9) { // glow bands
    float b = 0.5 + 0.5 * sin(uv.y * 3.0 - t * 0.8 + fbm2(uv * 0.5) * 3.0);
    emis = 0.6 + 0.9 * b * b;
    line = b;
    alb = 1.0;
  } else if (pat == 10) { // board-formed concrete: panel joints, tie holes, board grain.
    // Big plain slabs gave the model nothing to hold on to (it invented beams that drifted
    // as you walked); these marks are fixed to the surface, so its paint follows them.
    vec2 s = vec2(2.4, 1.2);
    vec2 g = abs(fract(uv / s - 0.5) - 0.5) * s;
    line = aaLine(min(g.x, g.y), 0.012, fw);
    vec2 th = abs(fract(uv / (s * 0.5) - 0.5) - 0.5) * s * 0.5;
    float hole = 1.0 - smoothstep(0.035, 0.035 + fw, length(th));
    float board = 0.035 * sin(uv.y * 20.9) * (1.0 - smoothstep(0.03, 0.1, fw));
    alb = 0.8 + 0.14 * (fbm2(uv * 0.5) - 0.5) + 0.1 * (hash12(floor(uv / s)) - 0.5) + board;
    alb *= 1.0 - 0.4 * hole;
  }
  return vec4(alb, line, emis, gloss);
}
`;

// Lights + zone data, shared by display and capture.
export const GLSL_LIGHTING = /* glsl */ `
uniform vec4 uLightPos[${MAX_LIGHTS}];   // xyz, radius
uniform vec4 uLightCol[${MAX_LIGHTS}];   // rgb*intensity (current, flickered), w = steady intensity ratio
uniform int uLightCount;
uniform vec4 uSurf[${MAX_SURF}];         // rgb color, emissive
uniform float uSurfPat[${MAX_SURF}];
uniform vec3 uZoneAmb[${MAX_ZONES}];
uniform vec3 uZoneLight[${MAX_ZONES}];
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uSkyColor;
uniform vec3 uSkyColor2;
uniform float uTime;

// returns diffuse light (rgb) and writes steady (unflickered) version
vec3 accumulateLights(vec3 p, vec3 n, out vec3 steady){
  vec3 acc = vec3(0.0);
  steady = vec3(0.0);
  for (int i = 0; i < ${MAX_LIGHTS}; i++) {
    if (i >= uLightCount) break;
    vec3 L = uLightPos[i].xyz - p;
    float d2 = dot(L, L);
    float r = uLightPos[i].w;
    float d = sqrt(d2);
    float win = saturate(1.0 - pow(d / r, 4.0));
    win *= win;
    float att = win / (0.6 + d2 * 0.55);
    float ndl = dot(n, L / max(d, 1e-3));
    float wrap = saturate((ndl + 0.35) / 1.35);
    vec3 c = uLightCol[i].rgb * wrap * att;
    acc += c;
    steady += c / max(uLightCol[i].w, 0.05);
  }
  return acc;
}

vec3 skyShade(vec3 dir, float t){
  float h = dir.y;
  vec3 col = mix(uSkyColor2, uSkyColor, smoothstep(-0.2, 0.7, h));
  float cl = fbm2(dir.xz / max(h + 0.25, 0.08) * 1.3 + vec2(t * 0.004, 0.0));
  col += uSkyColor * 0.35 * smoothstep(0.45, 0.85, cl) * smoothstep(-0.05, 0.3, h);
  // sparse stars
  vec2 sp = vec2(atan(dir.z, dir.x) * 60.0, asin(clamp(dir.y, -1.0, 1.0)) * 60.0);
  vec2 cell = floor(sp);
  float st = hash12(cell);
  float star = step(0.985, st) * smoothstep(0.3, 0.0, length(fract(sp) - 0.5)) * smoothstep(0.0, 0.25, h);
  col += vec3(0.9, 0.85, 1.0) * star * 1.5;
  return col;
}
`;

// Common vertex shader for world-space level geometry.
export const LEVEL_VERT = /* glsl */ `
attribute vec2 atlasUv;
attribute float surface;
attribute float zone;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUv;
varying vec2 vAtlasUv;
flat varying int vSurf;
flat varying int vZone;
varying float vViewZ;
void main(){
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vNormal = normalize(mat3(modelMatrix) * normal);
  vUv = uv;
  vAtlasUv = atlasUv;
  vSurf = int(surface + 0.5);
  vZone = int(zone + 0.5);
  vec4 mv = viewMatrix * wp;
  vViewZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;
