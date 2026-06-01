/* =========================================================================
   shaders.js  —  live GLSL fragment shaders for the Shader Library
   Each entry renders to a fullscreen triangle in WebGL2.
   A fixed uniform set is shared across every shader (unused ones are
   optimised out and guarded in the engine).
   ========================================================================= */

const SHADER_HEADER = `#version 300 es
precision highp float;

uniform vec2  uRes;        // canvas resolution (px)
uniform float uTime;       // seconds
uniform vec2  uMouse;      // mouse in px (origin bottom-left)
uniform float uSpeed;      // animation speed
uniform float uScale;      // pattern scale / zoom
uniform float uIntensity;  // brightness / glow
uniform float uWarp;       // distortion amount
uniform float uHue;        // hue rotation (radians)
uniform vec3  uColA;       // base color
uniform vec3  uColB;       // mid color
uniform vec3  uColC;       // highlight color

out vec4 fragColor;

float hash21(vec2 p){ p = fract(p*vec2(123.34,345.45)); p += dot(p,p+34.345); return fract(p.x*p.y); }
vec2  hash22(vec2 p){ vec3 a = fract(vec3(p.xyx)*vec3(123.34,234.34,345.65)); a += dot(a,a+34.45); return fract(vec2(a.x*a.y, a.y*a.z)); }

float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.0-2.0*f);
  float a = hash21(i), b = hash21(i+vec2(1,0));
  float c = hash21(i+vec2(0,1)), d = hash21(i+vec2(1,1));
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

float fbm(vec2 p){
  float v = 0.0, a = 0.5;
  mat2 m = mat2(1.6,1.2,-1.2,1.6);
  for(int i=0;i<6;i++){ v += a*noise(p); p = m*p; a *= 0.5; }
  return v;
}

vec3 hueShift(vec3 col, float h){
  const vec3 k = vec3(0.57735);
  float c = cos(h);
  return col*c + cross(k,col)*sin(h) + k*dot(k,col)*(1.0-c);
}
`;

/* Each shader supplies just its main() body via `body`.
   colors are authored 0-255 for readability (engine converts to 0-1). */

const SHADERS = [
  {
    id: 'fluid-flow',
    name: 'Fluid Flow',
    tag: 'DOMAIN WARP',
    blurb: 'Recursive domain-warped FBM — the signature billowing ink of the library.',
    colors: { uColA:[123,47,242], uColB:[255,45,120], uColC:[47,212,255] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2.5, step:0.01, value:1.0 },
      { key:'uScale',     name:'Scale',     min:0.5, max:5,   step:0.01, value:2.0 },
      { key:'uIntensity', name:'Intensity', min:0.3, max:2,   step:0.01, value:1.05 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.15;
  vec2 p = uv*uScale;
  vec2 q = vec2(fbm(p + t), fbm(p + vec2(5.2,1.3) - t));
  vec2 r = vec2(fbm(p + 4.0*q + vec2(1.7,9.2) + t*0.5),
                fbm(p + 4.0*q + vec2(8.3,2.8) - t*0.5));
  float f = fbm(p + 4.0*r);
  vec3 col = mix(uColA, uColB, clamp(f*f*2.0, 0.0, 1.0));
  col = mix(col, uColC, clamp(length(q), 0.0, 1.0));
  col = mix(col, uColB, clamp(r.x*r.x, 0.0, 1.0));
  col *= 0.55 + 0.7*f;
  col = pow(col, vec3(0.85));
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'plasma',
    name: 'Plasma Field',
    tag: 'TRIG SUM',
    blurb: 'Stacked sine fields beating against each other — the classic demoscene plasma.',
    colors: { uColA:[255,122,24], uColB:[230,37,107], uColC:[123,63,242] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:3,   step:0.01, value:1.0 },
      { key:'uScale',     name:'Scale',     min:1,   max:8,   step:0.01, value:3.5 },
      { key:'uWarp',      name:'Bands',     min:0.5, max:4,   step:0.01, value:1.6 },
      { key:'uIntensity', name:'Intensity', min:0.3, max:1.8, step:0.01, value:1.0 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.5;
  vec2 p = uv*uScale;
  float v = sin(p.x+t) + sin(p.y+t) + sin((p.x+p.y)*0.7+t) + sin(length(p)*2.0 - t);
  v += sin(length(p - vec2(sin(t*0.7), cos(t*0.5)))*3.0);
  v *= 0.2;
  vec3 col = mix(uColA, uColB, 0.5+0.5*sin(v*3.14159*uWarp));
  col = mix(col, uColC, 0.5+0.5*cos(v*3.14159));
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'curl-smoke',
    name: 'Curl Smoke',
    tag: 'FLOW FBM',
    blurb: 'Iterated advection of noise through its own gradient — drifting volumetric haze.',
    colors: { uColA:[10,8,26], uColB:[255,45,155], uColC:[47,212,255] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2.5, step:0.01, value:1.0 },
      { key:'uScale',     name:'Scale',     min:0.5, max:5,   step:0.01, value:2.4 },
      { key:'uIntensity', name:'Density',   min:0.4, max:2,   step:0.01, value:1.1 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.2;
  vec2 p = uv*uScale;
  float n = 0.0;
  for(int i=0;i<3;i++){
    p += 0.4*vec2(fbm(p + t), fbm(p.yx - t));
    n += fbm(p);
  }
  n /= 3.0;
  float d = smoothstep(0.2, 0.9, n);
  vec3 col = mix(uColA, uColB, d);
  col = mix(col, uColC, pow(d, 3.0));
  col *= 0.5 + 1.5*d;
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'metaballs',
    name: 'Metaballs',
    tag: 'IMPLICIT',
    blurb: 'Seven orbiting potential fields summed into a gooey, merging surface.',
    colors: { uColA:[26,10,42], uColB:[255,122,24], uColC:[255,45,120] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:3,   step:0.01, value:1.0 },
      { key:'uScale',     name:'Radius',    min:0.1, max:1.2, step:0.01, value:0.55 },
      { key:'uIntensity', name:'Glow',      min:0.4, max:2,   step:0.01, value:1.1 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.6;
  float m = 0.0;
  for(int i=0;i<7;i++){
    float fi = float(i);
    vec2 c = 0.62*vec2(sin(t*0.7 + fi*1.3), cos(t*0.6 + fi*2.1)) * (0.5+0.5*sin(fi));
    m += (0.05*uScale)/length(uv - c);
  }
  float e = smoothstep(0.8, 1.4, m);
  vec3 col = mix(uColA, uColB, e);
  col = mix(col, uColC, smoothstep(1.2, 2.4, m));
  col += uColC*0.3*smoothstep(0.6, 0.85, m);
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'aurora',
    name: 'Aurora Veil',
    tag: 'LAYERED',
    blurb: 'Stacked light ribbons bent by low-frequency noise — a slow polar curtain.',
    colors: { uColA:[123,63,242], uColB:[47,212,255], uColC:[255,45,155] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2,   step:0.01, value:1.0 },
      { key:'uScale',     name:'Waviness',  min:0.4, max:3,   step:0.01, value:1.2 },
      { key:'uIntensity', name:'Glow',      min:0.3, max:1.6, step:0.01, value:0.9 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = gl_FragCoord.xy/uRes.xy;
  vec2 p = uv - 0.5; p.x *= uRes.x/uRes.y;
  float t = uTime*uSpeed*0.3;
  vec3 col = vec3(0.0);
  for(float i=0.0;i<4.0;i++){
    float y = p.y + 0.30*sin(p.x*2.0*uScale + t + i*1.7) + 0.15*fbm(p*3.0 + t + i);
    float band = 0.018/abs(y - (-0.12 + i*0.13));
    vec3 c = mix(uColA, i<1.5 ? uColB : uColC, i/3.0);
    col += c*band;
  }
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity*0.6, 1.0);
}`
  },

  {
    id: 'voronoi',
    name: 'Voronoi Cells',
    tag: 'CELLULAR',
    blurb: 'Animated feature points with glowing partition borders — living stained glass.',
    colors: { uColA:[58,16,96], uColB:[47,212,255], uColC:[255,45,120] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2.5, step:0.01, value:1.0 },
      { key:'uScale',     name:'Density',   min:1,   max:8,   step:0.01, value:3.0 },
      { key:'uIntensity', name:'Border',    min:0.4, max:2,   step:0.01, value:1.1 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.3;
  vec2 p = uv*uScale;
  vec2 g = floor(p), f = fract(p);
  float md = 8.0; vec2 mr;
  for(int y=-1;y<=1;y++)
  for(int x=-1;x<=1;x++){
    vec2 o = vec2(float(x), float(y));
    vec2 r = hash22(g+o);
    vec2 pos = o + 0.5 + 0.5*sin(t + 6.2831*r) - f;
    float d = dot(pos,pos);
    if(d < md){ md = d; mr = r; }
  }
  float edge = sqrt(md);
  vec3 col = mix(uColA, uColB, hash21(mr));
  col = mix(col, uColC, hash21(mr+3.0)*0.5);
  col *= 0.35 + 0.9*edge;
  col += uColC*smoothstep(0.06, 0.0, edge);
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'caustics',
    name: 'Caustics',
    tag: 'REFRACTIVE',
    blurb: 'Folded coordinate space refracts light into shifting underwater filaments.',
    colors: { uColA:[6,18,31], uColB:[47,212,255], uColC:[160,240,255] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2.5, step:0.01, value:1.0 },
      { key:'uScale',     name:'Scale',     min:1,   max:8,   step:0.01, value:3.2 },
      { key:'uWarp',      name:'Sharpness', min:1,   max:8,   step:0.01, value:4.0 },
      { key:'uIntensity', name:'Glow',      min:0.4, max:2,   step:0.01, value:1.1 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.4;
  vec2 q = uv*uScale;
  for(int i=0;i<4;i++){
    q += vec2(sin(q.y*1.5 + t), cos(q.x*1.5 - t))*0.5;
  }
  float c = pow(1.0 - abs(sin(q.x)*cos(q.y)), uWarp);
  vec3 col = mix(uColA, uColB, c);
  col += uColC*pow(c, 2.0);
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'liquid-marble',
    name: 'Liquid Marble',
    tag: 'TURBULENCE',
    blurb: 'Noise warped through itself then folded into sine veins — flowing marble.',
    colors: { uColA:[26,6,51], uColB:[255,45,120], uColC:[255,176,32] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2,   step:0.01, value:1.0 },
      { key:'uScale',     name:'Scale',     min:0.5, max:5,   step:0.01, value:2.2 },
      { key:'uWarp',      name:'Veining',   min:0,   max:4,   step:0.01, value:2.0 },
      { key:'uIntensity', name:'Intensity', min:0.4, max:1.8, step:0.01, value:1.0 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.1;
  vec2 p = uv*uScale;
  float warp = fbm(p*1.5 + t);
  float marble = fbm(p + warp*uWarp + vec2(t,0.0));
  float v = 0.5 + 0.5*sin((p.x + marble*4.0)*3.14159);
  vec3 col = mix(uColA, uColB, v);
  col = mix(col, uColC, smoothstep(0.4, 0.85, marble));
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  {
    id: 'fluid-cursor',
    name: 'Fluid Cursor',
    tag: 'INTERACTIVE',
    blurb: 'A swirling glow that bends the field around your pointer. Move the mouse.',
    colors: { uColA:[6,6,15], uColB:[255,45,155], uColC:[47,212,255] },
    params: [
      { key:'uSpeed',     name:'Speed',     min:0,   max:2.5, step:0.01, value:1.0 },
      { key:'uScale',     name:'Scale',     min:0.5, max:4,   step:0.01, value:1.6 },
      { key:'uWarp',      name:'Swirl',     min:0,   max:3,   step:0.01, value:1.4 },
      { key:'uIntensity', name:'Glow',      min:0.4, max:2,   step:0.01, value:1.2 },
      { key:'uHue',       name:'Hue',       min:-3.1,max:3.1, step:0.01, value:0.0 }
    ],
    body: `
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  vec2 m  = (uMouse - 0.5*uRes)/uRes.y;
  float t = uTime*uSpeed*0.3;
  vec2 toM = uv - m;
  float dist = length(toM);
  float ang = atan(toM.y, toM.x);
  float swirl = ang + (0.6/(dist+0.2))*sin(t*2.0)*uWarp;
  vec2 p = vec2(cos(swirl), sin(swirl))*dist*uScale;
  float f = fbm(p*2.0 + t);
  float glow = smoothstep(1.2, 0.0, dist);
  vec3 col = mix(uColA, uColB, f);
  col = mix(col, uColC, glow);
  col += uColC*glow*0.6;
  col = hueShift(col, uHue);
  fragColor = vec4(col*uIntensity, 1.0);
}`
  },

  /* ===== three-fluid-fx ports (engine-driven, not single-pass GLSL) =====
     These cards are powered by fluidfx.js — a real Navier–Stokes fluid sim
     and a GPGPU particle system, both reacting to the pointer. */
  {
    id: 'fluid-sim',
    name: 'Fluid Sim',
    tag: 'NAVIER–STOKES',
    kind: 'fluid',
    overlay: 'colorful',
    interactive: true,
    blurb: 'A true Stable-Fluids solver. Drag the pointer to inject swirling, rainbow ink into the velocity field.',
    colors: { uColA:[255,45,155], uColB:[47,212,255], uColC:[123,63,242] },
    bg: [0.02,0.02,0.05],
    sim: { curlStrength:0.45, velocityDissipation:0.985, densityDissipation:0.94 },
    state: { intensity:1.2, vibrance:0.2 },
    params: [
      { key:'intensity',           name:'Glow',     min:0.3, max:3,   step:0.01,  value:1.2 },
      { key:'splatForce',          name:'Force',    min:1,   max:18,  step:0.1,   value:6 },
      { key:'curlStrength',        name:'Curl',     min:0,   max:2,   step:0.01,  value:0.45 },
      { key:'velocityDissipation', name:'Vel diss', min:0.9, max:1,   step:0.001, value:0.985 }
    ]
  },

  {
    id: 'rainbow-ink',
    name: 'Rainbow Ink',
    tag: 'FLUID · DYE',
    kind: 'fluid',
    overlay: 'rainbowink',
    interactive: true,
    blurb: 'Coloured dye advected by the fluid — each stroke leaves a multi-hue ribbon that mixes as it flows.',
    colors: { uColA:[255,80,80], uColB:[80,160,255], uColC:[120,255,120] },
    bg: [0.015,0.012,0.03],
    sim: { curlStrength:0.5, velocityDissipation:0.99, densityDissipation:0.97, dyeDissipation:0.975 },
    state: { intensity:1.0, vibrance:0.25 },
    params: [
      { key:'intensity',           name:'Ink',      min:0.3, max:3,   step:0.01,  value:1.0 },
      { key:'splatForce',          name:'Force',    min:1,   max:18,  step:0.1,   value:6 },
      { key:'curlStrength',        name:'Curl',     min:0,   max:2,   step:0.01,  value:0.5 },
      { key:'velocityDissipation', name:'Vel diss', min:0.9, max:1,   step:0.001, value:0.99 }
    ]
  },

  {
    id: 'flow-particles',
    name: 'Flow Particles',
    tag: 'GPGPU · 2D',
    kind: 'particle',
    mode: 'plane2d',
    interactive: true,
    blurb: 'A disc of thousands of GPGPU billboards, sprung to a rest pose and shoved around by the fluid velocity.',
    colors: { uColA:[255,120,40], uColB:[230,40,120], uColC:[120,80,255] },
    bg: [0.03,0.02,0.05],
    tint: [0.18,0.05,0.22],
    params: [
      { key:'pointSize',    name:'Size',   min:1,   max:16, step:0.1,  value:9 },
      { key:'flowStrength', name:'Flow',   min:0,   max:5,  step:0.01, value:1.05 },
      { key:'spring',       name:'Spring', min:0.4, max:6,  step:0.01, value:4.0 },
      { key:'dragLin',      name:'Drag',   min:0,   max:2,  step:0.01, value:0.28 }
    ]
  },

  {
    id: 'particle-cloud',
    name: 'Particle Cloud',
    tag: 'GPGPU · 3D',
    kind: 'particle',
    mode: 'cloud3d',
    interactive: true,
    blurb: 'A slowly spinning sphere of particles disturbed by the velocity field — the signature three-fluid-fx cloud.',
    colors: { uColA:[80,180,255], uColB:[255,80,160], uColC:[255,180,60] },
    bg: [0.027,0.031,0.043],
    tint: [0.08,0.3,0.32],
    state: { rotationSpeed:0.07, depthLift:0.95 },
    params: [
      { key:'pointSize',     name:'Size',   min:1,   max:16, step:0.1,  value:9 },
      { key:'flowStrength',  name:'Flow',   min:0,   max:5,  step:0.01, value:1.05 },
      { key:'spring',        name:'Spring', min:0.4, max:6,  step:0.01, value:4.0 },
      { key:'rotationSpeed', name:'Spin',   min:-2,  max:2,  step:0.01, value:0.07 }
    ]
  },

  {
    id: 'mega-demo',
    name: 'Mega Demo',
    tag: 'COMBINED',
    kind: 'particle',
    mode: 'cloud3d',
    interactive: true,
    backdrop: true,
    particleSize: 64,
    blurb: 'Everything at once — a dense particle cloud over an FBM-lit gradient backdrop, stirred by the fluid velocity field.',
    colors: { uColA:[255,120,40], uColB:[80,160,255], uColC:[200,60,180] },
    bg: [0.02,0.02,0.05],
    tint: [0.10,0.22,0.34],
    backdropWarm: [0.62,0.0,0.20],
    backdropCool: [0.10,0.25,0.80],
    state: { rotationSpeed:0.06, depthLift:1.0, pointSize:8 },
    params: [
      { key:'pointSize',     name:'Size',   min:1,   max:16, step:0.1,  value:8 },
      { key:'flowStrength',  name:'Flow',   min:0,   max:5,  step:0.01, value:1.1 },
      { key:'spring',        name:'Spring', min:0.4, max:6,  step:0.01, value:4.0 },
      { key:'rotationSpeed', name:'Spin',   min:-2,  max:2,  step:0.01, value:0.06 }
    ]
  },

  {
    id: 'fluid-text',
    name: 'Fluid Text',
    tag: 'TYPE · FLUID',
    kind: 'fluidtext',
    interactive: true,
    text: 'Fluid Text',
    kicker: 'THREE · FLUID · FX',
    lead: 'Live type, bent by fluid.',
    blurb: 'Live typography refracted by the velocity field, with rainbow ink trailing the cursor.',
    colors: { uColA:[255,122,95], uColB:[120,200,255], uColC:[255,80,180] },
    bg: [0.02,0.02,0.05],
    state: { intensity:1.0, distort:0.5, vibrance:0.4 },
    params: [
      { key:'distort',      name:'Distort', min:0,   max:1.5, step:0.01, value:0.5 },
      { key:'intensity',    name:'Ink',     min:0,   max:3,   step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force',   min:1,   max:18,  step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',    min:0,   max:2,   step:0.01, value:0.18 }
    ]
  },

  /* ===== three-fluid-fx DISTORTION passes (engine-driven) =====
     Post-process refraction of a procedural backdrop by the fluid field.
     Drag the pointer to bend the scene. */
  {
    id:'simple-distortion', name:'Simple Distortion', tag:'DISTORT · UV', kind:'fluid',
    overlay:'simple', group:'Distortion', scene:'text', text:'mydesignbox', interactive:true,
    blurb:'Straight UV warp of the mydesignbox wordmark along the velocity field — the cheapest distortion, no chromatic split.',
    colors:{ uColA:[176,38,255], uColB:[255,42,133], uColC:[255,178,74] }, bg:[0,0,0],
    state:{ intensity:1.7, opacity:1.0 }, sim:{ curlStrength:0.32 },
    params:[
      { key:'intensity',    name:'Amount', min:0, max:5,  step:0.01, value:1.7 },
      { key:'opacity',      name:'Mix',    min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force',  min:1, max:18, step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',   min:0, max:2,  step:0.01, value:0.32 }
    ]
  },
  {
    id:'rgb-shift', name:'RGB Shift', tag:'DISTORT · RGB', kind:'fluid',
    overlay:'rgbshift', group:'Distortion', scene:'text', text:'mydesignbox', interactive:true,
    blurb:'Density-driven chromatic split on the wordmark — R and B slide apart along the flow while green stays put.',
    colors:{ uColA:[176,38,255], uColB:[255,42,133], uColC:[255,178,74] }, bg:[0,0,0],
    state:{ intensity:1.5, opacity:1.0 }, sim:{ curlStrength:0.35 },
    params:[
      { key:'intensity',    name:'Split', min:0, max:4,  step:0.01, value:1.5 },
      { key:'opacity',      name:'Mix',   min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force', min:1, max:18, step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',  min:0, max:2,  step:0.01, value:0.35 }
    ]
  },
  {
    id:'chromatic-distortion', name:'Chromatic', tag:'DISTORT · OIL', kind:'fluid',
    overlay:'chromatic', group:'Distortion', scene:'text', text:'mydesignbox', interactive:true,
    blurb:'Oil-slick chromatic distortion of the wordmark — each RGB channel rides its own velocity component off a blurred field.',
    colors:{ uColA:[176,38,255], uColB:[255,42,133], uColC:[255,178,74] }, bg:[0,0,0],
    state:{ intensity:1.4, opacity:1.0 }, sim:{ curlStrength:0.4 },
    params:[
      { key:'intensity',    name:'Chroma', min:0, max:4,  step:0.01, value:1.4 },
      { key:'opacity',      name:'Mix',    min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force',  min:1, max:18, step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',   min:0, max:2,  step:0.01, value:0.4 }
    ]
  },
  {
    id:'water-distortion', name:'Water', tag:'DISTORT · REFRACT', kind:'fluid',
    overlay:'water', group:'Distortion', scene:'text', text:'mydesignbox', interactive:true,
    blurb:'Density-as-height refraction — the dye gradient bends the wordmark with a per-channel Snell split.',
    colors:{ uColA:[176,38,255], uColB:[255,42,133], uColC:[255,178,74] }, bg:[0,0,0],
    state:{ intensity:1.3, opacity:1.0 }, sim:{ curlStrength:0.45 },
    params:[
      { key:'intensity',    name:'Refract', min:0, max:3,  step:0.01, value:1.3 },
      { key:'opacity',      name:'Mix',     min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force',   min:1, max:18, step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',    min:0, max:2,  step:0.01, value:0.45 }
    ]
  },
  {
    id:'water-caustics', name:'Water Caustics', tag:'DISTORT · CAUSTIC', kind:'fluid',
    overlay:'watercaustics', group:'Distortion', scene:'text', text:'mydesignbox', interactive:true,
    blurb:'Surface refraction of the wordmark plus a procedural caustic web, gated by the fluid surface and slope.',
    colors:{ uColA:[176,38,255], uColB:[255,42,133], uColC:[255,178,74] }, bg:[0,0,0],
    state:{ intensity:1.3, opacity:1.0 }, sim:{ curlStrength:0.4 },
    params:[
      { key:'intensity',    name:'Refract', min:0, max:3,  step:0.01, value:1.3 },
      { key:'opacity',      name:'Mix',     min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force',   min:1, max:18, step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',    min:0, max:2,  step:0.01, value:0.4 }
    ]
  },

  /* ===== three-fluid-fx OVERLAY passes (engine-driven) =====
     A fluid-coloured layer composited over the procedural scene. */
  {
    id:'soft-cursor', name:'Soft Cursor', tag:'OVERLAY · DYE', kind:'fluid',
    overlay:'default', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Cursor-coloured dye haze with a core/rim gradient down the stroke body — the canonical fluid cursor.',
    colors:{ uColA:[10,10,22], uColB:[120,200,255], uColC:[0,212,255] }, bg:[0.02,0.02,0.05],
    state:{ intensity:1.2, vibrance:0.3, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Ink',      min:0, max:3,   step:0.01, value:1.2 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.3 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'volume-cursor', name:'Volume Cursor', tag:'OVERLAY · DYE', kind:'fluid',
    overlay:'volume', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Dye haze with fake-normal volumetric shading taken from the dye thickness gradient.',
    colors:{ uColA:[12,10,24], uColB:[127,234,255], uColC:[127,234,255] }, bg:[0.02,0.02,0.05],
    state:{ intensity:1.1, vibrance:0.25, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Ink',      min:0, max:3,   step:0.01, value:1.1 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.25 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'trail', name:'Trail', tag:'OVERLAY · WAKE', kind:'fluid',
    overlay:'trail', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Directional wake — a sharp leading edge with a long fade tail in the cursor colour.',
    colors:{ uColA:[10,8,20], uColB:[255,61,138], uColC:[255,120,180] }, bg:[0.02,0.018,0.04],
    state:{ intensity:1.2, vibrance:0.25, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Glow',     min:0, max:3,   step:0.01, value:1.2 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.25 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'oil', name:'Oil', tag:'OVERLAY · PALETTE', kind:'fluid',
    overlay:'oil', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Multi-tap density glow with an animated ember / mint / cream palette — a slick of oil on the flow.',
    colors:{ uColA:[12,10,8], uColB:[255,84,51], uColC:[20,200,174] }, bg:[0.022,0.018,0.03],
    state:{ intensity:1.1, vibrance:0.3, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Glow',     min:0, max:3,   step:0.01, value:1.1 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.3 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'velocity-field', name:'Velocity Field', tag:'OVERLAY · VEL', kind:'fluid',
    overlay:'velocity', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'The raw velocity field rendered additively as an RG-coloured glaze — see the flow itself.',
    colors:{ uColA:[8,8,18], uColB:[80,140,255], uColC:[255,80,160] }, bg:[0.02,0.02,0.045],
    state:{ intensity:1.3, opacity:1.0 },
    params:[
      { key:'intensity',    name:'Glow',  min:0, max:4,  step:0.01, value:1.3 },
      { key:'opacity',      name:'Mix',   min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force', min:1, max:18, step:0.1,  value:6 },
      { key:'curlStrength', name:'Curl',  min:0, max:2,  step:0.01, value:0.5 }
    ]
  },
  {
    id:'rainbow-fish', name:'Rainbow Fish', tag:'OVERLAY · HUE', kind:'fluid',
    overlay:'rainbowfish', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Hue from the velocity angle, brightness from speed — a closed rainbow ring around every vortex.',
    colors:{ uColA:[10,8,20], uColB:[80,200,255], uColC:[255,80,200] }, bg:[0.02,0.018,0.04],
    state:{ intensity:1.2, vibrance:0.35, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Glow',     min:0, max:3,   step:0.01, value:1.2 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.35 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:7 }
    ]
  },
  {
    id:'glaze', name:'Glaze', tag:'OVERLAY · TINT', kind:'fluid',
    overlay:'glaze', group:'Overlay', scene:'shapes', interactive:true,
    blurb:'Minimal additive density tint — the simplest overlay that still reads as fluid.',
    colors:{ uColA:[16,10,6], uColB:[255,115,56], uColC:[255,170,80] }, bg:[0.022,0.016,0.03],
    state:{ intensity:1.2, vibrance:0.2, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Glaze',    min:0, max:3,   step:0.01, value:1.2 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.2 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'burn', name:'Burn', tag:'OVERLAY · FIRE', kind:'fluid',
    overlay:'burn', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Flame fingers flung along the velocity field, ember-to-flame palette, flickering.',
    colors:{ uColA:[10,6,4], uColB:[255,77,0], uColC:[255,180,40] }, bg:[0.02,0.012,0.02],
    state:{ intensity:1.3, vibrance:0.25, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Heat',     min:0, max:3,   step:0.01, value:1.3 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.25 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:7 }
    ]
  },
  {
    id:'smoke', name:'Smoke', tag:'OVERLAY · DYE', kind:'fluid',
    overlay:'smoke', group:'Overlay', scene:'shapes', interactive:true,
    blurb:'A white cigarette-smoke wash whose opacity is the per-stroke dye density.',
    colors:{ uColA:[16,16,20], uColB:[220,225,235], uColC:[245,247,255] }, bg:[0.025,0.025,0.03],
    state:{ intensity:1.1, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Smoke', min:0, max:3,   step:0.01, value:1.1 },
      { key:'opacity',    name:'Mix',   min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force', min:1, max:18,  step:0.1,  value:6 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.0 }
    ]
  },
  {
    id:'art-ink', name:'Art Ink', tag:'OVERLAY · INK', kind:'fluid',
    overlay:'artink', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'Per-stroke dye boosted to a vibrant, saturated ink and composited additively.',
    colors:{ uColA:[8,8,16], uColB:[255,80,160], uColC:[80,200,255] }, bg:[0.018,0.018,0.035],
    state:{ intensity:1.0, vibrance:0.4, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Ink',      min:0, max:3,   step:0.01, value:1.0 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.4 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:7 }
    ]
  },
  {
    id:'color-water', name:'Color Water', tag:'OVERLAY · WASH', kind:'fluid',
    overlay:'colorwater', group:'Overlay', scene:'gradient', interactive:true,
    blurb:'A watercolour wash — translucent alpha-mix tint plus a soft scene-tinted glow.',
    colors:{ uColA:[8,12,16], uColB:[80,180,220], uColC:[255,140,90] }, bg:[0.016,0.02,0.03],
    state:{ intensity:1.1, vibrance:0.3, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Wash',     min:0, max:3,   step:0.01, value:1.1 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.3 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'liquid-lens', name:'Liquid Lens', tag:'OVERLAY · LENS', kind:'fluid',
    overlay:'liquidlens', group:'Overlay', scene:'torus', interactive:true,
    blurb:'Velocity-refracted scene multiplied by a dye tint — the dreamers.js liquid lens port.',
    colors:{ uColA:[6,14,20], uColB:[60,200,210], uColC:[255,120,160] }, bg:[0.012,0.025,0.035],
    state:{ intensity:1.1, vibrance:0.3, opacity:1.0 },
    params:[
      { key:'intensity',  name:'Tint',     min:0, max:3,   step:0.01, value:1.1 },
      { key:'vibrance',   name:'Vibrance', min:0, max:1.5, step:0.01, value:0.3 },
      { key:'opacity',    name:'Mix',      min:0, max:1,   step:0.01, value:1.0 },
      { key:'splatForce', name:'Force',    min:1, max:18,  step:0.1,  value:6 }
    ]
  },
  {
    id:'density-tint', name:'Density Tint', tag:'OVERLAY · TINT', kind:'fluid',
    overlay:'dtint', group:'Overlay', scene:'shapes', interactive:true,
    blurb:'A subtle additive tint proportional to fluid density — the gentlest hint of a fluid cursor.',
    colors:{ uColA:[6,18,16], uColB:[25,168,154], uColC:[120,255,220] }, bg:[0.014,0.03,0.028],
    state:{ intensity:1.4, opacity:1.0 },
    params:[
      { key:'intensity',    name:'Tint',  min:0, max:4,  step:0.01, value:1.4 },
      { key:'opacity',      name:'Mix',   min:0, max:1,  step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force', min:1, max:18, step:0.1,  value:6 },
      { key:'curlStrength', name:'Curl',  min:0, max:2,  step:0.01, value:0.45 }
    ]
  },

  /* ===== mydesignbox branded fluid type (FluidTextView) =====
     Same engine as Fluid Text — live DOM-style type rasterised to a texture,
     refracted by the velocity gradient with rainbow dye trailing the cursor. */
  {
    id:'mydesignbox', name:'My Design Box', tag:'TYPE · BRAND', kind:'fluidtext',
    group:'Type', interactive:true,
    text:'mydesignbox', kicker:'MY · DESIGN · BOX', lead:'Creative code & design.',
    blurb:'The mydesignbox wordmark as live type, bent by the fluid and trailed by rainbow ink.',
    colors:{ uColA:[255,122,95], uColB:[243,240,232], uColC:[80,200,255] }, bg:[0.02,0.02,0.05],
    state:{ intensity:1.1, distort:0.55, vibrance:0.4 },
    params:[
      { key:'distort',      name:'Distort', min:0, max:1.5, step:0.01, value:0.55 },
      { key:'intensity',    name:'Ink',     min:0, max:3,   step:0.01, value:1.1 },
      { key:'splatForce',   name:'Force',   min:1, max:18,  step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',    min:0, max:2,   step:0.01, value:0.18 }
    ]
  },
  {
    id:'mydesignbox-fi', name:'mydesignbox.fi', tag:'TYPE · DOMAIN', kind:'fluidtext',
    group:'Type', interactive:true,
    text:'mydesignbox.fi', kicker:'LIVE · WEBGL2 · 2026', lead:'Shaders, bent by fluid.',
    blurb:'The live domain wordmark — refracted typography with a multi-hue ink wake.',
    colors:{ uColA:[255,90,140], uColB:[120,200,255], uColC:[255,200,120] }, bg:[0.018,0.02,0.045],
    state:{ intensity:1.0, distort:0.65, vibrance:0.45 },
    params:[
      { key:'distort',      name:'Distort', min:0, max:1.5, step:0.01, value:0.65 },
      { key:'intensity',    name:'Ink',     min:0, max:3,   step:0.01, value:1.0 },
      { key:'splatForce',   name:'Force',   min:1, max:18,  step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',    min:0, max:2,   step:0.01, value:0.2 }
    ]
  },
  {
    id:'mydesignbox-tagline', name:'Make It Move', tag:'TYPE · TAGLINE', kind:'fluidtext',
    group:'Type', interactive:true,
    text:'Make it move', kicker:'MYDESIGNBOX', lead:'Type that flows with you.',
    blurb:'A tagline that lives — every word warps and inks as the fluid passes through it.',
    colors:{ uColA:[120,255,200], uColB:[255,122,95], uColC:[180,140,255] }, bg:[0.02,0.018,0.045],
    state:{ intensity:1.15, distort:0.5, vibrance:0.5 },
    params:[
      { key:'distort',      name:'Distort', min:0, max:1.5, step:0.01, value:0.5 },
      { key:'intensity',    name:'Ink',     min:0, max:3,   step:0.01, value:1.15 },
      { key:'splatForce',   name:'Force',   min:1, max:18,  step:0.1,  value:7 },
      { key:'curlStrength', name:'Curl',    min:0, max:2,   step:0.01, value:0.18 }
    ]
  }
];

/* Coarse families for the gallery filter bar. New engine cards carry their own
   `group`; the original procedural + engine cards are tagged here by id. */
const GROUPS = {
  'fluid-flow':'Procedural', 'plasma':'Procedural', 'curl-smoke':'Procedural',
  'metaballs':'Procedural', 'aurora':'Procedural', 'voronoi':'Procedural',
  'caustics':'Procedural', 'liquid-marble':'Procedural', 'fluid-cursor':'Procedural',
  'fluid-sim':'Overlay', 'rainbow-ink':'Overlay',
  'flow-particles':'Particles', 'particle-cloud':'Particles', 'mega-demo':'Particles',
  'fluid-text':'Type',
};
SHADERS.forEach(s => { if (!s.group) s.group = GROUPS[s.id] || 'Procedural'; });

window.SHADER_HEADER = SHADER_HEADER;
window.SHADERS = SHADERS;
