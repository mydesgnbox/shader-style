/* =========================================================================
   fluidfx.js — vanilla WebGL2 port of the three-fluid-fx effects
   (https://github.com/artcodev/three-fluid-fx)

   Ports, with no Three.js dependency:
     • FluidSim       — Stable-Fluids Navier-Stokes solver (splat → curl →
                        vorticity → divergence → pressure → gradient-subtract
                        → advect), the same multi-pass pipeline the library
                        runs. Produces velocity / density / dye textures.
     • Overlay passes — Colorful (rainbow back-trace), RainbowInk (dye hue),
                        Volume/Smoke (volumetric dye), DensityTint.
     • ParticleSystem — GPGPU flow particles advected by the velocity field
                        (disc / sphere), instanced billboards, energy-driven
                        emissive palette.
     • FluidView / ParticleView — wire a canvas + sim (+ particles) + pointer
                        splats + idle auto-motion, exposing the same surface
                        (.active / .render(time,dt) / .dispose) the gallery
                        engine already drives.
   ========================================================================= */
window.FluidFX = (() => {
  'use strict';

  /* ----------------------------------------------------------------- GL utils */
  function compile(gl, type, src, label) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('[fluidfx] compile', label, gl.getShaderInfoLog(sh));
      return null;
    }
    return sh;
  }

  // attribs: { name: location } bound before link so geometry VAOs can use
  // fixed locations across every program.
  function program(gl, vsSrc, fsSrc, attribs, label) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsSrc, label + '.vs');
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc, label + '.fs');
    if (!vs || !fs) return null;
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    if (attribs) for (const n in attribs) gl.bindAttribLocation(p, attribs[n], n);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.error('[fluidfx] link', label, gl.getProgramInfoLog(p));
      return null;
    }
    const loc = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      const name = info.name.replace(/\[0\]$/, '');
      loc[name] = gl.getUniformLocation(p, name);
    }
    return { p, loc };
  }

  // RGBA16F is colour-renderable via EXT_color_buffer_float and is guaranteed
  // texture-filterable (LINEAR) in core WebGL2 — no float-linear extension.
  function makeTex(gl, w, h, filter, internalFormat, format, type, data) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, data || null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  function makeFBO(gl, w, h, opts) {
    opts = opts || {};
    const filter = opts.filter || gl.NEAREST;
    const internalFormat = opts.internalFormat || gl.RGBA16F;
    const format = opts.format || gl.RGBA;
    const type = opts.type || gl.HALF_FLOAT;
    const tex = makeTex(gl, w, h, filter, internalFormat, format, type, opts.data);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo, w, h, filter, internalFormat, format, type };
  }

  function makeDouble(gl, w, h, opts) {
    return { read: makeFBO(gl, w, h, opts), write: makeFBO(gl, w, h, opts), w, h };
  }
  function swap(d) { const r = d.read; d.read = d.write; d.write = r; }

  /* ------------------------------------------------------------ mini matrix lib
     Column-major Float32Array(16) — WebGL upload convention. */
  const M = {
    ident: () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
      ]);
    },
    translation(x, y, z) {
      return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
    },
    rotationY(a) {
      const c = Math.cos(a), s = Math.sin(a);
      return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
    },
    mul(a, b) { // a * b
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
      return o;
    },
    // 3x3 rotation block of a mat4, inverted (orthonormal rotation → transpose)
    invRot3(m) {
      // transpose of the upper-left 3x3 (valid for pure rotation)
      return new Float32Array([m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]]);
    },
    mul3v(m3, x, y, z) {
      return [
        m3[0] * x + m3[3] * y + m3[6] * z,
        m3[1] * x + m3[4] * y + m3[7] * z,
        m3[2] * x + m3[5] * y + m3[8] * z,
      ];
    },
    norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; },
  };

  function hsv2rgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, p, v];
      default: return [v, p, q];
    }
  }

  /* =======================================================================
     Simulation shaders (ported to GLSL ES 3.00)
     ======================================================================= */
  const SIM_VS = `#version 300 es
  in vec2 position;
  out vec2 vUv; out vec2 vL; out vec2 vR; out vec2 vT; out vec2 vB;
  uniform vec2 texelSize;
  void main(){
    vUv = position * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(position, 0.0, 1.0);
  }`;

  const QUAD_VS = `#version 300 es
  in vec2 position; out vec2 vUv;
  void main(){ vUv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }`;

  const CLEAR_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; uniform sampler2D uTexture; uniform float value; out vec4 o;
  void main(){ o = value * texture(uTexture, vUv); }`;

  const SPLAT_VS = `#version 300 es
  in vec2 position; out vec2 vLocalUv; uniform vec2 uCenter; uniform vec2 uScale;
  void main(){ vLocalUv = position; gl_Position = vec4(position * uScale + uCenter, 0.0, 1.0); }`;

  const SPLAT_FS = `#version 300 es
  precision highp float; in vec2 vLocalUv; uniform vec3 color; out vec4 o;
  void main(){ float r = length(vLocalUv); if (r > 1.0) discard; float a = 1.0 - r; a *= a; o = vec4(color * a, a); }`;

  const CURL_FS = `#version 300 es
  precision highp float;
  in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
  uniform sampler2D uVelocity; out vec4 o;
  void main(){
    float L = texture(uVelocity, vL).y;
    float R = texture(uVelocity, vR).y;
    float T = texture(uVelocity, vT).x;
    float B = texture(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    o = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
  }`;

  const VORTICITY_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
  uniform sampler2D uVelocity; uniform sampler2D uCurl; uniform float curl; uniform float dt; out vec4 o;
  void main(){
    float L = texture(uCurl, vL).x;
    float R = texture(uCurl, vR).x;
    float T = texture(uCurl, vT).x;
    float B = texture(uCurl, vB).x;
    float C = texture(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 vel = texture(uVelocity, vUv).xy;
    o = vec4(vel + force * dt, 0.0, 1.0);
  }`;

  const DIVERGENCE_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
  uniform sampler2D uVelocity; uniform float uReflectWalls; out vec4 o;
  void main(){
    float L = texture(uVelocity, vL).x;
    float R = texture(uVelocity, vR).x;
    float T = texture(uVelocity, vT).y;
    float B = texture(uVelocity, vB).y;
    vec2 C = texture(uVelocity, vUv).xy;
    if (uReflectWalls > 0.5) {
      if (vL.x < 0.0) { L = -C.x; }
      if (vR.x > 1.0) { R = -C.x; }
      if (vT.y > 1.0) { T = -C.y; }
      if (vB.y < 0.0) { B = -C.y; }
    }
    float div = 0.5 * (R - L + T - B);
    o = vec4(div, 0.0, 0.0, 1.0);
  }`;

  const PRESSURE_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
  uniform sampler2D uPressure; uniform sampler2D uDivergence; out vec4 o;
  void main(){
    float L = texture(uPressure, vL).x;
    float R = texture(uPressure, vR).x;
    float T = texture(uPressure, vT).x;
    float B = texture(uPressure, vB).x;
    float divergence = texture(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    o = vec4(pressure, 0.0, 0.0, 1.0);
  }`;

  const GRADIENT_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
  uniform sampler2D uPressure; uniform sampler2D uVelocity; out vec4 o;
  void main(){
    float L = texture(uPressure, vL).x;
    float R = texture(uPressure, vR).x;
    float T = texture(uPressure, vT).x;
    float B = texture(uPressure, vB).x;
    vec2 velocity = texture(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    o = vec4(velocity, 0.0, 1.0);
  }`;

  const ADVECT_FS = `#version 300 es
  precision highp float;
  in vec2 vUv;
  uniform sampler2D uVelocity; uniform sampler2D uSource;
  uniform vec2 texelSize; uniform float dt; uniform float dissipation; uniform float uBFECC;
  out vec4 o;
  void main(){
    if (uBFECC < 0.5) {
      vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
      o = dissipation * texture(uSource, coord);
    } else {
      vec2 vel = texture(uVelocity, vUv).xy;
      vec2 spotOld = vUv - vel * dt * texelSize;
      vec2 velBack = texture(uVelocity, spotOld).xy;
      vec2 spotForward = spotOld + velBack * dt * texelSize;
      vec2 error = spotForward - vUv;
      vec2 spotMid = vUv - error * 0.5;
      vec2 velMid = texture(uVelocity, spotMid).xy;
      vec2 coord = spotMid - velMid * dt * texelSize;
      o = dissipation * texture(uSource, coord);
    }
    o.a = 1.0;
  }`;

  /* =======================================================================
     FluidSim — vanilla WebGL2 Stable-Fluids solver
     ======================================================================= */
  const FLUID_PROFILES = {
    performance: { sim: 96, dye: 192, iters: 6 },
    balanced: { sim: 160, dye: 320, iters: 12 },
    quality: { sim: 256, dye: 640, iters: 18 },
  };

  class FluidSim {
    constructor(gl, opts) {
      opts = opts || {};
      this.gl = gl;
      const prof = FLUID_PROFILES[opts.profile || 'balanced'];
      this.simRes = opts.simResolution || prof.sim;
      this.dyeRes = opts.dyeResolution || prof.dye;
      this.pressureIterations = opts.pressureIterations || prof.iters;
      this.densityDissipation = opts.densityDissipation ?? 0.93;
      this.velocityDissipation = opts.velocityDissipation ?? 0.985;
      this.pressureDissipation = opts.pressureDissipation ?? 0.8;
      this.curlStrength = opts.curlStrength ?? 0.4;
      this.splatRadius = opts.splatRadius ?? 0.0028;
      this.splatForce = opts.splatForce ?? 6;
      this.baseDelta = opts.baseDelta ?? 1 / 60;
      this.dyeDissipation = opts.dyeDissipation ?? this.densityDissipation;
      this.enableVorticity = opts.enableVorticity ?? false;
      this.bfecc = opts.bfecc ?? true;
      this.reflectWalls = opts.reflectWalls ?? true;
      this.enableDye = opts.enableDye ?? false;

      const A = { position: 0 };
      this.progs = {
        clear: program(gl, SIM_VS, CLEAR_FS, A, 'clear'),
        splat: program(gl, SPLAT_VS, SPLAT_FS, A, 'splat'),
        curl: program(gl, SIM_VS, CURL_FS, A, 'curl'),
        vorticity: program(gl, SIM_VS, VORTICITY_FS, A, 'vorticity'),
        divergence: program(gl, SIM_VS, DIVERGENCE_FS, A, 'divergence'),
        pressure: program(gl, SIM_VS, PRESSURE_FS, A, 'pressure'),
        gradient: program(gl, SIM_VS, GRADIENT_FS, A, 'gradient'),
        advect: program(gl, SIM_VS, ADVECT_FS, A, 'advect'),
      };
      this.ok = Object.values(this.progs).every(Boolean);
      if (!this.ok) return;

      // fullscreen triangle + splat quad geometry (location 0 = position)
      this.triVao = gl.createVertexArray();
      gl.bindVertexArray(this.triVao);
      const triBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, triBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

      this.quadVao = gl.createVertexArray();
      gl.bindVertexArray(this.quadVao);
      const quadBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      this.viewW = 1; this.viewH = 1;
      this.simW = this.simRes; this.simH = this.simRes;
      this.dyeW = this.dyeRes; this.dyeH = this.dyeRes;
      this._alloc();
      this.splats = [];
    }

    _alloc() {
      const gl = this.gl, L = gl.LINEAR, N = gl.NEAREST;
      const lin = { filter: L };
      const near = { filter: N };
      this.velocity = makeDouble(gl, this.simW, this.simH, lin);
      this.pressure = makeDouble(gl, this.simW, this.simH, near);
      this.divergence = makeFBO(gl, this.simW, this.simH, near);
      this.curl = makeFBO(gl, this.simW, this.simH, near);
      this.density = makeDouble(gl, this.dyeW, this.dyeH, lin);
      this.dye = makeDouble(gl, this.dyeW, this.dyeH, lin);
    }

    get velocityTexture() { return this.velocity.read.tex; }
    get densityTexture() { return this.density.read.tex; }
    get dyeTexture() { return this.dye.read.tex; }

    resize(w, h) {
      this.viewW = Math.max(1, w); this.viewH = Math.max(1, h);
      const aspect = this.viewW / this.viewH;
      let sw, sh, dw, dh;
      if (aspect >= 1) {
        sw = this.simRes; sh = Math.max(1, Math.round(this.simRes / aspect));
        dw = this.dyeRes; dh = Math.max(1, Math.round(this.dyeRes / aspect));
      } else {
        sw = Math.max(1, Math.round(this.simRes * aspect)); sh = this.simRes;
        dw = Math.max(1, Math.round(this.dyeRes * aspect)); dh = this.dyeRes;
      }
      if (sw === this.simW && sh === this.simH && dw === this.dyeW && dh === this.dyeH) return;
      this.simW = sw; this.simH = sh; this.dyeW = dw; this.dyeH = dh;
      this._dispose();
      this._alloc();
    }

    addSplat(x, y, dx, dy, options) {
      options = options || {};
      this.splats.push({
        x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)),
        dx, dy, radius: options.radius ?? this.splatRadius,
        color: options.color, dyeColor: options.dyeColor,
      });
    }

    _blit(target, prog) {
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.w, target.h);
      gl.bindVertexArray(this.triVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    _applySplat(s) {
      const gl = this.gl, aspect = this.viewW / this.viewH;
      const color = s.color || [s.dx, s.dy, 1];
      const halfSize = 3 * Math.sqrt(s.radius);
      const prog = this.progs.splat;
      gl.useProgram(prog.p);
      gl.uniform2f(prog.loc.uCenter, s.x * 2 - 1, s.y * 2 - 1);
      gl.uniform2f(prog.loc.uScale, halfSize / aspect, halfSize);
      gl.bindVertexArray(this.quadVao);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.uniform3f(prog.loc.color, color[0], color[1], color[2]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.read.fbo);
      gl.viewport(0, 0, this.simW, this.simH);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.density.read.fbo);
      gl.viewport(0, 0, this.dyeW, this.dyeH);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (this.enableDye && s.dyeColor) {
        gl.uniform3f(prog.loc.color, s.dyeColor[0], s.dyeColor[1], s.dyeColor[2]);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.dye.read.fbo);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      gl.disable(gl.BLEND);
    }

    step(dtSeconds) {
      if (!this.ok) return;
      const gl = this.gl;
      const dt = Math.min(Math.max(dtSeconds, 1e-6), 1 / 60);
      const dtScale = this.baseDelta > 0 ? dt / this.baseDelta : 1;
      const simTexel = [1 / this.simW, 1 / this.simH];
      const dyeTexel = [1 / this.dyeW, 1 / this.dyeH];
      const bfecc = this.bfecc ? 1 : 0;

      for (let i = 0; i < this.splats.length; i++) this._applySplat(this.splats[i]);
      this.splats.length = 0;

      gl.bindVertexArray(this.triVao);

      if (this.enableVorticity) {
        let pr = this.progs.curl;
        gl.useProgram(pr.p);
        gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
        this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 0);
        this._blit(this.curl, pr);

        pr = this.progs.vorticity;
        gl.useProgram(pr.p);
        gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
        gl.uniform1f(pr.loc.curl, this.curlStrength);
        gl.uniform1f(pr.loc.dt, dt);
        this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 0);
        this._bindTex(pr, 'uCurl', this.curl.tex, 1);
        this._blit(this.velocity.write, pr);
        swap(this.velocity);
      }

      let pr = this.progs.divergence;
      gl.useProgram(pr.p);
      gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
      gl.uniform1f(pr.loc.uReflectWalls, this.reflectWalls ? 1 : 0);
      this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 0);
      this._blit(this.divergence, pr);

      pr = this.progs.clear;
      gl.useProgram(pr.p);
      gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
      gl.uniform1f(pr.loc.value, Math.pow(this.pressureDissipation, dtScale));
      this._bindTex(pr, 'uTexture', this.pressure.read.tex, 0);
      this._blit(this.pressure.write, pr);
      swap(this.pressure);

      pr = this.progs.pressure;
      gl.useProgram(pr.p);
      gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
      this._bindTex(pr, 'uDivergence', this.divergence.tex, 1);
      for (let i = 0; i < this.pressureIterations; i++) {
        this._bindTex(pr, 'uPressure', this.pressure.read.tex, 0);
        this._blit(this.pressure.write, pr);
        swap(this.pressure);
      }

      pr = this.progs.gradient;
      gl.useProgram(pr.p);
      gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
      this._bindTex(pr, 'uPressure', this.pressure.read.tex, 0);
      this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 1);
      this._blit(this.velocity.write, pr);
      swap(this.velocity);

      pr = this.progs.advect;
      gl.useProgram(pr.p);
      gl.uniform2f(pr.loc.texelSize, simTexel[0], simTexel[1]);
      gl.uniform1f(pr.loc.dt, dt);
      gl.uniform1f(pr.loc.uBFECC, bfecc);
      gl.uniform1f(pr.loc.dissipation, Math.pow(this.velocityDissipation, dtScale));
      this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 0);
      this._bindTex(pr, 'uSource', this.velocity.read.tex, 1);
      this._blit(this.velocity.write, pr);
      swap(this.velocity);

      gl.uniform2f(pr.loc.texelSize, dyeTexel[0], dyeTexel[1]);
      gl.uniform1f(pr.loc.dissipation, Math.pow(this.densityDissipation, dtScale));
      this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 0);
      this._bindTex(pr, 'uSource', this.density.read.tex, 1);
      this._blit(this.density.write, pr);
      swap(this.density);

      if (this.enableDye) {
        gl.uniform1f(pr.loc.dissipation, Math.pow(this.dyeDissipation, dtScale));
        this._bindTex(pr, 'uVelocity', this.velocity.read.tex, 0);
        this._bindTex(pr, 'uSource', this.dye.read.tex, 1);
        this._blit(this.dye.write, pr);
        swap(this.dye);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _bindTex(prog, name, tex, unit) {
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (prog.loc[name]) gl.uniform1i(prog.loc[name], unit);
    }

    _dispose() {
      const gl = this.gl;
      [this.velocity, this.pressure, this.density, this.dye].forEach(d => {
        if (!d) return;
        gl.deleteTexture(d.read.tex); gl.deleteFramebuffer(d.read.fbo);
        gl.deleteTexture(d.write.tex); gl.deleteFramebuffer(d.write.fbo);
      });
      [this.divergence, this.curl].forEach(f => {
        if (!f) return; gl.deleteTexture(f.tex); gl.deleteFramebuffer(f.fbo);
      });
    }

    dispose() {
      this._dispose();
      const gl = this.gl;
      if (this.progs) for (const k in this.progs) { if (this.progs[k]) gl.deleteProgram(this.progs[k].p); }
    }
  }

  /* =======================================================================
     Overlay passes — render fluid textures to the screen with a visual style.
     `uBg` stands in for the (here flat, dark) background scene.
     ======================================================================= */
  const OVERLAY_COMMON = `
  vec3 vibrant(vec3 c, float v) {
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    return clamp(mix(vec3(lum), c, 1.0 + v), 0.0, 1.0);
  }
  vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
  }
  vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y); float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
  }`;

  /* Procedural backdrop. The three-fluid-fx passes are post-process effects
     that composite over a rendered 3D scene; here there is none, so each card
     synthesises one. `uScene` picks gradient (0) / shapes (1) / torus (2) —
     mirroring the reference example's scene types — coloured by the card's
     Base/Mid/Highlight palette. Distortion passes refract this; overlays glow
     over it. A faint fbm keeps texture everywhere so refraction always reads. */
  const SCENE_GLSL = `
  uniform vec3 uColA; uniform vec3 uColB; uniform vec3 uColC;
  uniform int  uScene; uniform vec2 uRes;
  float sHash(vec2 p){ p = fract(p*vec2(123.34,345.45)); p += dot(p,p+34.345); return fract(p.x*p.y); }
  float sNoise(vec2 p){
    vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
    float a = sHash(i), b = sHash(i+vec2(1,0)), c = sHash(i+vec2(0,1)), d = sHash(i+vec2(1,1));
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  }
  float sFbm(vec2 p){ float v = 0.0, a = 0.5; for(int i=0;i<5;i++){ v += a*sNoise(p); p *= 1.9; a *= 0.5; } return v; }
  vec3 sceneColor(vec2 uv){
    if (uScene == 3) return texture(uSceneTex, uv).rgb;   // rasterised wordmark (gradient text on black)
    vec2 p = uv*2.0 - 1.0; p.x *= uRes.x / max(uRes.y, 1.0);
    float n = sFbm(uv*4.0 + uTime*0.04);
    vec3 base = mix(uBg, uBg*1.7 + 0.015, uv.y) + (n - 0.5)*0.06;
    if (uScene == 1) {            // SHAPES — soft drifting orbs
      vec3 col = base;
      for (int i=0;i<5;i++){
        float fi = float(i);
        vec2 c = 0.75*vec2(sin(uTime*0.17 + fi*1.7), cos(uTime*0.14 + fi*2.3));
        vec3 cc = i < 2 ? uColA : (i < 4 ? uColB : uColC);
        col += cc * smoothstep(0.7, 0.0, length(p - c)) * 0.30;
      }
      return col;
    }
    if (uScene == 2) {            // TORUS — concentric refractive rings
      float r = length(p);
      float rings = 0.5 + 0.5*sin(r*15.0 - uTime*0.5);
      float halo = smoothstep(1.15, 0.15, r);
      vec3 col = mix(uColA, uColB, rings);
      col = mix(base, col, halo*0.55);
      return col + uColC*pow(rings, 3.0)*halo*0.45;
    }
    float g = 0.5 + 0.5*sin(uv.x*3.2 - uv.y*2.1 + uTime*0.22);  // GRADIENT
    vec3 col = mix(uColA, uColB, uv.x);
    col = mix(col, uColC, uv.y*0.65);
    return mix(base, col, 0.20 + 0.10*g);
  }`;

  /* Shared preamble for every fluid effect shader: all three fluid textures
     (`tFluid` rg=velocity b=density, `tDye`, `tVelocity`), the common tunables,
     the colour helpers and the procedural scene. Unused declarations are
     optimised out (program() only records active uniforms), so each effect just
     samples what it needs and composites via `mix(scene, eff, uOpacity)`. */
  const FX_HEAD = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 o;
  uniform sampler2D tFluid;
  uniform sampler2D tDye;
  uniform sampler2D tVelocity;
  uniform sampler2D uSceneTex;
  uniform vec3  uBg;
  uniform float uTime;
  uniform float uIntensity;
  uniform float uVibrance;
  uniform float uOpacity;
  uniform vec2  uTexel;
  uniform vec3  uCursorColor;
  uniform vec3  uTint;
  ${OVERLAY_COMMON}
  ${SCENE_GLSL}`;

  const COLORFUL_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec2 vel = texture(tFluid, vUv).rg;
    float glow = 0.0; vec3 color = vec3(0.0);
    for (float i = 0.0; i < 6.0; i += 1.0) {
      vec2 origin = vUv - vel * i * 0.035;
      float d = texture(tFluid, origin).b;
      float w = (1.0 - i / 7.0) * d;
      glow += w;
      float hueA = origin.x * 1.6 + origin.y * 0.9 + uTime * 0.05;
      float hueB = origin.y * 1.2 - origin.x * 0.4 - uTime * 0.03;
      color += mix(hsv2rgb(vec3(fract(hueA), 0.9, 1.0)), hsv2rgb(vec3(fract(hueB), 0.85, 0.95)), 0.5) * w;
    }
    if (glow > 0.0) color /= glow;
    color = vibrant(color, uVibrance);
    vec3 eff = scene + color * clamp(glow * uIntensity * 0.55, 0.0, 1.4);
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const RAINBOWINK_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    float amp = length(dye);
    if (amp < 1e-4) { o = vec4(scene, 1.0); return; }
    float baseHue = rgb2hsv(dye / amp).x;
    float depth = pow(clamp(amp * 2.5, 0.0, 1.0), 0.7);
    float hue = fract(baseHue + (1.0 - depth) * (0.32 + sin(baseHue * 6.28318 * 3.0) * 0.13));
    float sat = mix(0.75, 1.0, depth);
    vec3 col = vibrant(hsv2rgb(vec3(hue, sat, 1.0)), uVibrance);
    vec3 eff = scene + col * depth * uIntensity * 1.2;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const VOLUME_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    float dL = length(texture(tDye, vUv - vec2(uTexel.x * 2.0, 0.0)).rgb);
    float dR = length(texture(tDye, vUv + vec2(uTexel.x * 2.0, 0.0)).rgb);
    float dD = length(texture(tDye, vUv - vec2(0.0, uTexel.y * 2.0)).rgb);
    float dU = length(texture(tDye, vUv + vec2(0.0, uTexel.y * 2.0)).rgb);
    vec2 grad = vec2(dR - dL, dU - dD);
    float gmag = length(grad);
    vec2 ndir = grad / max(gmag, 1e-5);
    float lit = dot(ndir, normalize(vec2(-0.6, 0.8)));
    float strength = smoothstep(0.0, 0.04, gmag);
    float shade = mix(1.0, mix(0.78, 1.0, lit * 0.5 + 0.5), strength);
    float density = clamp(length(dye) * uIntensity * 3.0, 0.0, 0.95);
    vec3 tint = vibrant(uCursorColor, uVibrance) * shade;
    vec3 eff = mix(scene, tint, density);
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const DENSITYTINT_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 o;
  uniform sampler2D tFluid; uniform vec3 uTint; uniform float uIntensity;
  void main(){
    float density = clamp(texture(tFluid, vUv).b, 0.0, 1.0);
    o = vec4(uTint * density * uIntensity, 1.0);
  }`;

  /* ===================================================================== */
  /* three-fluid-fx effect passes, ported to single-pass WebGL2 overlays.  */
  /* Each composites over sceneColor() and fades via uOpacity. Faithful to */
  /* the GLSL in three-fluid-fx (distortion/ + overlay/), with tDiffuse    */
  /* swapped for the procedural scene.                                     */
  /* ===================================================================== */

  /* ---- distortion: warp / refract the scene by the fluid field ---- */
  const SIMPLE_FS = FX_HEAD + `
  void main(){
    vec2 vel = texture(tFluid, vUv).rg;
    vec2 uv = clamp(vUv - vel * uIntensity * 0.0003, 0.0, 1.0);
    o = vec4(mix(sceneColor(vUv), sceneColor(uv), uOpacity), 1.0);
  }`;

  const RGBSHIFT_FS = FX_HEAD + `
  void main(){
    vec3 fl = texture(tFluid, vUv).rgb;
    float density = clamp(fl.b, 0.0, 1.0);
    vec2 vel = fl.rg;
    vec2 dir = vel / max(length(vel), 1e-4);
    vec2 shift = dir * pow(density, 1.4) * uIntensity * 0.012;
    vec3 eff = vec3(sceneColor(vUv + shift).r, sceneColor(vUv).g, sceneColor(vUv - shift).b);
    o = vec4(mix(sceneColor(vUv), eff, uOpacity), 1.0);
  }`;

  const CHROMATIC_FS = FX_HEAD + `
  void main(){
    vec3 fl = texture(tFluid, vUv).rgb * 0.36;
    fl += texture(tFluid, vUv + vec2(uTexel.x*2.0, 0.0)).rgb * 0.16;
    fl += texture(tFluid, vUv - vec2(uTexel.x*2.0, 0.0)).rgb * 0.16;
    fl += texture(tFluid, vUv + vec2(0.0, uTexel.y*2.0)).rgb * 0.16;
    fl += texture(tFluid, vUv - vec2(0.0, uTexel.y*2.0)).rgb * 0.16;
    vec2 vel = fl.rg;
    float falloff = pow(clamp(fl.b, 0.0, 1.0), 1.2);
    vec2 chroma = vel * 0.003 * uIntensity * falloff;
    vec2 d = vUv - vel * 0.0002 * uIntensity * falloff;
    vec3 eff = vec3(
      sceneColor(d + vec2( chroma.x,  chroma.y)).r,
      sceneColor(d + vec2(-chroma.x,  chroma.y)).g,
      sceneColor(d + vec2(-chroma.x, -chroma.y)).b);
    o = vec4(mix(sceneColor(vUv), eff, uOpacity), 1.0);
  }`;

  const WATER_FS = FX_HEAD + `
  void main(){
    float hL = texture(tFluid, vUv - vec2(uTexel.x*2.0, 0.0)).b;
    float hR = texture(tFluid, vUv + vec2(uTexel.x*2.0, 0.0)).b;
    float hD = texture(tFluid, vUv - vec2(0.0, uTexel.y*2.0)).b;
    float hU = texture(tFluid, vUv + vec2(0.0, uTexel.y*2.0)).b;
    vec2 off = vec2(hR - hL, hU - hD) * uIntensity * 0.6;
    vec3 eff = vec3(sceneColor(vUv + off*0.95).r, sceneColor(vUv + off).g, sceneColor(vUv + off*1.05).b);
    o = vec4(mix(sceneColor(vUv), eff, uOpacity), 1.0);
  }`;

  const WATERCAUSTICS_FS = FX_HEAD + `
  float causticWeb(vec2 uv, float t){
    const float TAU = 6.28318530718;
    vec2 p = mod(uv*TAU, TAU) - 250.0;
    vec2 i = p; float c = 1.0; float inten = 0.005;
    for (int n = 0; n < 5; n++) {
      float tt = t * (1.0 - 3.5 / float(n + 1));
      i = p + vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
      c += 1.0 / length(vec2(p.x / (sin(i.x + tt) / inten), p.y / (cos(i.y + tt) / inten)));
    }
    c /= 5.0; c = 1.17 - pow(c, 1.4);
    return clamp(pow(abs(c), 8.0), 0.0, 1.0);
  }
  void main(){
    vec3 fl = texture(tFluid, vUv).rgb;
    float hC = fl.b; vec2 vel = fl.rg;
    float hL = texture(tFluid, vUv - vec2(uTexel.x*2.0, 0.0)).b;
    float hR = texture(tFluid, vUv + vec2(uTexel.x*2.0, 0.0)).b;
    float hD = texture(tFluid, vUv - vec2(0.0, uTexel.y*2.0)).b;
    float hU = texture(tFluid, vUv + vec2(0.0, uTexel.y*2.0)).b;
    vec2 normal = vec2(hR - hL, hU - hD);
    vec2 off = normal * uIntensity * 0.6;
    vec3 refr = vec3(sceneColor(vUv + off*0.95).r, sceneColor(vUv + off).g, sceneColor(vUv + off*1.05).b);
    float surface = smoothstep(0.015, 0.16, hC);
    float slope = smoothstep(0.0015, 0.04, length(normal));
    float web = causticWeb(vUv*4.0 + vel*0.0012, uTime*0.5 + 23.0);
    vec3 caustic = clamp(vec3(web) + vec3(0.0, 0.35, 0.5), 0.0, 1.0);
    float energy = pow(web, 1.25) * surface * mix(0.4, 1.0, slope);
    vec3 eff = refr + caustic * energy * uIntensity * 0.38;
    o = vec4(mix(sceneColor(vUv), eff, uOpacity), 1.0);
  }`;

  /* ---- overlay: composite a fluid-coloured layer over the scene ---- */
  const DEFAULT_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    float far = 0.0;
    far += length(texture(tDye, vUv + uTexel * vec2( 8.0,  0.0)).rgb);
    far += length(texture(tDye, vUv + uTexel * vec2(-8.0,  0.0)).rgb);
    far += length(texture(tDye, vUv + uTexel * vec2( 0.0,  8.0)).rgb);
    far += length(texture(tDye, vUv + uTexel * vec2( 0.0, -8.0)).rgb);
    far *= 0.25;
    float core = smoothstep(0.02, 0.55, far * uIntensity * 4.0);
    vec2 vel = texture(tFluid, vUv).rg;
    float kinetic = clamp(length(vel) * 0.02, 0.0, 1.0);
    vec3 hsv = rgb2hsv(uCursorColor);
    float sat = clamp(hsv.y * mix(0.20, 1.0, core) + kinetic * hsv.y * 0.35, 0.0, 1.0);
    float val = hsv.z * mix(0.78, 1.0, core);
    vec3 tint = vibrant(hsv2rgb(vec3(hsv.x, sat, val)), uVibrance);
    float density = clamp(length(dye) * uIntensity * 3.0, 0.0, 0.95);
    vec3 eff = mix(scene, tint, density);
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const TRAIL_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 fluid = texture(tFluid, vUv).rgb;
    vec2 vel = fluid.rg;
    float here = clamp(fluid.b, 0.0, 1.0);
    float tail = 0.0, wsum = 0.0;
    for (float i = 1.0; i < 8.0; i += 1.0) {
      float w = 1.0 - i / 8.0;
      tail += texture(tFluid, vUv - vel * i * 0.04).b * w;
      wsum += w;
    }
    tail /= wsum;
    float glow = (tail * 0.7 + pow(here, 4.0) * 1.4) * uIntensity;
    vec3 eff = scene + vibrant(uCursorColor, uVibrance) * glow;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const OIL_FS = FX_HEAD + `
  vec3 oilPalette(float t){
    vec3 ember = vec3(1.0, 0.33, 0.20);
    vec3 mint  = vec3(0.08, 0.78, 0.68);
    vec3 cream = vec3(1.0, 0.84, 0.55);
    return mix(mix(ember, cream, smoothstep(0.15, 0.85, t)), mint, smoothstep(0.55, 1.0, t) * 0.42);
  }
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 fluid = texture(tFluid, vUv).rgb;
    float speed = length(fluid.rg);
    float trail = clamp(fluid.b, 0.0, 1.0);
    for (float i = 1.0; i < 6.0; i += 1.0)
      trail += texture(tFluid, vUv - fluid.rg * i * 0.035).b * (1.0 - i / 7.0);
    float glow = clamp(trail * uIntensity, 0.0, 1.0);
    vec3 color = vibrant(oilPalette(fract(glow * 0.62 + speed * 0.015 + uTime * 0.025)), uVibrance);
    float alpha = clamp(glow * 0.58 + speed * 0.012, 0.0, 0.86);
    vec3 eff = mix(scene + color * alpha * 0.86, color, alpha * 0.14);
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const VELOCITY_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec2 vel = texture(tVelocity, vUv).xy * 0.04 * uIntensity;
    float len = clamp(length(vel), 0.0, 1.0);
    vel = vel * 1.5 + 0.1;
    vec3 eff = scene + vec3(vel.x, vel.y, 1.0) * len;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const RAINBOWFISH_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec2 vel = texture(tVelocity, vUv).xy * 0.04;
    float speed = length(vel);
    float hueA = atan(vel.y, vel.x) / 6.28318 + 0.5 + uTime * 0.05;
    float hueB = vUv.x * 1.2 + vUv.y * 0.8 + uTime * 0.04;
    vec3 color = vibrant(mix(hsv2rgb(vec3(fract(hueA), 0.92, 1.0)), hsv2rgb(vec3(fract(hueB), 0.7, 0.95)), 0.35), uVibrance);
    vec3 eff = scene + color * pow(clamp(speed * 8.0, 0.0, 1.0), 2.5) * 1.6 * uIntensity;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const GLAZE_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    float density = clamp(texture(tFluid, vUv).b, 0.0, 1.0);
    vec3 eff = scene + vibrant(vec3(1.0, 0.45, 0.22), uVibrance) * density * uIntensity;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const BURN_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec2 vel = texture(tFluid, vUv).rg;
    float fingers = 0.0;
    for (float i = 0.0; i < 5.0; i += 1.0)
      fingers += texture(tFluid, vUv - vel * (i + 1.0) * 0.05).b * (1.0 - i / 5.0);
    fingers *= uIntensity;
    vec3 burnColor = vec3(1.0, 0.3, 0.0);
    vec3 ghost = mix(vec3(0.8, 0.15, 0.0), burnColor, clamp(fingers, 0.0, 1.0));
    ghost += burnColor * pow(clamp(fingers, 0.0, 1.0), 2.0) * 2.0;
    float smoke = fingers * 0.3;
    vec3 fire = ghost + vec3(0.1, 0.1, 0.15) * smoke;
    float flicker = 0.8 + 0.2 * sin(uTime * 15.0 + fingers * 20.0);
    fire *= flicker;
    float alpha = clamp(fingers * 0.5 * flicker + smoke * 0.2, 0.0, 0.85);
    vec3 eff = mix(scene, vibrant(fire, uVibrance), alpha);
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const SMOKE_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    float density = clamp(length(dye) * uIntensity * 3.0, 0.0, 0.95);
    vec3 eff = mix(scene, vec3(0.95, 0.97, 1.0), density);
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const ARTINK_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    float amp = length(dye);
    vec3 boosted = amp > 1e-5 ? vibrant(dye / amp, uVibrance) * amp : dye;
    vec3 eff = scene + boosted * uIntensity * 3.0;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const COLORWATER_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    float density = length(dye);
    vec3 hue = density > 1e-4 ? vibrant(dye / density, uVibrance) : vec3(1.0);
    float alpha = (1.0 - exp(-density * uIntensity * 3.0)) * 0.72;
    vec3 eff = mix(scene, hue * 1.1, alpha) + scene * hue * alpha * 0.35;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  const LIQUIDLENS_FS = FX_HEAD + `
  void main(){
    vec3 dye = texture(tDye, vUv).rgb * 0.5;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0,  1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2( 1.0, -1.0)).rgb * 0.125;
    dye += texture(tDye, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.125;
    vec2 vel = texture(tVelocity, vUv).xy * 0.04;
    float density = length(dye);
    float gate = clamp(density * 4.0, 0.0, 1.0);
    vec3 scene = sceneColor(vUv + vel * gate * 0.012);
    vec3 boosted = density > 1e-5 ? vibrant(dye / density, uVibrance) * density : dye;
    vec3 tint = min(boosted * uIntensity * 1.4, vec3(1.6));
    vec3 eff = scene + scene * tint;
    o = vec4(mix(sceneColor(vUv), eff, uOpacity), 1.0);
  }`;

  /* card-facing density tint (composites over the scene); the bare
     DENSITYTINT_FS above stays the additive tint ParticleView blends in. */
  const DTINT_FS = FX_HEAD + `
  void main(){
    vec3 scene = sceneColor(vUv);
    float density = clamp(texture(tFluid, vUv).b, 0.0, 1.0);
    vec3 eff = scene + uTint * density * uIntensity;
    o = vec4(mix(scene, eff, uOpacity), 1.0);
  }`;

  /* overlay registry. `dye` enables the dye FBO + coloured pointer strokes;
     `category` drives the modal's source view + the gallery grouping. The
     render path binds tFluid→0, tDye→1, tVelocity→2 and sets whichever
     uniforms the program actually compiled. */
  const OVERLAYS = {
    // ---- overlay ----
    colorful:    { fs: COLORFUL_FS,    dye: false, category: 'overlay' },
    rainbowink:  { fs: RAINBOWINK_FS,  dye: true,  category: 'overlay' },
    volume:      { fs: VOLUME_FS,      dye: true,  category: 'overlay' },
    default:     { fs: DEFAULT_FS,     dye: true,  category: 'overlay' },
    trail:       { fs: TRAIL_FS,       dye: false, category: 'overlay' },
    oil:         { fs: OIL_FS,         dye: false, category: 'overlay' },
    velocity:    { fs: VELOCITY_FS,    dye: false, category: 'overlay' },
    rainbowfish: { fs: RAINBOWFISH_FS, dye: false, category: 'overlay' },
    glaze:       { fs: GLAZE_FS,       dye: false, category: 'overlay' },
    burn:        { fs: BURN_FS,        dye: false, category: 'overlay' },
    smoke:       { fs: SMOKE_FS,       dye: true,  category: 'overlay' },
    artink:      { fs: ARTINK_FS,      dye: true,  category: 'overlay' },
    colorwater:  { fs: COLORWATER_FS,  dye: true,  category: 'overlay' },
    liquidlens:  { fs: LIQUIDLENS_FS,  dye: true,  category: 'overlay' },
    dtint:       { fs: DTINT_FS,       dye: false, category: 'overlay' },
    // ---- distortion ----
    simple:        { fs: SIMPLE_FS,        dye: false, category: 'distortion' },
    rgbshift:      { fs: RGBSHIFT_FS,      dye: false, category: 'distortion' },
    chromatic:     { fs: CHROMATIC_FS,     dye: false, category: 'distortion' },
    water:         { fs: WATER_FS,         dye: false, category: 'distortion' },
    watercaustics: { fs: WATERCAUSTICS_FS, dye: false, category: 'distortion' },
  };

  /* Backdrop — FBM height field lit by two orbiting lights (warm + cool), the
     low-poly gradient behind the "Mega" demo. Ported from examples Backdrop. */
  const BACKDROP_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 o;
  uniform float uTime; uniform vec2 uRes;
  uniform vec3 uBase; uniform vec3 uWarm; uniform vec3 uCool;
  float h21(vec2 p){ p=fract(p*vec2(123.34,345.45)); p+=dot(p,p+34.345); return fract(p.x*p.y); }
  float vnoise(vec2 p){ vec2 i=floor(p),f=fract(p); vec2 u=f*f*(3.0-2.0*f);
    float a=h21(i),b=h21(i+vec2(1,0)),c=h21(i+vec2(0,1)),d=h21(i+vec2(1,1));
    return mix(mix(a,b,u.x),mix(c,d,u.x),u.y); }
  float fbm(vec2 p){ float v=0.0,a=0.5; mat2 m=mat2(1.6,1.2,-1.2,1.6);
    for(int i=0;i<5;i++){ v+=a*vnoise(p); p=m*p; a*=0.5; } return v; }
  void main(){
    vec2 asp = vec2(uRes.x/uRes.y, 1.0);
    vec2 p = (vUv*2.0-1.0)*asp;
    float ang = uTime*0.15;
    vec2 la = vec2(cos(ang), sin(ang))*1.0;            // warm light
    vec2 lb = -la;                                      // cool light, 180° apart
    float n = fbm(vUv*3.2 + uTime*0.05);
    float facet = fbm(vUv*7.0 - uTime*0.03);
    float da = smoothstep(2.0, 0.0, length(p - la));
    float db = smoothstep(2.0, 0.0, length(p - lb));
    vec3 col = uBase;
    col += uWarm * da * (0.55 + 0.6*n);
    col += uCool * db * (0.55 + 0.6*n);
    col *= 0.85 + 0.3*facet;                            // faint low-poly shimmer
    col += 0.015;
    o = vec4(col, 1.0);
  }`;

  /* Fluid Text — the rendered-text texture is refracted by the density
     gradient (density-as-height, per-channel chromatic split) and rainbow dye
     ink is added on top. Mirrors the fluid-text example (distortion → overlay). */
  const FLUIDTEXT_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 o;
  uniform sampler2D uText; uniform sampler2D tFluid; uniform sampler2D tDye;
  uniform vec2 uTexel; uniform float uIntensity; uniform float uDistort; uniform vec3 uBg; uniform float uVibrance;
  ${OVERLAY_COMMON}
  void main(){
    float hL = texture(tFluid, vUv - vec2(uTexel.x*2.0,0.0)).b;
    float hR = texture(tFluid, vUv + vec2(uTexel.x*2.0,0.0)).b;
    float hD = texture(tFluid, vUv - vec2(0.0,uTexel.y*2.0)).b;
    float hU = texture(tFluid, vUv + vec2(0.0,uTexel.y*2.0)).b;
    vec2 normal = vec2(hR-hL, hU-hD);
    vec2 off = normal * uDistort * 0.6;
    vec4 tr = texture(uText, vUv + off*0.95);
    vec4 tg = texture(uText, vUv + off);
    vec4 tb = texture(uText, vUv + off*1.05);
    vec3 textCol = vec3(tr.r, tg.g, tb.b);
    float textA = tg.a;
    vec3 scene = mix(uBg, textCol, textA);
    vec3 dye = texture(tDye, vUv).rgb;
    float amp = length(dye);
    if (amp > 1e-4) {
      float baseHue = rgb2hsv(dye/amp).x;
      float depth = pow(clamp(amp*2.5,0.0,1.0),0.7);
      float hue = fract(baseHue + (1.0-depth)*0.32);
      vec3 ink = vibrant(hsv2rgb(vec3(hue, mix(0.75,1.0,depth), 1.0)), uVibrance);
      scene += ink * depth * uIntensity;
    }
    o = vec4(scene, 1.0);
  }`;

  /* =======================================================================
     Particle shaders (GPGPU) — ported from flowParticles.ts
     ======================================================================= */
  const P_VELOCITY_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 o;
  uniform sampler2D uPositionTexture; uniform sampler2D uVelocityTexture;
  uniform sampler2D uDestinationTexture; uniform sampler2D uAttributeTexture; uniform sampler2D uFlow;
  uniform mat4 uViewMatrix; uniform mat4 uProjectionMatrix; uniform mat4 uModelMatrix;
  uniform vec3 uCameraRight; uniform vec3 uCameraUp;
  uniform float uDeltaTime, uFlowStrength, uDepthLift, uMaxFlowSpeed, uFlowThresh, uFlowPow,
    uPerpendicularAngle, uOmega, uZeta, uDragLin, uDragQuad, uAMax, uVMaxScale, uSideVariation,
    uPlaneLock, uDepthScale;
  float hash(float n){ return fract(sin(n) * 43758.5453123); }
  void main(){
    vec4 position = texture(uPositionTexture, vUv);
    vec4 velocity = texture(uVelocityTexture, vUv);
    vec4 destination = texture(uDestinationTexture, vUv);
    vec4 attr = texture(uAttributeTexture, vUv);
    vec3 pos = position.xyz; vec3 vel = velocity.xyz; vec3 dest = destination.xyz;
    float stiffness = destination.w; float vmax = attr.y * uVMaxScale; float seed = attr.w;
    vec3 error = dest - pos;
    float omega = uOmega * max(0.0, stiffness);
    vec3 aSpring = omega * omega * error;
    vec3 aDamp = -2.0 * uZeta * omega * vel;
    float speed = length(vel);
    vec3 aDrag = vec3(0.0);
    if (speed > 1e-5) aDrag = -uDragLin * vel - uDragQuad * speed * vel;
    vec3 aCore = aSpring + aDamp + aDrag;
    vec3 aFlow = vec3(0.0);
    vec3 worldPos = (uModelMatrix * vec4(pos, 1.0)).xyz;
    mat3 invModelRotation = inverse(mat3(uModelMatrix));
    vec4 clip = uProjectionMatrix * uViewMatrix * vec4(worldPos, 1.0);
    if (clip.w > 0.00001) {
      vec2 ndc = clip.xy / clip.w;
      vec2 uv = ndc * 0.5 + 0.5;
      if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
        vec2 flow = texture(uFlow, uv).xy;
        float flowMag = length(flow);
        float norm = (flowMag - uFlowThresh) / max(1e-5, uMaxFlowSpeed);
        float factor = smoothstep(0.0, 1.0, clamp(norm, 0.0, 1.0));
        factor = pow(factor, max(1.0, uFlowPow));
        flow *= factor;
        flow *= min(1.0, uMaxFlowSpeed / max(flowMag, 1e-5));
        vec3 flowWorld = flow.x * uCameraRight + flow.y * uCameraUp;
        vec3 flowLocal = invModelRotation * flowWorld;
        aFlow += flowLocal * uFlowStrength;
        if (uDepthLift > 0.0 && length(flowLocal) > 1e-5) {
          vec3 forward = normalize(cross(uCameraRight, uCameraUp));
          vec3 forwardLocal = invModelRotation * forward;
          vec3 flowDir = normalize(flowLocal);
          vec3 sideDir = normalize(cross(forwardLocal, flowDir));
          float sideSign = hash(seed * 12.9898) > 0.5 ? 1.0 : -1.0;
          float perSeed = mix(1.0, mix(0.35, 1.0, hash(seed * 37.719)), clamp(uSideVariation, 0.0, 1.0));
          aFlow += sideDir * sideSign * perSeed * length(flow) * uPerpendicularAngle * uDepthLift;
          aFlow += forwardLocal * (hash(seed * 91.17) - 0.5) * length(flow) * 0.18 * uDepthLift;
        }
        vec3 forwardW = normalize(cross(uCameraRight, uCameraUp));
        float signedDepth = dot(worldPos, forwardW);
        float behind = max(0.0, -signedDepth) / max(uDepthScale, 0.01);
        aFlow *= exp(-behind * behind);
      }
    }
    vec3 acceleration = aCore + aFlow;
    acceleration.z = mix(acceleration.z, aCore.z, uPlaneLock);
    float aMag = length(acceleration);
    if (aMag > uAMax) { acceleration = acceleration / aMag * uAMax; aMag = uAMax; }
    vel += acceleration * uDeltaTime;
    vel.z = mix(vel.z, 0.0, uPlaneLock);
    float newSpeed = length(vel);
    if (newSpeed > vmax) { vel = vel / newSpeed * vmax; newSpeed = vmax; }
    vec3 velCore = velocity.xyz + aCore * uDeltaTime;
    float flowEnergy = length(vel - velCore);
    float desiredEnergy = smoothstep(0.15, 2.8, newSpeed) * 0.35 + smoothstep(0.05, 1.8, flowEnergy) * 1.35;
    float alpha = 1.0 - pow(0.5, uDeltaTime / 0.08);
    float energy = mix(velocity.w, desiredEnergy, alpha);
    o = vec4(vel, energy);
  }`;

  const P_POSITION_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 o;
  uniform sampler2D uPositionTexture; uniform sampler2D uVelocityTexture; uniform sampler2D uDestinationTexture;
  uniform float uDeltaTime; uniform float uPlaneLock;
  void main(){
    vec4 pos = texture(uPositionTexture, vUv);
    vec3 vel = texture(uVelocityTexture, vUv).xyz;
    vec3 dest = texture(uDestinationTexture, vUv).xyz;
    pos.xyz += vel * uDeltaTime;
    pos.z = mix(pos.z, dest.z, uPlaneLock);
    o = vec4(pos.xyz, 1.0);
  }`;

  const P_RENDER_VS = `#version 300 es
  precision highp float;
  in vec3 position; in vec2 uv; in vec2 aParticleUv; in float aSeed;
  uniform sampler2D uPositionTexture; uniform sampler2D uVelocityTexture; uniform sampler2D uAttributeTexture;
  uniform float uPointSize; uniform float uTime;
  uniform vec3 uCameraRightLocal; uniform vec3 uCameraUpLocal;
  uniform mat4 uProjectionMatrix; uniform mat4 uModelViewMatrix;
  out vec2 vUv; out vec3 vParticleColor; out vec3 vParticlePalette;
  const float BILLBOARD = 0.006;
  float hash31(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123); }
  float valueNoise(vec3 p){
    vec3 i = floor(p); vec3 f = fract(p); vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i); float n100 = hash31(i + vec3(1,0,0));
    float n010 = hash31(i + vec3(0,1,0)); float n110 = hash31(i + vec3(1,1,0));
    float n001 = hash31(i + vec3(0,0,1)); float n101 = hash31(i + vec3(1,0,1));
    float n011 = hash31(i + vec3(0,1,1)); float n111 = hash31(i + vec3(1,1,1));
    float nx00 = mix(n000, n100, u.x); float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x); float nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
  }
  void main(){
    vec3 pos = texture(uPositionTexture, aParticleUv).xyz;
    float energy = clamp(texture(uVelocityTexture, aParticleUv).w, 0.0, 1.0);
    vec4 attr = texture(uAttributeTexture, aParticleUv);
    float worldSize = uPointSize * attr.x * BILLBOARD;
    vec3 offset = (uCameraRightLocal * position.x + uCameraUpLocal * position.y) * worldSize;
    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(pos + offset, 1.0);
    vUv = uv;
    float e = smoothstep(0.0, 1.0, energy);
    vec3 patternPos = pos * 0.72;
    vec3 noisePos = patternPos * 1.15 + vec3(aSeed * 7.1, uTime * 0.08, -uTime * 0.05);
    float n0 = valueNoise(noisePos);
    float n1 = valueNoise(noisePos * 2.31 + vec3(13.5, 9.2, 5.7));
    float noise = n0 * 0.68 + n1 * 0.32;
    float marble = sin((patternPos.x + patternPos.y * 0.7 - patternPos.z * 0.4) * 2.4 + noise * 4.5 + uTime * 0.05);
    float paletteDrift = sin(uTime * 0.11 + aSeed * 6.28318530718) * 0.025;
    float a = fract(aSeed * 0.18 + noise * 0.48 + marble * 0.12 + e * 0.26 + paletteDrift);
    float phase = a * 6.3;
    vec3 palette = vec3(cos(phase), cos(phase + 83.0), cos(phase + 21.0)) * 0.56 + 0.55;
    float cyanAmount = smoothstep(0.62, 1.05, palette.z + palette.y * 0.55 - palette.x * 0.7);
    float whiteAmount = smoothstep(0.72, 1.0, min(min(palette.x, palette.y), palette.z));
    vec3 paletteWarm = palette * vec3(1.08, 0.93, 0.72) + vec3(0.025, 0.0, 0.0);
    palette = mix(palette, paletteWarm, min(0.5, cyanAmount * 0.28 + whiteAmount * 0.35));
    float emissionStrength = 0.9 + e * 1.45;
    vParticlePalette = palette;
    vParticleColor = palette * emissionStrength;
  }`;

  const P_RENDER_FS = `#version 300 es
  precision highp float;
  in vec2 vUv; in vec3 vParticleColor; in vec3 vParticlePalette; out vec4 o;
  const float RADIUS = 0.5;
  const float EDGE_AA = 0.012;
  const vec2 FOCUS = vec2(-0.12, 0.14);
  const float FALLOFF = 0.68;
  const float LIGHT_BOOST = 0.28;
  const float CHROMA_BOOST = 0.34;
  const float ALPHA_BOOST = 0.18;
  const float COLOR_PEAK = 1.35;
  const float ALPHA_CUTOFF = 0.04;
  void main(){
    vec2 p = vUv - 0.5; float d = length(p);
    float aa = max(fwidth(d), EDGE_AA);
    float alpha = 1.0 - smoothstep(RADIUS - aa, RADIUS, d);
    vec2 gradientP = p - FOCUS;
    float centerGradient = 1.0 - smoothstep(0.0, FALLOFF, length(gradientP) / RADIUS);
    vec2 normalUv = gradientP / RADIUS;
    float normalZ = sqrt(max(0.0, 1.0 - dot(normalUv, normalUv)));
    vec3 normal = normalize(vec3(normalUv, normalZ));
    vec3 lightDir = normalize(vec3(-0.42, 0.55, 0.72));
    vec3 halfDir = normalize(vec3(-0.16, 0.22, 1.0));
    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(normal, halfDir), 0.0), 18.0) * 0.28;
    float phongShade = (0.74 + diffuse * 0.42) * (0.84 + normalZ * 0.16);
    vec3 finalColor = vParticleColor * phongShade + vParticlePalette * specular;
    finalColor = finalColor * (1.0 + centerGradient * LIGHT_BOOST) + vParticlePalette * (centerGradient * CHROMA_BOOST);
    float peak = max(max(finalColor.r, finalColor.g), max(finalColor.b, COLOR_PEAK));
    finalColor *= COLOR_PEAK / peak;
    float visibleAlpha = min(1.0, alpha * (1.0 + centerGradient * ALPHA_BOOST));
    if (visibleAlpha <= ALPHA_CUTOFF) discard;
    o = vec4(finalColor, visibleAlpha);
  }`;

  /* =======================================================================
     ParticleSystem — GPGPU ping-pong + instanced billboard rendering
     ======================================================================= */
  const P_DEFAULTS = {
    flowStrength: 1.05, depthLift: 0.95, flowThreshold: 0.05, maxFlowSpeed: 12,
    responseGamma: 4, perpendicularAngle: 1.25, sideVariation: 1, depthAttenuationScale: 2,
    spring: 4.0, zeta: 1.15, dragLin: 0.28, dragQuad: 0.05, aMax: 24, vMaxScale: 1,
    pointSize: 9, rotationSpeed: 0.07,
  };

  function fillParticleData(mode, size) {
    const count = size * size;
    const positions = new Float32Array(count * 4);
    const velocities = new Float32Array(count * 4);
    const destinations = new Float32Array(count * 4);
    const attributes = new Float32Array(count * 4);
    const uvs = new Float32Array(count * 2);
    const seeds = new Float32Array(count);
    const GA = Math.PI * (3 - Math.sqrt(5));
    const hash11 = (n) => { const s = Math.sin(n) * 43758.5453123; return s - Math.floor(s); };
    for (let i = 0; i < count; i++) {
      const u = ((i % size) + 0.5) / size, v = (Math.floor(i / size) + 0.5) / size;
      const seed = (i * 0.61803398875) % 1;
      let x = 0, y = 0, z = 0;
      if (mode === 'plane2d') {
        const angle = i * GA, r01 = Math.sqrt((i + 0.5) / count), R = 2.0;
        x = Math.cos(angle) * r01 * R; y = Math.sin(angle) * r01 * R; z = 0;
      } else {
        const R = 2.0, yNorm = 1 - 2 * ((i + 0.5) / count);
        const ring = Math.sqrt(Math.max(0, 1 - yNorm * yNorm)), theta = i * GA;
        x = Math.cos(theta) * ring * R; y = yNorm * R; z = Math.sin(theta) * ring * R;
      }
      positions[i * 4] = x; positions[i * 4 + 1] = y; positions[i * 4 + 2] = z; positions[i * 4 + 3] = 1;
      destinations[i * 4] = x; destinations[i * 4 + 1] = y; destinations[i * 4 + 2] = z;
      destinations[i * 4 + 3] = mode === 'plane2d' ? 1.15 : 0.82;
      const sizeRand = hash11(i * 12.9898 + 78.233);
      attributes[i * 4] = mode === 'plane2d' ? 0.8 + sizeRand * 0.5 : 0.75 + sizeRand * 0.6;
      attributes[i * 4 + 1] = mode === 'plane2d' ? 3.2 : 2.6;
      attributes[i * 4 + 2] = 0; attributes[i * 4 + 3] = seed;
      uvs[i * 2] = u; uvs[i * 2 + 1] = v; seeds[i] = seed;
    }
    return { positions, velocities, destinations, attributes, uvs, seeds };
  }

  class ParticleSystem {
    constructor(gl, mode, size) {
      this.gl = gl;
      this.mode = mode;
      this.size = size;
      this.planeLock = mode === 'plane2d' ? 1 : 0;
      const A = { position: 0 };
      this.velProg = program(gl, QUAD_VS, P_VELOCITY_FS, A, 'p_velocity');
      this.posProg = program(gl, QUAD_VS, P_POSITION_FS, A, 'p_position');
      this.renderProg = program(gl, P_RENDER_VS, P_RENDER_FS,
        { position: 0, uv: 1, aParticleUv: 2, aSeed: 3 }, 'p_render');
      this.ok = !!(this.velProg && this.posProg && this.renderProg);
      if (!this.ok) return;

      const data = fillParticleData(mode, size);
      const near = { filter: gl.NEAREST };
      const f32 = { filter: gl.NEAREST, internalFormat: gl.RGBA32F, type: gl.FLOAT };
      // ping-pong pos/vel seeded on both read & write
      this.position = makeDouble(gl, size, size, Object.assign({}, near, { data: data.positions, internalFormat: gl.RGBA16F, type: gl.FLOAT }));
      this.velocity = makeDouble(gl, size, size, Object.assign({}, near, { data: data.velocities, internalFormat: gl.RGBA16F, type: gl.FLOAT }));
      this.destTex = makeTex(gl, size, size, gl.NEAREST, gl.RGBA32F, gl.RGBA, gl.FLOAT, data.destinations);
      this.attrTex = makeTex(gl, size, size, gl.NEAREST, gl.RGBA32F, gl.RGBA, gl.FLOAT, data.attributes);

      // fullscreen triangle for the GPGPU update passes
      this.triVao = gl.createVertexArray();
      gl.bindVertexArray(this.triVao);
      const triBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, triBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

      // instanced billboard geometry
      this.renderVao = gl.createVertexArray();
      gl.bindVertexArray(this.renderVao);
      const posBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      const uvBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      const puvBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, puvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, data.uvs, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(2, 1);
      const seedBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, seedBuf);
      gl.bufferData(gl.ARRAY_BUFFER, data.seeds, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(3, 1);
      const idx = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
      gl.bindVertexArray(null);

      this.count = size * size;
      this.time = 0;
      this.spinAngle = 0;
    }

    _blit(target) {
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.w, target.h);
      gl.bindVertexArray(this.triVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    _bind(prog, name, tex, unit) {
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (prog.loc[name]) gl.uniform1i(prog.loc[name], unit);
    }

    // velocityField: fluid velocity texture; params: tuning (P_DEFAULTS shape)
    step(dt, velocityField, params, camera) {
      if (!this.ok) return;
      const gl = this.gl;
      dt = Math.min(Math.max(dt, 1e-6), 1 / 30);
      this.time += dt;
      this.spinAngle += (params.rotationSpeed ?? 0) * dt;

      const model = this.mode === 'plane2d' ? M.ident() : M.rotationY(this.spinAngle);
      const view = M.translation(0, 0, -camera.dist);
      const proj = M.perspective(camera.fov, camera.aspect, 0.1, 100);
      const invRot = M.invRot3(model);
      const rightLocal = M.norm(M.mul3v(invRot, 1, 0, 0));
      const upLocal = M.norm(M.mul3v(invRot, 0, 1, 0));

      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);

      // --- velocity pass ---
      let pr = this.velProg;
      gl.useProgram(pr.p);
      this._bind(pr, 'uPositionTexture', this.position.read.tex, 0);
      this._bind(pr, 'uVelocityTexture', this.velocity.read.tex, 1);
      this._bind(pr, 'uDestinationTexture', this.destTex, 2);
      this._bind(pr, 'uAttributeTexture', this.attrTex, 3);
      this._bind(pr, 'uFlow', velocityField, 4);
      gl.uniformMatrix4fv(pr.loc.uViewMatrix, false, view);
      gl.uniformMatrix4fv(pr.loc.uProjectionMatrix, false, proj);
      gl.uniformMatrix4fv(pr.loc.uModelMatrix, false, model);
      gl.uniform3f(pr.loc.uCameraRight, 1, 0, 0);
      gl.uniform3f(pr.loc.uCameraUp, 0, 1, 0);
      gl.uniform1f(pr.loc.uDeltaTime, dt);
      gl.uniform1f(pr.loc.uFlowStrength, params.flowStrength);
      gl.uniform1f(pr.loc.uDepthLift, this.mode === 'plane2d' ? 0 : params.depthLift);
      gl.uniform1f(pr.loc.uMaxFlowSpeed, params.maxFlowSpeed);
      gl.uniform1f(pr.loc.uFlowThresh, params.flowThreshold);
      gl.uniform1f(pr.loc.uFlowPow, params.responseGamma);
      gl.uniform1f(pr.loc.uPerpendicularAngle, params.perpendicularAngle);
      gl.uniform1f(pr.loc.uOmega, params.spring);
      gl.uniform1f(pr.loc.uZeta, params.zeta);
      gl.uniform1f(pr.loc.uDragLin, params.dragLin);
      gl.uniform1f(pr.loc.uDragQuad, params.dragQuad);
      gl.uniform1f(pr.loc.uAMax, params.aMax);
      gl.uniform1f(pr.loc.uVMaxScale, params.vMaxScale);
      gl.uniform1f(pr.loc.uSideVariation, params.sideVariation);
      gl.uniform1f(pr.loc.uPlaneLock, this.planeLock);
      gl.uniform1f(pr.loc.uDepthScale, params.depthAttenuationScale);
      this._blit(this.velocity.write);
      swap(this.velocity);

      // --- position pass ---
      pr = this.posProg;
      gl.useProgram(pr.p);
      this._bind(pr, 'uPositionTexture', this.position.read.tex, 0);
      this._bind(pr, 'uVelocityTexture', this.velocity.read.tex, 1);
      this._bind(pr, 'uDestinationTexture', this.destTex, 2);
      gl.uniform1f(pr.loc.uDeltaTime, dt);
      gl.uniform1f(pr.loc.uPlaneLock, this.planeLock);
      this._blit(this.position.write);
      swap(this.position);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // stash render-time matrices
      this._modelView = M.mul(view, model);
      this._proj = proj;
      this._rightLocal = rightLocal;
      this._upLocal = upLocal;
      this._pointSize = params.pointSize;
    }

    // render to the currently-bound framebuffer (default = canvas)
    render(viewportW, viewportH) {
      if (!this.ok || !this._proj) return;
      const gl = this.gl;
      const pr = this.renderProg;
      gl.useProgram(pr.p);
      gl.viewport(0, 0, viewportW, viewportH);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
      this._bind(pr, 'uPositionTexture', this.position.read.tex, 0);
      this._bind(pr, 'uVelocityTexture', this.velocity.read.tex, 1);
      this._bind(pr, 'uAttributeTexture', this.attrTex, 2);
      gl.uniform1f(pr.loc.uPointSize, this._pointSize);
      gl.uniform1f(pr.loc.uTime, this.time);
      gl.uniform3fv(pr.loc.uCameraRightLocal, this._rightLocal);
      gl.uniform3fv(pr.loc.uCameraUpLocal, this._upLocal);
      gl.uniformMatrix4fv(pr.loc.uProjectionMatrix, false, this._proj);
      gl.uniformMatrix4fv(pr.loc.uModelViewMatrix, false, this._modelView);
      gl.bindVertexArray(this.renderVao);
      gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, this.count);
      gl.bindVertexArray(null);
      gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
      gl.disable(gl.DEPTH_TEST);
    }

    dispose() {
      const gl = this.gl;
      [this.position, this.velocity].forEach(d => {
        if (!d) return;
        gl.deleteTexture(d.read.tex); gl.deleteFramebuffer(d.read.fbo);
        gl.deleteTexture(d.write.tex); gl.deleteFramebuffer(d.write.fbo);
      });
      if (this.destTex) gl.deleteTexture(this.destTex);
      if (this.attrTex) gl.deleteTexture(this.attrTex);
      if (this.velProg) gl.deleteProgram(this.velProg.p);
      if (this.posProg) gl.deleteProgram(this.posProg.p);
      if (this.renderProg) gl.deleteProgram(this.renderProg.p);
    }
  }

  /* =======================================================================
     Pointer + idle auto-motion → splats. Shared by both view types.
     ======================================================================= */
  function makePointerDriver(canvas, sim, opts) {
    opts = opts || {};
    const colored = opts.coloredStrokes ?? false;
    const captureTouch = opts.captureTouch ?? false;
    const touchForceScale = opts.touchForceScale ?? 6.5;
    const touchRadiusScale = opts.touchRadiusScale ?? 2.4;
    let lastX = 0, lastY = 0, hasPointer = false, lastT = 0, activePointerId = null;
    let strokeColor = hsv2rgb(Math.random(), 1, 1);
    let colorTimer = 0;
    const state = { lastInteract: -1e9 };

    const preventTouchDefault = (e) => {
      if (captureTouch && e.pointerType === 'touch' && e.cancelable) e.preventDefault();
    };
    const acceptsPointer = (e) => {
      if (e.isPrimary === false) return false;
      return activePointerId === null || e.pointerId === activePointerId;
    };
    const seedPointer = (e) => {
      lastX = e.clientX;
      lastY = e.clientY;
      lastT = e.timeStamp || performance.now();
      hasPointer = true;
      state.lastInteract = lastT;
    };
    const applySample = (sample, source) => {
      const r = canvas.getBoundingClientRect();
      if (r.width < 4 || r.height < 4) return;
      const now = sample.timeStamp || source.timeStamp || performance.now();
      const gap = now - lastT;
      if (gap > 200) hasPointer = false;
      lastT = now;
      if (colored) {
        colorTimer += Math.min(Math.max(gap, 0), 100) / 1000 * 10;
        if (!hasPointer || colorTimer >= 1) { if (colorTimer >= 1) colorTimer %= 1; strokeColor = hsv2rgb(Math.random(), 1, 1); }
      }
      const x = (sample.clientX - r.left) / r.width;
      const y = 1 - (sample.clientY - r.top) / r.height;
      const mx = typeof sample.movementX === 'number' && sample.movementX !== 0 ? sample.movementX : sample.clientX - lastX;
      const my = typeof sample.movementY === 'number' && sample.movementY !== 0 ? sample.movementY : sample.clientY - lastY;
      const dx = hasPointer ? mx : 0;
      const dy = hasPointer ? -my : 0;
      lastX = sample.clientX; lastY = sample.clientY; hasPointer = true;
      const touch = source.pointerType === 'touch';
      // A finger drag travels fewer pixels-per-event than a mouse flick (and iOS
      // never reports movementX/Y), so keep the jitter floor much lower on touch
      // or slow strokes get swallowed entirely and feel dead.
      if (Math.abs(dx) + Math.abs(dy) < (touch ? 0.05 : 0.25)) return;
      state.lastInteract = now;
      const force = sim.splatForce * (touch ? touchForceScale : 1);
      const dyeColor = colored ? [strokeColor[0] * 0.3, strokeColor[1] * 0.3, strokeColor[2] * 0.3] : undefined;
      const splatOpts = {};
      if (touch) splatOpts.radius = sim.splatRadius * touchRadiusScale;
      if (dyeColor) splatOpts.dyeColor = dyeColor;
      sim.addSplat(x, y, dx * force, dy * force, Object.keys(splatOpts).length ? splatOpts : undefined);
    };
    const onDown = (e) => {
      if (!acceptsPointer(e)) return;
      preventTouchDefault(e);
      activePointerId = e.pointerId;
      seedPointer(e);
      if (canvas.setPointerCapture) {
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
      }
    };
    const onMove = (e) => {
      if (!acceptsPointer(e)) return;
      preventTouchDefault(e);
      const samples = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : null;
      if (samples && samples.length) samples.forEach(sample => applySample(sample, e));
      else applySample(e, e);
    };
    const reset = (e) => {
      if (e && activePointerId !== null && e.pointerId !== activePointerId) return;
      hasPointer = false;
      activePointerId = null;
    };
    canvas.addEventListener('pointerdown', onDown, { passive: false });
    canvas.addEventListener('pointermove', onMove, { passive: false });
    canvas.addEventListener('pointerup', reset);
    canvas.addEventListener('pointerout', reset);
    canvas.addEventListener('pointercancel', reset);
    canvas.addEventListener('lostpointercapture', reset);

    state.detach = () => {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', reset);
      canvas.removeEventListener('pointerout', reset);
      canvas.removeEventListener('pointercancel', reset);
      canvas.removeEventListener('lostpointercapture', reset);
    };
    return state;
  }

  /* =======================================================================
     FluidView — canvas running a fluid sim + an overlay + pointer/idle splats
     ======================================================================= */
  class FluidView {
    constructor(canvas, def, opts) {
      opts = opts || {};
      this.canvas = canvas;
      this.def = def;
      this.maxDim = opts.maxDim || 520;
      this.keepContext = opts.keepContext ?? false;
      this.interactive = opts.interactive ?? false;
      this.active = true;
      this.elapsed = 0;
      this.autoPrev = null;
      this.autoPhase = Math.random() * 10;
      const overlayKey = def.overlay || 'colorful';
      const overlayDef = OVERLAYS[overlayKey];

      const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false, powerPreference: 'high-performance' });
      this.ok = false;
      if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;
      this.gl = gl;
      // Paint the dark theme at once so the canvas never flashes white before
      // the first rendered frame.
      gl.clearColor(0.02, 0.02, 0.05, 1); gl.clear(gl.COLOR_BUFFER_BIT);

      this.sim = new FluidSim(gl, Object.assign({
        profile: opts.profile || 'balanced',
        enableDye: !!overlayDef.dye,
      }, def.sim || {}));
      if (!this.sim.ok) return;

      this.overlayProg = program(gl, QUAD_VS, overlayDef.fs, { position: 0 }, 'overlay.' + overlayKey);
      if (!this.overlayProg) return;
      this.overlayDef = overlayDef;

      // fullscreen triangle for overlay
      this.triVao = gl.createVertexArray();
      gl.bindVertexArray(this.triVao);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      // tunables exposed to the GUI
      this.state = {
        intensity: 1.1, vibrance: 0.15, opacity: 1.0,
        splatForce: this.sim.splatForce, curlStrength: this.sim.curlStrength,
        velocityDissipation: this.sim.velocityDissipation,
        densityDissipation: this.sim.densityDissipation,
      };
      if (def.state) Object.assign(this.state, def.state);
      // Palette (0..1) drives the procedural backdrop + the cursor/tint colours,
      // and surfaces as Base/Mid/Highlight pickers in the modal — the same UX
      // the procedural GLSL cards already have.
      ['uColA', 'uColB', 'uColC'].forEach(k => {
        const c = (def.colors && def.colors[k]) || [218, 232, 255];
        this.state[k] = [c[0] / 255, c[1] / 255, c[2] / 255];
      });
      this.bg = def.bg || [0.02, 0.02, 0.05];
      this.scene = ({ gradient: 0, shapes: 1, torus: 2, text: 3 })[def.scene] ?? 0;

      // Text scene — rasterise a wordmark (default "mydesignbox") to a texture
      // that the effect samples as its scene, so distortion passes refract the
      // real letters (mirrors three-fluid-fx's brand-text RenderPass scene).
      if (this.scene === 3) {
        this.sceneText = def.text || 'mydesignbox';
        this.sceneCanvas = document.createElement('canvas');
        this.sceneTexKey = '';
        this.sceneTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      }

      this.pointer = makePointerDriver(canvas, this.sim, {
        coloredStrokes: !!overlayDef.dye,
        captureTouch: opts.touchCapture ?? false,
      });
      this.ok = true;
    }

    _resize() {
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      let w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
      const scale = Math.min(1, this.maxDim / Math.max(w, h, 1));
      w = Math.max(2, Math.round(w * scale)); h = Math.max(2, Math.round(h * scale));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.sim.resize(w, h); }
    }

    _autoSplat(dt) {
      // idle drift: emit gentle splats along a Lissajous path when the pointer
      // has been quiet, so cards stay alive like the reference demo.
      const now = performance.now();
      if (now - this.pointer.lastInteract < 900) { this.autoPrev = null; return; }
      const t = this.elapsed, ph = this.autoPhase;
      const x = 0.5 + 0.30 * Math.sin(t * 0.42 + ph);
      const y = 0.5 + 0.24 * Math.sin(t * 0.53 * 1.3 + ph * 1.7);
      if (!this.autoPrev) { this.autoPrev = { x, y }; return; }
      const cw = this.canvas.clientWidth || 320, chh = this.canvas.clientHeight || 200;
      const dx = (x - this.autoPrev.x) * cw, dy = (y - this.autoPrev.y) * chh;
      this.autoPrev = { x, y };
      const force = this.sim.splatForce * 1.0;
      const dye = this.sim.enableDye ? hsv2rgb((t * 0.06 + ph) % 1, 1, 1).map(c => c * 0.3) : undefined;
      this.sim.addSplat(x, y, dx * force, dy * force, dye ? { dyeColor: dye } : undefined);
    }

    syncParams() {
      const s = this.state;
      this.sim.splatForce = s.splatForce;
      this.sim.curlStrength = s.curlStrength;
      this.sim.velocityDissipation = s.velocityDissipation;
      this.sim.densityDissipation = s.densityDissipation;
      this.sim.dyeDissipation = s.densityDissipation;
    }

    // Rasterise the wordmark to `sceneTex` (cached by size + palette). Caller
    // must have bound `sceneTex` to the active unit. Gradient runs uColA→B→C so
    // the modal's Base/Mid/Highlight pickers recolour the letters live.
    _renderSceneText(w, h) {
      const S = this.state, a = S.uColA, b = S.uColB, c = S.uColC;
      const hex = v => '#' + v.map(x => Math.max(0, Math.min(255, Math.round(x * 255))).toString(16).padStart(2, '0')).join('');
      const ca = hex(a), cb = hex(b), cc = hex(c);
      const key = w + 'x' + h + '|' + this.sceneText + '|' + ca + cb + cc;
      if (key === this.sceneTexKey) return;
      this.sceneTexKey = key;
      const tc = this.sceneCanvas; tc.width = w; tc.height = h;
      const ctx = tc.getContext('2d');
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const FONT = ' Inter, ui-sans-serif, system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
      let ts = Math.round(h * 0.42);
      ctx.font = '800 ' + ts + 'px' + FONT;
      while (ts > 8 && ctx.measureText(this.sceneText).width > w * 0.92) { ts -= 2; ctx.font = '800 ' + ts + 'px' + FONT; }
      const grad = ctx.createLinearGradient(w * 0.04, 0, w * 0.96, 0);
      grad.addColorStop(0, ca); grad.addColorStop(0.55, cb); grad.addColorStop(1, cc);
      ctx.fillStyle = grad;
      ctx.fillText(this.sceneText, w / 2, h / 2);
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tc);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    }

    render(time, dt) {
      if (!this.ok || !this.active) return;
      this._resize();
      this.elapsed += dt;
      this.syncParams();
      this._autoSplat(dt);
      this.sim.step(dt);

      const gl = this.gl, pr = this.overlayProg, S = this.state;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
      gl.useProgram(pr.p);
      // Bind all three fluid fields to fixed units; each effect samples the
      // ones it declares (the rest compile out and have no location).
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.sim.densityTexture);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.sim.dyeTexture);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.sim.velocityTexture);
      if (pr.loc.tFluid) gl.uniform1i(pr.loc.tFluid, 0);
      if (pr.loc.tDye) gl.uniform1i(pr.loc.tDye, 1);
      if (pr.loc.tVelocity) gl.uniform1i(pr.loc.tVelocity, 2);
      if (pr.loc.uBg) gl.uniform3fv(pr.loc.uBg, this.bg);
      if (pr.loc.uIntensity) gl.uniform1f(pr.loc.uIntensity, S.intensity);
      if (pr.loc.uVibrance) gl.uniform1f(pr.loc.uVibrance, S.vibrance);
      if (pr.loc.uOpacity) gl.uniform1f(pr.loc.uOpacity, S.opacity ?? 1);
      if (pr.loc.uTime) gl.uniform1f(pr.loc.uTime, this.elapsed);
      if (pr.loc.uTexel) gl.uniform2f(pr.loc.uTexel, 1 / this.sim.dyeW, 1 / this.sim.dyeH);
      if (pr.loc.uRes) gl.uniform2f(pr.loc.uRes, this.canvas.width, this.canvas.height);
      if (pr.loc.uScene) gl.uniform1i(pr.loc.uScene, this.scene);
      if (pr.loc.uColA) gl.uniform3fv(pr.loc.uColA, S.uColA);
      if (pr.loc.uColB) gl.uniform3fv(pr.loc.uColB, S.uColB);
      if (pr.loc.uColC) gl.uniform3fv(pr.loc.uColC, S.uColC);
      if (pr.loc.uCursorColor) gl.uniform3fv(pr.loc.uCursorColor, S.uColC);
      if (pr.loc.uTint) gl.uniform3fv(pr.loc.uTint, S.uColB);
      if (this.scene === 3 && this.sceneTex) {
        gl.activeTexture(gl.TEXTURE0 + 3);
        gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
        this._renderSceneText(this.canvas.width, this.canvas.height); // re-uploads only if size/palette changed
        if (pr.loc.uSceneTex) gl.uniform1i(pr.loc.uSceneTex, 3);
      }
      gl.bindVertexArray(this.triVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    dispose() {
      this.active = false;
      if (this.pointer) this.pointer.detach();
      if (this.overlayProg) this.gl.deleteProgram(this.overlayProg.p);
      if (this.sceneTex) this.gl.deleteTexture(this.sceneTex);
      if (this.sim) this.sim.dispose();
      // The modal reuses one canvas (= one context); keep it alive across opens
      // and only free GL objects. Standalone card views may release the context.
      if (!this.keepContext && this.gl) { const e = this.gl.getExtension('WEBGL_lose_context'); if (e) e.loseContext(); }
    }
  }

  /* =======================================================================
     ParticleView — fluid velocity field driving a GPGPU particle cloud
     ======================================================================= */
  class ParticleView {
    constructor(canvas, def, opts) {
      opts = opts || {};
      this.canvas = canvas;
      this.def = def;
      this.maxDim = opts.maxDim || 560;
      this.keepContext = opts.keepContext ?? false;
      this.interactive = opts.interactive ?? false;
      this.active = true;
      this.elapsed = 0;
      this.autoPrev = null;
      this.autoPhase = Math.random() * 10;
      this.mode = def.mode || 'cloud3d';

      const gl = canvas.getContext('webgl2', { alpha: false, antialias: true, premultipliedAlpha: false, powerPreference: 'high-performance' });
      this.ok = false;
      if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;
      this.gl = gl;
      gl.clearColor(0.027, 0.031, 0.043, 1); gl.clear(gl.COLOR_BUFFER_BIT);

      this.sim = new FluidSim(gl, Object.assign({
        profile: opts.profile || 'balanced',
        enableDye: false, reflectWalls: true, curlStrength: 0.2,
        densityDissipation: 0.97,
      }, def.sim || {}));
      if (!this.sim.ok) return;

      const size = (opts.particleSize) || def.particleSize || (this.mode === 'plane2d' ? 48 : 56);
      this.particles = new ParticleSystem(gl, this.mode, size);
      if (!this.particles.ok) return;

      // faint density-tint background behind the particles
      this.tintProg = program(gl, QUAD_VS, DENSITYTINT_FS, { position: 0 }, 'densitytint');
      // optional FBM-lit backdrop (the "Mega" demo's low-poly gradient)
      this.backdrop = !!def.backdrop;
      this.backdropProg = this.backdrop ? program(gl, QUAD_VS, BACKDROP_FS, { position: 0 }, 'backdrop') : null;
      this.backdropBase = def.backdropBase || [0.016, 0.024, 0.04];
      this.backdropWarm = def.backdropWarm || [0.62, 0.0, 0.20];
      this.backdropCool = def.backdropCool || [0.10, 0.25, 0.78];
      this.triVao = gl.createVertexArray();
      gl.bindVertexArray(this.triVao);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      this.bg = def.bg || [0.027, 0.031, 0.043];
      this.tint = def.tint || [0.08, 0.3, 0.32];
      this.tintIntensity = def.tintIntensity ?? 0.18;
      this.state = Object.assign({}, P_DEFAULTS, def.state || {});
      if (this.mode === 'plane2d') this.tint = def.tint || [0.18, 0.05, 0.22];

      this.pointer = makePointerDriver(canvas, this.sim, {
        coloredStrokes: false,
        captureTouch: opts.touchCapture ?? false,
      });
      this.ok = true;
    }

    _resize() {
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      let w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
      const scale = Math.min(1, this.maxDim / Math.max(w, h, 1));
      w = Math.max(2, Math.round(w * scale)); h = Math.max(2, Math.round(h * scale));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.sim.resize(w, h); }
    }

    _autoSplat(dt) {
      const now = performance.now();
      if (now - this.pointer.lastInteract < 900) { this.autoPrev = null; return; }
      const t = this.elapsed, ph = this.autoPhase;
      const x = 0.5 + 0.32 * Math.sin(t * 0.5 + ph);
      const y = 0.5 + 0.28 * Math.sin(t * 0.61 * 1.3 + ph * 1.7);
      if (!this.autoPrev) { this.autoPrev = { x, y }; return; }
      const cw = this.canvas.clientWidth || 320, chh = this.canvas.clientHeight || 200;
      const dx = (x - this.autoPrev.x) * cw, dy = (y - this.autoPrev.y) * chh;
      this.autoPrev = { x, y };
      const force = this.sim.splatForce * 1.1;
      this.sim.addSplat(x, y, dx * force, dy * force);
    }

    render(time, dt) {
      if (!this.ok || !this.active) return;
      this._resize();
      this.elapsed += dt;
      this._autoSplat(dt);
      this.sim.step(dt);

      const gl = this.gl, c = this.canvas;
      const camera = { dist: 6.4, fov: 45 * Math.PI / 180, aspect: c.width / c.height };
      this.particles.step(dt, this.sim.velocityTexture, this.state, camera);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, c.width, c.height);
      gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.disable(gl.DEPTH_TEST);
      // opaque FBM-lit backdrop first (Mega), then the additive density tint
      if (this.backdropProg) {
        gl.useProgram(this.backdropProg.p);
        const bp = this.backdropProg.loc;
        if (bp.uTime) gl.uniform1f(bp.uTime, this.elapsed);
        if (bp.uRes) gl.uniform2f(bp.uRes, c.width, c.height);
        if (bp.uBase) gl.uniform3fv(bp.uBase, this.backdropBase);
        if (bp.uWarm) gl.uniform3fv(bp.uWarm, this.backdropWarm);
        if (bp.uCool) gl.uniform3fv(bp.uCool, this.backdropCool);
        gl.bindVertexArray(this.triVao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindVertexArray(null);
      }

      // faint density tint (additive)
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.tintProg.p);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.sim.densityTexture);
      if (this.tintProg.loc.tFluid) gl.uniform1i(this.tintProg.loc.tFluid, 0);
      if (this.tintProg.loc.uTint) gl.uniform3fv(this.tintProg.loc.uTint, this.tint);
      if (this.tintProg.loc.uIntensity) gl.uniform1f(this.tintProg.loc.uIntensity, this.tintIntensity);
      gl.bindVertexArray(this.triVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);

      // particles
      this.particles.render(c.width, c.height);
    }

    dispose() {
      this.active = false;
      if (this.pointer) this.pointer.detach();
      if (this.tintProg) this.gl.deleteProgram(this.tintProg.p);
      if (this.backdropProg) this.gl.deleteProgram(this.backdropProg.p);
      if (this.particles) this.particles.dispose();
      if (this.sim) this.sim.dispose();
      if (!this.keepContext && this.gl) { const e = this.gl.getExtension('WEBGL_lose_context'); if (e) e.loseContext(); }
    }
  }

  /* =======================================================================
     FluidTextView — DOM-style text rasterised to a texture, refracted by the
     fluid (density-as-height) with rainbow dye ink on top.
     ======================================================================= */
  class FluidTextView {
    constructor(canvas, def, opts) {
      opts = opts || {};
      this.canvas = canvas;
      this.def = def;
      this.maxDim = opts.maxDim || 640;
      this.keepContext = opts.keepContext ?? false;
      this.interactive = opts.interactive ?? false;
      this.active = true;
      this.elapsed = 0;
      this.autoPrev = null;
      this.autoPhase = Math.random() * 10;
      this.textKey = '';

      const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false, powerPreference: 'high-performance' });
      this.ok = false;
      if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;
      this.gl = gl;
      gl.clearColor(0.02, 0.02, 0.05, 1); gl.clear(gl.COLOR_BUFFER_BIT);

      this.sim = new FluidSim(gl, Object.assign({
        profile: opts.profile || 'balanced', enableDye: true, reflectWalls: false,
        splatRadius: 0.0016, splatForce: 7, curlStrength: 0.18,
        velocityDissipation: 0.99, densityDissipation: 0.94, dyeDissipation: 0.965,
      }, def.sim || {}));
      if (!this.sim.ok) return;

      this.prog = program(gl, QUAD_VS, FLUIDTEXT_FS, { position: 0 }, 'fluidtext');
      if (!this.prog) return;

      this.triVao = gl.createVertexArray();
      gl.bindVertexArray(this.triVao);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      this.textCanvas = document.createElement('canvas');
      this.textTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.textTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      this.title = def.text || 'Fluid Text';
      this.kicker = def.kicker || 'THREE · FLUID · FX';
      this.lead = def.lead || 'Live type, bent by fluid.';
      this.bg = def.bg || [0.02, 0.02, 0.05];
      this.state = Object.assign({
        intensity: 1.0, distort: 0.5, vibrance: 0.4,
        splatForce: this.sim.splatForce, curlStrength: this.sim.curlStrength,
      }, def.state || {});

      this.pointer = makePointerDriver(canvas, this.sim, {
        coloredStrokes: true,
        captureTouch: opts.touchCapture ?? false,
      });
      this.ok = true;
    }

    _renderText(w, h) {
      const key = w + 'x' + h;
      if (key === this.textKey) return;
      this.textKey = key;
      const tc = this.textCanvas; tc.width = w; tc.height = h;
      const ctx = tc.getContext('2d');
      ctx.clearRect(0, 0, w, h);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const cx = w / 2, cy = h / 2, s = Math.min(w, h);
      const FONT = 'ui-sans-serif, system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
      ctx.fillStyle = '#ff7a5f';
      ctx.font = '800 ' + Math.max(8, Math.round(s * 0.030)) + 'px ' + FONT;
      ctx.fillText(this.kicker.toUpperCase(), cx, cy - s * 0.17);
      let ts = Math.round(s * 0.19);
      ctx.font = '820 ' + ts + 'px ' + FONT;
      while (ts > 10 && ctx.measureText(this.title).width > w * 0.9) { ts -= 2; ctx.font = '820 ' + ts + 'px ' + FONT; }
      ctx.fillStyle = '#f3f0e8';
      ctx.fillText(this.title, cx, cy + s * 0.01);
      ctx.fillStyle = 'rgba(243,240,232,0.72)';
      ctx.font = '600 ' + Math.max(8, Math.round(s * 0.034)) + 'px ' + FONT;
      ctx.fillText(this.lead, cx, cy + s * 0.18);
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.textTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tc);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    }

    _resize() {
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      let w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
      const scale = Math.min(1, this.maxDim / Math.max(w, h, 1));
      w = Math.max(2, Math.round(w * scale)); h = Math.max(2, Math.round(h * scale));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.sim.resize(w, h); }
      this._renderText(c.width, c.height);
    }

    _autoSplat(dt) {
      const now = performance.now();
      if (now - this.pointer.lastInteract < 900) { this.autoPrev = null; return; }
      const t = this.elapsed, ph = this.autoPhase;
      const x = 0.5 + 0.34 * Math.sin(t * 0.4 + ph);
      const y = 0.5 + 0.22 * Math.sin(t * 0.52 * 1.3 + ph * 1.7);
      if (!this.autoPrev) { this.autoPrev = { x, y }; return; }
      const cw = this.canvas.clientWidth || 320, chh = this.canvas.clientHeight || 200;
      const dx = (x - this.autoPrev.x) * cw, dy = (y - this.autoPrev.y) * chh;
      this.autoPrev = { x, y };
      const force = this.sim.splatForce;
      const dye = hsv2rgb((t * 0.07 + ph) % 1, 1, 1).map(v => v * 0.3);
      this.sim.addSplat(x, y, dx * force, dy * force, { dyeColor: dye });
    }

    syncParams() {
      this.sim.splatForce = this.state.splatForce;
      this.sim.curlStrength = this.state.curlStrength;
    }

    render(time, dt) {
      if (!this.ok || !this.active) return;
      this._resize();
      this.elapsed += dt;
      this.syncParams();
      this._autoSplat(dt);
      this.sim.step(dt);

      const gl = this.gl, pr = this.prog, S = this.state;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
      gl.useProgram(pr.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.textTex);
      if (pr.loc.uText) gl.uniform1i(pr.loc.uText, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.sim.densityTexture);
      if (pr.loc.tFluid) gl.uniform1i(pr.loc.tFluid, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.sim.dyeTexture);
      if (pr.loc.tDye) gl.uniform1i(pr.loc.tDye, 2);
      if (pr.loc.uTexel) gl.uniform2f(pr.loc.uTexel, 1 / this.sim.dyeW, 1 / this.sim.dyeH);
      if (pr.loc.uIntensity) gl.uniform1f(pr.loc.uIntensity, S.intensity);
      if (pr.loc.uDistort) gl.uniform1f(pr.loc.uDistort, S.distort);
      if (pr.loc.uVibrance) gl.uniform1f(pr.loc.uVibrance, S.vibrance);
      if (pr.loc.uBg) gl.uniform3fv(pr.loc.uBg, this.bg);
      gl.bindVertexArray(this.triVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    dispose() {
      this.active = false;
      if (this.pointer) this.pointer.detach();
      if (this.prog) this.gl.deleteProgram(this.prog.p);
      if (this.textTex) this.gl.deleteTexture(this.textTex);
      if (this.sim) this.sim.dispose();
      if (!this.keepContext && this.gl) { const e = this.gl.getExtension('WEBGL_lose_context'); if (e) e.loseContext(); }
    }
  }

  let supportCache;
  function supported() {
    if (supportCache !== undefined) return supportCache;
    try {
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2');
      supportCache = !!(gl && gl.getExtension('EXT_color_buffer_float'));
      if (gl) {
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
      }
      return supportCache;
    } catch (e) {
      supportCache = false;
      return supportCache;
    }
  }

  return {
    FluidSim, ParticleSystem, FluidView, ParticleView, FluidTextView,
    OVERLAYS, P_DEFAULTS, supported,
    sources: {
      COLORFUL_FS, RAINBOWINK_FS, VOLUME_FS, BACKDROP_FS, FLUIDTEXT_FS,
      P_VELOCITY_FS, P_RENDER_VS, P_RENDER_FS,
      SPLAT_FS, ADVECT_FS,
    },
  };
})();
