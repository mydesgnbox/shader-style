/* =========================================================================
   gallery.js — WebGL2 engine + UI for the Shader Library
   ========================================================================= */
(() => {
  const VERT = `#version 300 es
  precision highp float;
  const vec2 verts[3] = vec2[3](vec2(-1.,-1.), vec2(3.,-1.), vec2(-1.,3.));
  void main(){ gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0); }`;

  const UNIFORMS = ['uRes','uTime','uMouse','uSpeed','uScale','uIntensity','uWarp','uHue','uColA','uColB','uColC'];

  // global controls (driven by the page-level lil-gui), applied to every view
  const GLOBAL = { speed:1.0, hue:0.0, glow:1.0, paused:false };

  /* ---------- one live shader surface ---------- */
  class ShaderView {
    constructor(canvas, def, { maxDim = 640, interactive = false } = {}) {
      this.canvas = canvas;
      this.def = def;
      this.maxDim = maxDim;
      this.interactive = interactive;
      this.active = true;
      this.mouse = [0.5, 0.5]; // normalized, y-up
      this.state = ShaderView.defaultState(def);
      this.ok = this.init();
      if (interactive) this.bindMouse();
    }

    static defaultState(def) {
      const s = {};
      def.params.forEach(p => { s[p.key] = p.value; });
      ['uColA','uColB','uColC'].forEach(k => {
        const c = def.colors[k];
        s[k] = [c[0]/255, c[1]/255, c[2]/255]; // lil-gui addColor expects 0..1
      });
      return s;
    }

    init() {
      const gl = this.canvas.getContext('webgl2', { antialias:false, alpha:false, premultipliedAlpha:false, preserveDrawingBuffer:true, powerPreference:'high-performance' });
      if (!gl) return false;
      this.gl = gl;
      const vs = this.compile(gl.VERTEX_SHADER, VERT);
      const fs = this.compile(gl.FRAGMENT_SHADER, window.SHADER_HEADER + this.def.body);
      if (!vs || !fs) return false;
      const prog = gl.createProgram();
      gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.error('link', this.def.id, gl.getProgramInfoLog(prog)); return false; }
      this.prog = prog;
      this.loc = {};
      UNIFORMS.forEach(u => { this.loc[u] = gl.getUniformLocation(prog, u); });
      gl.useProgram(prog);
      // Paint the dark theme immediately so the canvas never flashes white
      // before the first animation frame (or if the tab loaded in the background).
      gl.clearColor(0.02, 0.02, 0.05, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      return true;
    }

    compile(type, src) {
      const gl = this.gl, sh = gl.createShader(type);
      gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error('compile', this.def.id, gl.getShaderInfoLog(sh)); return null;
      }
      return sh;
    }

    bindMouse() {
      const set = (clientX, clientY) => {
        const r = this.canvas.getBoundingClientRect();
        this.mouse = [ (clientX - r.left)/r.width, 1.0 - (clientY - r.top)/r.height ];
      };
      this.canvas.addEventListener('pointermove', e => set(e.clientX, e.clientY));
      // hero also listens on window for ambient motion
    }

    resize() {
      const gl = this.gl, c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      let w = Math.round(c.clientWidth * dpr);
      let h = Math.round(c.clientHeight * dpr);
      const scale = Math.min(1, this.maxDim / Math.max(w, h));
      w = Math.max(2, Math.round(w * scale));
      h = Math.max(2, Math.round(h * scale));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    }

    render(time) {
      if (!this.ok || !this.active) return;
      const gl = this.gl;
      this.resize();
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.useProgram(this.prog);
      const L = this.loc, S = this.state;
      if (L.uRes)  gl.uniform2f(L.uRes, this.canvas.width, this.canvas.height);
      if (L.uTime) gl.uniform1f(L.uTime, time);
      if (L.uMouse) gl.uniform2f(L.uMouse, this.mouse[0]*this.canvas.width, this.mouse[1]*this.canvas.height);
      if (L.uSpeed) gl.uniform1f(L.uSpeed, (S.uSpeed ?? 1) * GLOBAL.speed);
      if (L.uScale) gl.uniform1f(L.uScale, S.uScale ?? 1);
      if (L.uIntensity) gl.uniform1f(L.uIntensity, (S.uIntensity ?? 1) * GLOBAL.glow);
      if (L.uWarp) gl.uniform1f(L.uWarp, S.uWarp ?? 1);
      if (L.uHue) gl.uniform1f(L.uHue, (S.uHue ?? 0) + GLOBAL.hue);
      if (L.uColA) gl.uniform3fv(L.uColA, S.uColA);
      if (L.uColB) gl.uniform3fv(L.uColB, S.uColB);
      if (L.uColC) gl.uniform3fv(L.uColC, S.uColC);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    dispose() {
      this.active = false;
      const gl = this.gl;
      if (gl) { const ext = gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); }
    }
  }

  /* ---------- master clock ---------- */
  const views = new Set();
  let clock = 0, last = performance.now();
  function loop(now) {
    const dt = Math.min(0.05, (now - last)/1000); last = now;
    if (!GLOBAL.paused) clock += dt;
    const simDt = GLOBAL.paused ? 0 : dt;   // fluid/particle views integrate real dt
    views.forEach(v => v.render(clock, simDt));
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  /* ---------- view factory: pick engine by def.kind ----------
     'fluid' / 'particle' cards are driven by fluidfx.js (real Navier–Stokes
     sim + GPGPU particles). If WebGL2 float targets are unavailable, or the
     engine view fails to init, fall back to a related procedural shader so
     the card is never blank. */
  function makeView(canvas, def, opts) {
    const kind = def.kind || 'glsl';
    if (kind !== 'glsl' && window.FluidFX && FluidFX.supported()) {
      try {
        const View = kind === 'particle' ? FluidFX.ParticleView
          : kind === 'fluidtext' ? FluidFX.FluidTextView
          : FluidFX.FluidView;
        const v = new View(canvas, def, opts);
        if (v && v.ok) return v;
      } catch (e) { console.warn('[gallery] fluidfx init failed for', def.id, e); }
    }
    if (kind !== 'glsl') {
      const fbId = kind === 'particle' ? 'metaballs' : 'fluid-flow';
      const fb = window.SHADERS.find(s => s.id === fbId && (s.kind || 'glsl') === 'glsl') || window.SHADERS[0];
      return new ShaderView(canvas, fb, { maxDim: opts.maxDim, interactive: opts.interactive });
    }
    return new ShaderView(canvas, def, opts);
  }

  // Source shown in the modal code panel for engine-driven cards.
  function codeSourceFor(def) {
    if (def.body) return (window.SHADER_HEADER + def.body).trim();
    const S = (window.FluidFX && FluidFX.sources) || {};
    if (def.kind === 'particle') {
      let src = (S.P_VELOCITY_FS || '') + '\n\n/* ===== billboard render ===== */\n' + (S.P_RENDER_FS || '');
      if (def.backdrop) src = (S.BACKDROP_FS || '') + '\n\n/* ===== particles ===== */\n' + src;
      return src.trim();
    }
    if (def.kind === 'fluidtext') {
      return ((S.FLUIDTEXT_FS || '') + '\n\n/* ===== advection (Stable Fluids) ===== */\n' + (S.ADVECT_FS || '')).trim();
    }
    if (def.kind === 'fluid') {
      const ov = (window.FluidFX && FluidFX.OVERLAYS && FluidFX.OVERLAYS[def.overlay]) || {};
      const fs = ov.fs || S.COLORFUL_FS || '';
      return (fs + '\n\n/* ===== advection (Stable Fluids) ===== */\n' + (S.ADVECT_FS || '')).trim();
    }
    return '';
  }

  /* ---------- build hero ---------- */
  const heroDef = window.SHADERS.find(s => s.id === 'fluid-flow');
  const heroView = new ShaderView(document.getElementById('hero-canvas'), heroDef, { maxDim: 1100, interactive:true });
  heroView.state.uScale = 2.4; heroView.state.uIntensity = 1.0;
  views.add(heroView);
  // ambient mouse on the whole hero region
  const heroEl = document.getElementById('hero');
  heroEl.addEventListener('pointermove', e => {
    const r = heroEl.getBoundingClientRect();
    heroView.mouse = [ (e.clientX-r.left)/r.width, 1.0-(e.clientY-r.top)/r.height ];
  });

  /* ---------- build cards ---------- */
  const grid = document.getElementById('grid');
  const cardViews = [];
  function colorHexes(def){ return ['uColA','uColB','uColC'].map(k=>{const c=def.colors[k];return `rgb(${c[0]},${c[1]},${c[2]})`;}); }

  window.SHADERS.forEach(def => {
    const card = document.createElement('article');
    card.className = 'card';
    card.dataset.tag = def.tag;
    card.dataset.group = def.group || 'Procedural';
    const propChips = def.params.filter(p=>p.key!=='uHue').map(p=>`<span>${p.name}</span>`).join('');
    const sw = colorHexes(def).map(c=>`<i style="color:${c};background:${c}"></i>`).join('');
    card.innerHTML = `
      <div class="stage">
        <canvas></canvas>
        <div class="grad"></div>
        <span class="tag">${def.tag}</span>
        <div class="swatches">${sw}</div>
        <span class="open-cue">OPEN · TWEAK ↗</span>
        <div class="meta">
          <h3>${def.name}</h3>
          <div class="blurb">${def.blurb}</div>
          <div class="props">${propChips}</div>
        </div>
      </div>`;
    grid.appendChild(card);
    cardViews.push({ def, card, view: null });
    card.addEventListener('click', () => openModal(def));
  });

  /* Lazy context lifecycle: a card holds a WebGL context only while it's on (or
     near) screen, and releases it when scrolled away. This caps the number of
     simultaneous WebGL2 contexts at roughly "what's visible" — comfortably
     under the browser's ~16-context limit even with the heavy fluid/particle
     cards. Procedural shaders re-init instantly; fluid/particle cards re-seed
     in a frame, so the recycling is visually seamless. */
  // Hard cap on simultaneously-live card contexts. Browsers allow ~16 WebGL
  // contexts; with 15 cards + the hero + a possible modal we must stay well
  // under that. When the cap is exceeded (e.g. a very tall window showing many
  // rows), release the live card farthest from the viewport centre.
  const MAX_LIVE_CARDS = 11;
  function ensureView(cv) {
    if (cv.view) return cv.view;
    cv.view = makeView(cv.card.querySelector('canvas'), cv.def, { maxDim: 520, interactive: !!cv.def.interactive });
    cv.view.active = !modalOpen;
    views.add(cv.view);
    return cv.view;
  }
  function releaseView(cv) {
    if (!cv.view) return;
    views.delete(cv.view);
    cv.view.dispose();
    cv.view = null;
    // Swap in a clean canvas so a later ensureView() gets a fresh context
    // instead of the one we just released.
    const fresh = document.createElement('canvas');
    cv.card.querySelector('canvas').replaceWith(fresh);
  }

  /* Staggered creation. Cards on (or near) screen are marked `wanted`; a pump
     then creates at most ONE context per frame — closest to the viewport
     centre first — and never beyond MAX_LIVE_CARDS. Creating every visible
     card in one burst can momentarily allocate >16 WebGL contexts (loseContext
     frees asynchronously), which makes the browser evict the oldest (the hero
     and first cards go blank). Pacing creation avoids that on tall displays. */
  const centerDist = c => { const r = c.card.getBoundingClientRect(); return Math.abs((r.top + r.bottom) / 2 - window.innerHeight / 2); };
  let pumpQueued = false;
  function schedulePump() { if (!pumpQueued) { pumpQueued = true; requestAnimationFrame(pumpViews); } }
  function pumpViews() {
    pumpQueued = false;
    cardViews.forEach(c => { if (c.view && !c.wanted) releaseView(c); });
    const live = cardViews.filter(c => c.view).length;
    if (live >= MAX_LIVE_CARDS) return;
    const wanted = cardViews.filter(c => c.wanted && !c.view);
    if (!wanted.length) return;
    wanted.sort((a, b) => centerDist(a) - centerDist(b));
    ensureView(wanted[0]);
    if (wanted.length > 1 && live + 1 < MAX_LIVE_CARDS) schedulePump(); // one more next frame
  }

  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      const cv = cardViews.find(c => c.card === e.target);
      if (!cv) return;
      cv.wanted = e.isIntersecting;
      if (!e.isIntersecting) releaseView(cv);
    });
    schedulePump();
  }, { rootMargin:'200px' });
  cardViews.forEach(c => io.observe(c.card));

  /* ---------- filters ---------- */
  // Group cards into a handful of coarse families (Procedural / Distortion /
  // Overlay / Particles / Type) so the bar stays usable across ~30+ cards —
  // far cleaner than one chip per per-card tag.
  const GROUP_ORDER = ['Procedural', 'Distortion', 'Overlay', 'Particles', 'Type'];
  const present = new Set(window.SHADERS.map(s => s.group || 'Procedural'));
  const groups = ['ALL', ...GROUP_ORDER.filter(g => present.has(g))];
  const filterBar = document.getElementById('filters');
  groups.forEach((t,i) => {
    const b = document.createElement('button');
    b.className = 'chip-filter' + (i===0?' active':'');
    b.textContent = t;
    b.addEventListener('click', () => {
      filterBar.querySelectorAll('.chip-filter').forEach(x=>x.classList.remove('active'));
      b.classList.add('active');
      cardViews.forEach(({card}) => {
        const show = (t==='ALL') || card.dataset.group===t;
        card.style.display = show ? '' : 'none';
      });
    });
    filterBar.appendChild(b);
  });

  /* ---------- detail modal ---------- */
  const modal = document.getElementById('modal');
  let modalOpen = false, modalView = null, modalGui = null;
  let modalCanvas = document.getElementById('modal-canvas');

  function highlight(src){
    return src
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/(\/\/[^\n]*)/g,'<span class="c">$1</span>')
      .replace(/\b(void|float|vec2|vec3|vec4|mat2|for|if|else|return|const|uniform|out|in|precision|highp)\b/g,'<span class="k">$1</span>')
      .replace(/\b(sin|cos|atan|length|dot|cross|mix|clamp|smoothstep|pow|fract|floor|abs|fbm|noise|hash21|hash22|hueShift|sqrt)\b/g,'<span class="t">$1</span>')
      .replace(/\b(\d+\.?\d*)\b/g,'<span class="n">$1</span>');
  }

  function openModal(def){
    modalOpen = true;
    modal.classList.add('open');
    cardViews.forEach(c => { if (c.view) c.view.active = false; });

    if (modalView) { views.delete(modalView); modalView.dispose(); }
    // Swap in a fresh canvas every open. A canvas only ever hands back its
    // first WebGL context (with that call's attributes), so reusing one canvas
    // across fluid↔particle↔glsl views is fragile. A new element each time
    // gives every view a clean context with the attributes it asked for —
    // so each open behaves exactly like a reliable first open.
    const freshCanvas = document.createElement('canvas');
    freshCanvas.id = 'modal-canvas';
    modalCanvas.replaceWith(freshCanvas);
    modalCanvas = freshCanvas;
    modalView = makeView(modalCanvas, def, { maxDim: 1400, interactive: true });
    views.add(modalView);

    // window-wide mouse so swirl follows everywhere in the view pane
    document.getElementById('modal-view').onpointermove = (e) => {
      const r = modalCanvas.getBoundingClientRect();
      modalView.mouse = [ (e.clientX-r.left)/r.width, 1.0-(e.clientY-r.top)/r.height ];
    };

    document.getElementById('modal-tag').textContent = def.tag;
    document.getElementById('modal-title').textContent = def.name;
    document.getElementById('modal-desc').textContent = def.blurb;
    document.getElementById('code-body').innerHTML = highlight(codeSourceFor(def));

    // build per-shader gui
    const host = document.getElementById('gui-host');
    host.innerHTML = '<div class="lbl">Uniforms</div>';
    if (modalGui) modalGui.destroy();
    modalGui = new lil.GUI({ container: host, title: def.name, width: 320 });
    const st = modalView.state || {};
    def.params.forEach(p => { if (p.key in st) modalGui.add(st, p.key, p.min, p.max, p.step).name(p.name); });
    if ('uColA' in st) {
      const cf = modalGui.addFolder('Colors');
      cf.addColor(st, 'uColA').name('Base');
      cf.addColor(st, 'uColB').name('Mid');
      cf.addColor(st, 'uColC').name('Highlight');
    }

    document.body.style.overflow = 'hidden';
  }

  function closeModal(){
    modalOpen = false;
    modal.classList.remove('open');
    document.body.style.overflow = '';
    if (modalView) { views.delete(modalView); modalView.dispose(); modalView = null; }
    if (modalGui) { modalGui.destroy(); modalGui = null; }
    // re-create / reactivate the cards that are back in view (the Intersection
    // observer also fires on scroll, but re-check now so on-screen cards resume
    // immediately after the modal closes).
    requestAnimationFrame(() => {
      cardViews.forEach(c => {
        const r = c.card.getBoundingClientRect();
        c.wanted = r.bottom > -200 && r.top < window.innerHeight + 200;
        if (!c.wanted) releaseView(c);
        else if (c.view) c.view.active = true;
      });
      schedulePump();
    });
  }

  document.getElementById('modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  window.addEventListener('keydown', e => { if (e.key === 'Escape' && modalOpen) closeModal(); });

  // copy code
  document.getElementById('copy-btn').addEventListener('click', () => {
    const txt = document.getElementById('code-body').textContent;
    navigator.clipboard?.writeText(txt);
    const b = document.getElementById('copy-btn'); const o = b.textContent;
    b.textContent = 'COPIED ✓'; setTimeout(()=>b.textContent=o, 1200);
  });

  /* ---------- global (page) gui ---------- */
  const gHost = document.getElementById('global-gui');
  const gg = new lil.GUI({ container: gHost, title: 'Global FX', width: 230 });
  gg.add(GLOBAL, 'speed', 0, 3, 0.01).name('Speed ×');
  gg.add(GLOBAL, 'hue', -3.14, 3.14, 0.01).name('Hue shift');
  gg.add(GLOBAL, 'glow', 0.4, 2, 0.01).name('Glow ×');
  gg.add(GLOBAL, 'paused').name('Pause');
})();
