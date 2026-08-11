(() => {
  const $ = id => document.getElementById(id);
  const out = $('out'), octx = out.getContext('2d', { willReadFrequently: false });
  const src = document.createElement('canvas'), sctx = src.getContext('2d', { willReadFrequently: true });

  // ---- bayer matrices -------------------------------------------------
  function bayer(n) {
    let m = [[0, 2], [3, 1]];
    while (m.length < n) {
      const s = m.length, next = [];
      for (let y = 0; y < s * 2; y++) next.push(new Array(s * 2));
      for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
        const v = m[y][x] * 4;
        next[y][x] = v; next[y][x + s] = v + 2;
        next[y + s][x] = v + 3; next[y + s][x + s] = v + 1;
      }
      m = next;
    }
    return m;
  }
  const B4 = bayer(4), B8 = bayer(8);

  // signature: build the UI's fill texture out of the tool's own dither
  (function bayerTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 8;
    const g = c.getContext('2d'), img = g.createImageData(8, 8);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const on = B8[y][x] / 64 < 0.42, i = (y * 8 + x) * 4;
      img.data[i] = 228; img.data[i + 1] = 23; img.data[i + 2] = 94;   // TWINEM pink
      img.data[i + 3] = on ? 255 : 0;
    }
    g.putImageData(img, 0, 0);
    document.documentElement.style.setProperty('--bayer', `url(${c.toDataURL()})`);
  })();

  // ---- diffusion kernels: [dx, dy, weight] -----------------------------
  const K = {
    fs: [[1, 0, 7 / 16], [-1, 1, 3 / 16], [0, 1, 5 / 16], [1, 1, 1 / 16]],
    atkinson: [[1, 0, .125], [2, 0, .125], [-1, 1, .125], [0, 1, .125], [1, 1, .125], [0, 2, .125]],
    jarvis: [[1,0,7/48],[2,0,5/48],[-2,1,3/48],[-1,1,5/48],[0,1,7/48],[1,1,5/48],[2,1,3/48],
             [-2,2,1/48],[-1,2,3/48],[0,2,5/48],[1,2,3/48],[2,2,1/48]],
    stucki: [[1,0,8/42],[2,0,4/42],[-2,1,2/42],[-1,1,4/42],[0,1,8/42],[1,1,4/42],[2,1,2/42],
             [-2,2,1/42],[-1,2,2/42],[0,2,4/42],[1,2,2/42],[2,2,1/42]],
    burkes: [[1,0,8/32],[2,0,4/32],[-2,1,2/32],[-1,1,4/32],[0,1,8/32],[1,1,4/32],[2,1,2/32]],
    sierra: [[1,0,5/32],[2,0,3/32],[-2,1,2/32],[-1,1,4/32],[0,1,5/32],[1,1,4/32],[2,1,2/32],
             [-1,2,2/32],[0,2,3/32],[1,2,2/32]],
    'sierra-lite': [[1, 0, .5], [-1, 1, .25], [0, 1, .25]]
  };

  const NOTES = {
    fs: 'The 1976 default. Four neighbours, tightest worms of the classic set.',
    atkinson: 'Original Macintosh. Only 3/4 of the error moves on, so highlights and shadows blow out to flat white and black. Crunchy.',
    jarvis: 'Twelve neighbours across three rows. Softest, slowest, longest worms.',
    stucki: 'Jarvis with cleaner weights. Sharper edges, still very fluid.',
    burkes: 'Stucki with the third row cut. Fast, punchy, good for video.',
    sierra: 'Between Stucki and Floyd-Steinberg. Balanced.',
    'sierra-lite': 'Three neighbours. Fastest diffusion, coarse texture.',
    bayer4: 'Fixed threshold grid. No worms, but stable frame to frame. Crosshatch look.',
    bayer8: 'Larger matrix, finer gradation, still perfectly stable in motion.',
    noise: 'Random threshold per pixel. Grain, not structure. Boils hard.',
    none: 'No dither. Straight cut at the threshold value.'
  };

  const PALETTES = [
    ['Classic', '#000000', '#FFFFFF'],
    ['TWINEM Pink', '#000000', '#E4175E'],
    ['TWINEM Teal', '#000000', '#3AC0C3']
  ];

  // slider values are stored raw (as the input element sees them), so a
  // preset is just a set of DOM values plus the non-slider toggles.
  const PRESETS = [
    ['Default', { algo: 'bayer8', diff: 86, blur: 0, scale: 1, con: 79, mid: 73, thr: 50, serp: false, inv: false, pal: 0 },
      'The saved look, and what the app loads with. Bayer 8x8 at 4px cells: an ordered grid, so it holds perfectly still frame to frame instead of boiling. Reset returns here too.']
  ];

  // must match the value attributes in index.html
  const DEFAULTS = { diff: 86, blur: 0, scale: 8, con: 79, mid: 73, thr: 50, hold: 1 };
  const DEFAULT_ALGO = 'bayer8';

  const S = {
    algo: DEFAULT_ALGO, diff: .86, blur: 0, serp: false,
    scale: 8, con: .79, mid: .73, thr: .5, inv: false,
    orient: matchMedia('(orientation: portrait)').matches ? 'port' : 'land',
    hold: 1, pal: 0, fgc: '#FFFFFF', bgc: '#000000', mode: 'demo', exporting: false
  };

  let video = null, stream = null, image = null;
  let lum = null, err = null, tmp = null, W = 0, H = 0, outImg = null;
  let holdCount = 0, frames = 0, fpsT = performance.now(), lastMs = 0;

  // ---- levels + blur ---------------------------------------------------
  function boxBlur(a, w, h, r) {
    if (r < 1) return a;
    if (!tmp || tmp.length !== a.length) tmp = new Float32Array(a.length);
    const d = r * 2 + 1;
    for (let y = 0; y < h; y++) {
      let acc = 0; const row = y * w;
      for (let i = -r; i <= r; i++) acc += a[row + Math.min(w - 1, Math.max(0, i))];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = acc / d;
        acc -= a[row + Math.min(w - 1, Math.max(0, x - r))];
        acc += a[row + Math.min(w - 1, Math.max(0, x + r + 1))];
      }
    }
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += tmp[Math.min(h - 1, Math.max(0, i)) * w + x];
      for (let y = 0; y < h; y++) {
        a[y * w + x] = acc / d;
        acc -= tmp[Math.min(h - 1, Math.max(0, y - r)) * w + x];
        acc += tmp[Math.min(h - 1, Math.max(0, y + r + 1)) * w + x];
      }
    }
    return a;
  }

  function levels(a) {
    const c = S.con, g = 1 / S.mid;
    for (let i = 0; i < a.length; i++) {
      let v = (a[i] - .5) * c + .5;
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      v = Math.pow(v, g);
      if (v < 0.06) v = 0;   // crush near-black: no lonely dots marching across dark fields
      a[i] = S.inv ? 1 - v : v;
    }
  }

  // ---- the dither pass -------------------------------------------------
  function dither(a, w, h) {
    const thr = S.thr, amt = S.diff, k = K[S.algo];

    if (S.algo === 'none') {
      for (let i = 0; i < a.length; i++) a[i] = a[i] >= thr ? 1 : 0;
      return;
    }
    if (S.algo === 'noise') {
      for (let i = 0; i < a.length; i++) a[i] = a[i] >= thr + (Math.random() - .5) * amt ? 1 : 0;
      return;
    }
    if (S.algo === 'bayer4' || S.algo === 'bayer8') {
      const m = S.algo === 'bayer4' ? B4 : B8, n = m.length, d = n * n;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const bias = (m[y % n][x % n] / d - .5) * amt;
        const i = y * w + x;
        a[i] = a[i] + bias >= thr ? 1 : 0;
      }
      return;
    }

    // error diffusion. sequential by definition: each pixel reads the
    // debt left by the ones before it.
    if (!err || err.length !== a.length) err = new Float32Array(a.length);
    err.fill(0);

    for (let y = 0; y < h; y++) {
      const rev = S.serp && (y & 1);
      for (let n = 0; n < w; n++) {
        const x = rev ? w - 1 - n : n;
        const i = y * w + x;
        const v = a[i] + err[i];
        const q = v >= thr ? 1 : 0;
        const e = (v - q) * amt;
        a[i] = q;
        for (let j = 0; j < k.length; j++) {
          const dx = rev ? -k[j][0] : k[j][0], dy = k[j][1];
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny >= h) continue;
          err[ny * w + nx] += e * k[j][2];
        }
      }
    }
  }

  // ---- source -> luminance --------------------------------------------
  function fillDemo(a, w, h, t) {
    const blobs = [
      [.5 + .28 * Math.cos(t * .00041), .5 + .22 * Math.sin(t * .00053), .30],
      [.5 + .30 * Math.cos(t * .00029 + 2.1), .5 + .26 * Math.sin(t * .00037 + 1.3), .24],
      [.5 + .20 * Math.cos(t * .00061 + 4.2), .5 + .30 * Math.sin(t * .00025 + 3.7), .19]
    ];
    for (let y = 0; y < h; y++) {
      const v = y / h;
      for (let x = 0; x < w; x++) {
        const u = x / w;
        let s = .18 + .30 * (1 - v);
        for (let b = 0; b < 3; b++) {
          const dx = u - blobs[b][0], dy = (v - blobs[b][1]) * (h / w) * (w / h);
          const d = Math.sqrt(dx * dx + dy * dy);
          s += .55 * Math.exp(-(d * d) / (2 * blobs[b][2] * blobs[b][2] * .25));
        }
        a[y * w + x] = s > 1 ? 1 : s;
      }
    }
  }

  // center-crop the source to the chosen aspect (16:9 or 9:16)
  function cropRect(sw, sh) {
    const ar = S.orient === 'port' ? 9 / 16 : 16 / 9;
    let cw = sw, ch = sh;
    if (sw / sh > ar) cw = sh * ar; else ch = sw / ar;
    return { cw, ch, cx: (sw - cw) / 2, cy: (sh - ch) / 2 };
  }

  function grab() {
    let sw, sh;
    if (S.mode === 'video' && video && video.videoWidth) { sw = video.videoWidth; sh = video.videoHeight; }
    else if (S.mode === 'image' && image) { sw = image.naturalWidth; sh = image.naturalHeight; }
    else { sw = 960; sh = 540; }

    const { cw, ch, cx, cy } = cropRect(sw, sh);
    const w = Math.max(8, Math.round(cw / S.scale)), h = Math.max(8, Math.round(ch / S.scale));
    if (w !== W || h !== H) {
      W = w; H = h;
      lum = new Float32Array(W * H);
      err = new Float32Array(W * H);
      tmp = new Float32Array(W * H);
      src.width = W; src.height = H;
      out.width = W; out.height = H;
      outImg = octx.createImageData(W, H);
      $('r-res').textContent = `grid: ${W} x ${H}`;
    }

    if (S.mode === 'demo') { fillDemo(lum, W, H, performance.now()); return; }

    if (S.mirror) { sctx.save(); sctx.scale(-1, 1); sctx.drawImage(video, cx, cy, cw, ch, -W, 0, W, H); sctx.restore(); }
    else sctx.drawImage(S.mode === 'video' ? video : image, cx, cy, cw, ch, 0, 0, W, H);
    const d = sctx.getImageData(0, 0, W, H).data;
    for (let i = 0, p = 0; i < lum.length; i++, p += 4) {
      lum[i] = (d[p] * 0.2126 + d[p + 1] * 0.7152 + d[p + 2] * 0.0722) / 255;
    }
  }

  // every lit pixel is the TWINEM sparkle, tinted to the dot color
  const sprite = new Image();
  sprite.src = 'assets/twinem_pixel.svg';
  let spriteOK = false;
  sprite.onload = () => { spriteOK = true; patFor = ''; };
  const bit = document.createElement('canvas'), bctx = bit.getContext('2d');
  const glyph = document.createElement('canvas'), gctx = glyph.getContext('2d');
  let bitImg = null, pat = null, patFor = '';

  function ensurePattern(cell) {
    if (!spriteOK) return null;
    const want = S.fgc + '@' + cell;
    if (pat && patFor === want) return pat;
    const t = document.createElement('canvas');
    t.width = t.height = cell;
    const tc = t.getContext('2d');
    tc.drawImage(sprite, 0, 0, cell, cell);
    tc.globalCompositeOperation = 'source-in';
    tc.fillStyle = S.fgc;
    tc.fillRect(0, 0, cell, cell);
    pat = gctx.createPattern(t, 'repeat');
    patFor = want;
    return pat;
  }

  function paint() {
    const cell = Math.max(2, Math.min(24, Math.round(1600 / W)));
    const ow = W * cell, oh = H * cell;
    if (out.width !== ow || out.height !== oh) { out.width = ow; out.height = oh; }
    if (glyph.width !== ow || glyph.height !== oh) { glyph.width = ow; glyph.height = oh; }
    if (bit.width !== W || bit.height !== H) { bit.width = W; bit.height = H; bitImg = bctx.createImageData(W, H); }
    const d = bitImg.data;
    for (let i = 0, p = 0; i < lum.length; i++, p += 4) {
      d[p] = 255; d[p + 1] = 255; d[p + 2] = 255; d[p + 3] = lum[i] ? 255 : 0;
    }
    bctx.putImageData(bitImg, 0, 0);
    gctx.clearRect(0, 0, ow, oh);
    gctx.fillStyle = ensurePattern(cell) || S.fgc;
    gctx.fillRect(0, 0, ow, oh);
    gctx.globalCompositeOperation = 'destination-in';
    gctx.imageSmoothingEnabled = false;
    gctx.drawImage(bit, 0, 0, ow, oh);
    gctx.globalCompositeOperation = 'source-over';
    octx.fillStyle = S.bgc;
    octx.fillRect(0, 0, ow, oh);
    octx.drawImage(glyph, 0, 0);
  }

  const hexCache = {};
  function hex(s) {
    if (!hexCache[s]) hexCache[s] = [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
    return hexCache[s];
  }

  let lastTick = 0;
  function frame() {
    requestAnimationFrame(frame);
    const now = performance.now();
    if (now - lastTick < 1000 / 30) return;   // locked to 30 fps, pixels do not need more
    lastTick = Math.max(lastTick + 1000 / 30, now - 1000 / 30);   // even pacing on 60/120 Hz displays
    frames++;
    if (now - fpsT > 500) {
      $('r-fps').textContent = Math.round(frames / ((now - fpsT) / 1000)) + ' fps';
      frames = 0; fpsT = now;
    }
    if (S.mode === 'image' || S.exporting) return;
    if (S.mode === 'video') syncTransport();
    if (++holdCount < S.hold) return;
    holdCount = 0;
    render();
  }

  function render() {
    const t0 = performance.now();
    grab();
    boxBlur(lum, W, H, Math.round(S.blur));
    levels(lum);
    dither(lum, W, H);
    paint();
    lastMs = performance.now() - t0;
    $('r-ms').textContent = `pass: ${lastMs.toFixed(1)} ms`;
  }

  // ---- wiring ----------------------------------------------------------
  const pr = $('presets'), sw = $('swatches');
  let applying = false;

  function clearPreset() {
    if (applying) return;
    [...pr.children].forEach(c => c.setAttribute('aria-pressed', 'false'));
  }

  function bind(id, key, fn, label, fmt) {
    const el = $(id), fill = el.closest('.slider').querySelector('.fill');
    const upd = () => {
      const n = +el.value;
      S[key] = fn(n);
      fill.style.setProperty('--fill', ((n - el.min) / (el.max - el.min) * 100) + '%');
      if (label) $(label).textContent = fmt(S[key]);
      clearPreset();
      if (S.mode === 'image') render();
    };
    el.addEventListener('input', upd);
    upd();
  }

  bind('diff', 'diff', n => n / 100, 'v-diff', v => Math.round(v * 100) + '%');
  bind('blur', 'blur', n => n / 10, 'v-blur', v => v.toFixed(1));
  bind('scale', 'scale', n => n, 'v-scale', v => v + ' px');
  bind('con', 'con', n => n / 100, 'v-con', v => v.toFixed(2));
  bind('mid', 'mid', n => n / 100, 'v-mid', v => v.toFixed(2));
  bind('thr', 'thr', n => n / 100, 'v-thr', v => v.toFixed(2));
  bind('hold', 'hold', n => n, 'v-hold', v => v === 1 ? '1 (full)' : v + ' (' + Math.round(60 / v) + 'fps)');

  $('serp').addEventListener('change', e => { S.serp = e.target.checked; clearPreset(); if (S.mode === 'image') render(); });
  $('inv').addEventListener('change', e => { S.inv = e.target.checked; clearPreset(); if (S.mode === 'image') render(); });

  $('algo').addEventListener('change', e => {
    S.algo = e.target.value;
    $('algo-note').textContent = NOTES[S.algo];
    clearPreset();
    if (S.mode === 'image') render();
  });
  $('algo-note').textContent = NOTES[S.algo];

  // primary = the dots, secondary = the background. Never the same color:
  // picking a conflict swaps the two.
  const COLORS = [['Black', '#000000'], ['White', '#FFFFFF'], ['Pink', '#E4175E'], ['Teal', '#3AC0C3']];
  const FG_OPTS = ['#FFFFFF', '#E4175E', '#3AC0C3'];
  const BG_OPTS = ['#000000', '#FFFFFF', '#E4175E', '#3AC0C3'];
  function buildColorRow(label, opts, key, otherKey) {
    const row = document.createElement('div');
    row.className = 'colorrow';
    if (label) {
      const cap = document.createElement('b');
      cap.textContent = label;
      row.appendChild(cap);
    }
    opts.forEach(hexv => {
      const b = document.createElement('button');
      b.className = 'swatch mono';
      b.title = (COLORS.find(c => c[1] === hexv) || ['?'])[0];
      b.dataset.c = hexv;
      b.innerHTML = `<i style="background:${hexv}"></i>`;
      b.onclick = () => {
        if (S[otherKey] === hexv) { S[otherKey] = S[key]; }  // swap, never equal
        S[key] = hexv;
        syncColorRows();
        if (S.mode === 'image') render();
      };
      row.appendChild(b);
    });
    sw.appendChild(row);
  }
  function syncColorRows() {
    [...sw.querySelectorAll('.colorrow')].forEach((row, ri) => {
      const key = ri === 0 ? 'fgc' : 'bgc';
      [...row.querySelectorAll('.swatch')].forEach(b =>
        b.setAttribute('aria-pressed', b.dataset.c === S[key]));
    });
  }
  buildColorRow('', FG_OPTS, 'fgc', 'bgc');

  syncColorRows();

  PRESETS.forEach(([name, p, note]) => {
    const b = document.createElement('button');
    b.textContent = name;
    b.setAttribute('aria-pressed', 'false');
    b.onclick = () => {
      applying = true;
      ['diff', 'blur', 'scale', 'con', 'mid', 'thr'].forEach(id => {
        $(id).value = p[id];
        $(id).dispatchEvent(new Event('input'));
      });
      $('algo').value = p.algo; $('algo').dispatchEvent(new Event('change'));
      $('serp').checked = p.serp; S.serp = p.serp;
      $('inv').checked = p.inv; S.inv = p.inv;
      S.fgc = '#FFFFFF'; S.bgc = '#000000'; syncColorRows();
      applying = false;
      [...pr.children].forEach(c => c.setAttribute('aria-pressed', c === b));
      $('preset-note').textContent = note;
      if (S.mode === 'image') render();
    };
    pr.appendChild(b);
  });

  function markPreset(i) {
    [...pr.children].forEach((c, j) => c.setAttribute('aria-pressed', j === i));
    if (PRESETS[i]) $('preset-note').textContent = PRESETS[i][2];
  }

  // the app loads in the saved look, so show it as active
  markPreset(0);

  function setMode(m, label) {
    S.mode = m;
    $('r-src').textContent = 'source: ' + label;
    // source buttons latch: exactly one of LOOP / CAM / FILE stays pressed
    $('demo').setAttribute('aria-pressed', label === 'logo loop');
    $('cam').setAttribute('aria-pressed', label === 'webcam');
    const fileBtn = document.querySelector('label[for="file"]');
    if (fileBtn) fileBtn.setAttribute('aria-pressed', label !== 'logo loop' && label !== 'webcam');
    W = H = 0;
    // the demo loop just loops; transport is only for footage they bring
    showTransport(m === 'video' && label !== 'logo loop' && video && isFinite(video.duration));
    if (m === 'image') render();
    requestAnimationFrame(sizeLabel);
  }

  function stopCam() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }

  $('demo').onclick = () => { stopCam(); loadURL('assets/twinem_stainless_loop.mp4', 'logo loop', true); };

  // orientation is automatic: the window's shape decides TV vs MOBILE
  const syncOrient = () => {
    document.querySelector('.tv').classList.toggle('portrait', S.orient === 'port');
    document.title = S.orient === 'port' ? 'TWINEM MOBILE' : 'TWINEM TV';
  };
  const followMq = e => { S.orient = e.matches ? 'port' : 'land'; W = H = 0; syncOrient(); };
  const mq = matchMedia('(orientation: portrait)');
  if (mq.addEventListener) mq.addEventListener('change', followMq); else mq.addListener(followMq);
  let rzT = null;
  addEventListener('resize', () => {
    clearTimeout(rzT);
    rzT = setTimeout(() => followMq({ matches: innerHeight > innerWidth }), 120);
  });
  addEventListener('orientationchange', () => followMq({ matches: innerHeight > innerWidth }));
  followMq({ matches: innerHeight > innerWidth });

  function loadURL(url, label, isVideo, serverPath) {
    stopCam();
    S.mirror = false;
    S.serverPath = serverPath || null;
    if (isVideo) {
      video = document.createElement('video');
      video.src = url; video.muted = true; video.playsInline = true; video.loop = true;
      video.onloadeddata = () => {
        setMode('video', label);
        video.play().catch(() => {});   // everything runs at a flat 30 fps, no detection
      };
      video.addEventListener('timeupdate', syncTransport);
      video.addEventListener('play', syncTransport);
      video.addEventListener('pause', syncTransport);
      video.addEventListener('ended', () => {
        if (!S.exporting) { video.currentTime = 0; video.play().catch(() => {}); }
      });
    } else {
      image = new Image();
      image.onload = () => setMode('image', label);
      image.onerror = () => { $('r-src').textContent = 'source: could not load ' + label; };
      image.src = url;
    }
  }

  function loadFile(f) {
    if (!f) return;
    loadURL(URL.createObjectURL(f), f.name, f.type.startsWith('video'));
  }

  $('file').addEventListener('change', e => loadFile(e.target.files[0]));

  // drop anywhere on the stage
  const stage = document.querySelector('.stage');
  stage.addEventListener('dragover', e => { e.preventDefault(); stage.classList.add('drop'); });
  stage.addEventListener('dragleave', () => stage.classList.remove('drop'));
  stage.addEventListener('drop', e => {
    e.preventDefault();
    stage.classList.remove('drop');
    loadFile(e.dataTransfer.files[0]);
  });

  // ?img=path or ?video=path, relative to this server
  (function fromQuery() {
    const q = new URLSearchParams(location.search);
    const img = q.get('img'), vid = q.get('video');
    if (img) loadURL(img, img, false);
    else if (vid) loadURL(vid, vid, true, vid);  // server-side path, so audio can be carried through
    else loadURL('assets/twinem_stainless_loop.mp4', 'logo loop', true);  // TWINEM: their chrome IS the demo
  })();

  $('cam').onclick = async () => {
    try {
      stopCam();
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      video = document.createElement('video');
      video.srcObject = stream; video.muted = true; video.playsInline = true;
      await video.play();
      S.mirror = true;   // selfie view: mirror like a mirror
      setMode('video', 'webcam');
    } catch (err) {
      $('r-src').textContent = 'source: webcam blocked. check browser permissions.';
    }
  };

  $('png').onclick = () => {
    if (S.mode === 'image') render();
    const a = document.createElement('a');
    a.download = `dither_${S.algo}_${Date.now()}.png`;
    a.href = out.toDataURL('image/png');
    a.click();
  };

  $('reset').onclick = () => {
    applying = true;
    for (const id in DEFAULTS) { $(id).value = DEFAULTS[id]; $(id).dispatchEvent(new Event('input')); }
    $('algo').value = DEFAULT_ALGO; $('algo').dispatchEvent(new Event('change'));
    $('serp').checked = false; S.serp = false;
    $('inv').checked = false; S.inv = false;
    S.fgc = '#FFFFFF'; S.bgc = '#000000'; syncColorRows();
    [...sw.children].forEach((c, j) => c.setAttribute('aria-pressed', j === 0));
    applying = false;
    markPreset(0);
    if (S.mode === 'image') render();
  };

  // ---- transport -------------------------------------------------------
  const tr = $('transport'), seekEl = $('seek'), seekFill = tr.querySelector('.fill');

  function tc(s) {
    if (!isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60), r = s - m * 60;
    return m + ':' + (r < 10 ? '0' : '') + r.toFixed(2);
  }

  function showTransport(on) {
    tr.hidden = !on;
    if (on) syncTransport();
  }

  let scrubbing = false;

  function syncTransport() {
    if (!video || !isFinite(video.duration)) return;
    $('tc-cur').textContent = tc(video.currentTime);
    $('tc-dur').textContent = tc(video.duration);
    $('play').textContent = video.paused ? 'Play' : 'Pause';
    if (!scrubbing) {
      const f = video.currentTime / video.duration;
      seekEl.value = Math.round(f * 10000);
      seekFill.style.setProperty('--fill', (f * 100) + '%');
    }
  }

  $('play').onclick = () => {
    if (!video || S.exporting) return;
    if (video.paused) video.play().catch(() => {}); else video.pause();
    syncTransport();
  };

  seekEl.addEventListener('pointerdown', () => { scrubbing = true; });
  seekEl.addEventListener('pointerup', () => { scrubbing = false; });
  seekEl.addEventListener('input', () => {
    if (!video || !isFinite(video.duration)) return;
    const f = seekEl.value / 10000;
    seekFill.style.setProperty('--fill', (f * 100) + '%');
    video.currentTime = f * video.duration;
    $('tc-cur').textContent = tc(video.currentTime);
  });

  // ---- frame rate ------------------------------------------------------
  const RATES = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60];

  // browsers do not expose the source frame rate, so measure it from a
  // handful of presented frames and snap to the nearest standard rate.
  function detectFps(v) {
    if (!v.requestVideoFrameCallback) { v.play().catch(() => {}); return; }
    const times = [];
    let last = null;
    const step = (_, meta) => {
      if (last !== null && meta.mediaTime > last) times.push(meta.mediaTime - last);
      last = meta.mediaTime;
      if (times.length < 20 && !v.paused) v.requestVideoFrameCallback(step);
      else finish();
    };
    let done = false;
    function finish() {
      if (done) return;
      done = true;
      v.currentTime = 0;
      if (!S.exporting) v.play().catch(() => {});
      if (times.length < 5) return;
      times.sort((a, b) => a - b);
      const med = times[times.length >> 1];
      if (!med) return;
      const guess = 1 / med;
      const best = RATES.reduce((a, b) => Math.abs(b - guess) < Math.abs(a - guess) ? b : a);
      if (Math.abs(best - guess) / best < 0.05) {
        $('fps').value = String(best);
        $('v-fps').textContent = best + ' (detected)';
      }
    }
    v.muted = true;
    v.play().then(() => v.requestVideoFrameCallback(step)).catch(() => {});
    setTimeout(finish, 3000);
  }

  $('fps').addEventListener('change', () => { $('v-fps').textContent = $('fps').value; });

  function sizeLabel() {
    const srcW = video && video.videoWidth, srcH = video && video.videoHeight;
    let sW = srcW, sH = srcH;
    if (srcW && srcH) { const c = cropRect(srcW, srcH); sW = Math.round(c.cw); sH = Math.round(c.ch); }
    const useSrc = $('outsize').value === 'source';
    const w = useSrc && sW ? sW : W, h = useSrc && sH ? sH : H;
    $('v-size').textContent = w && h ? `${w} x ${h}` : '-';
    return { w, h };
  }
  $('outsize').addEventListener('change', sizeLabel);

  // ---- export ----------------------------------------------------------
  const prog = $('progress'), barFill = $('bar-fill');
  let cancelExport = false;

  function seekTo(v, t) {
    return new Promise(res => {
      if (Math.abs(v.currentTime - t) < 1e-4) return res();
      let settled = false;
      const done = () => { if (settled) return; settled = true; v.removeEventListener('seeked', done); res(); };
      v.addEventListener('seeked', done);
      setTimeout(done, 2000);
      v.currentTime = t;
    });
  }

  const api = (path, body, raw) => fetch(path, {
    method: 'POST',
    body: raw ? body : JSON.stringify(body || {})
  }).then(r => raw ? r : r.json());

  async function runExport() {
    if (S.mode !== 'video' || !video || !isFinite(video.duration)) {
      $('export-note').textContent = 'Load a video first. Export walks a clip frame by frame, so there is nothing to walk on a still or the demo field.';
      return;
    }

    const fps = parseFloat($('fps').value);
    const { w, h } = sizeLabel();
    const total = Math.max(1, Math.round(video.duration * fps));

    video.pause();
    S.exporting = true;
    cancelExport = false;
    prog.hidden = false;
    $('export').setAttribute('aria-pressed', 'true');

    let job = null;
    const t0 = performance.now();

    try {
      const started = await api('/api/export/start', {
        fps, w, h,
        codec: $('codec').value,
        algo: S.algo,
        audio: S.serverPath || null
      });
      if (started.error) throw new Error(started.error);
      job = started.job;

      for (let i = 0; i < total; i++) {
        if (cancelExport) break;
        await seekTo(video, Math.min(i / fps, video.duration - 1e-3));
        render();
        const blob = await new Promise(r => out.toBlob(r, 'image/png'));
        const res = await api('/api/export/frame?job=' + job, blob, true);
        if (!res.ok) throw new Error((await res.json()).error || 'frame rejected');

        const done = i + 1, frac = done / total;
        barFill.style.width = (frac * 100) + '%';
        const rate = done / ((performance.now() - t0) / 1000);
        const left = rate > 0 ? (total - done) / rate : 0;
        $('prog-text').textContent =
          `${done} / ${total} frames  ${tc(left)} left`;
        $('tc-cur').textContent = tc(video.currentTime);
      }

      if (cancelExport) {
        await api('/api/export/cancel?job=' + job);
        $('export-note').textContent = 'Export cancelled. Nothing written.';
      } else {
        $('prog-text').textContent = 'Encoding';
        const fin = await api('/api/export/finish?job=' + job);
        if (fin.ok) {
          const mb = (fin.size / 1048576).toFixed(1);
          $('export-note').textContent =
            `Saved ${fin.frames} frames at ${w} x ${h}, ${mb} MB. Revealed in Finder: ${fin.path}`;
        } else {
          $('export-note').textContent = 'ffmpeg failed: ' + (fin.log || '').slice(-300);
        }
      }
    } catch (e) {
      if (job) await api('/api/export/cancel?job=' + job).catch(() => {});
      $('export-note').textContent = 'Export failed: ' + e.message +
        '. This needs server.py running, not a plain static server.';
    } finally {
      S.exporting = false;
      prog.hidden = true;
      barFill.style.width = '0';
      $('export').setAttribute('aria-pressed', 'false');
      syncTransport();
    }
  }

  $('export').onclick = () => { if (!S.exporting) runExport(); };
  $('cancel').onclick = () => { cancelExport = true; };

  frame();
})();
