/* AXON: cómo aprende una red neuronal.
   Dibujo e interacción con D3 v7 (escalas, zoom, transiciones, quadtree, blur2, ejes, formas).
   El canvas solo es la superficie para 1.000 puntos con brillo y miles de trazos. */
(() => {
  "use strict";
  const $ = s => document.querySelector(s);
  const PALETTE = ["#5aa9ff", "#ff7f6e", "#7ee08a", "#ff4f8b", "#b18cff",
                   "#c9a27e", "#f78fe0", "#b9c4cf", "#e9e26a", "#3fd6c4"];
  const HOLO = "#6fe3ff", CORE = "#ffb454", VOID = "#03080f", INK = "#e4f2f9", MUTE = "#8ea8ba", FAINT = "#56718a";
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const LOCALE = d3.formatLocale({ decimal: ",", thousands: ".", grouping: [3], currency: ["S/ ", ""] });
  const num = LOCALE.format(",d");
  const pct = (v, d = 1) => v == null || Number.isNaN(v) ? "sin dato" : `${LOCALE.format(`.${d}f`)(v * 100)} %`;
  const QUESTIONS = {
    epochs: { code: "Pregunta 1 de 4", title: "¿En qué época aparece la estructura de clases?" },
    layers: { code: "Pregunta 2 de 4", title: "¿En qué capa se separan las clases?" },
    confusion: { code: "Pregunta 3 de 4", title: "¿Qué clases confunde la red y qué imágenes hacen de puente?" },
    hard: { code: "Pregunta 4 de 4", title: "¿Qué imágenes le cuestan y cómo cambia de opinión?" },
  };

  const cache = {};
  const S = {
    key: null, d: null, axis: "epochs", fixedLayer: 0, fixedEpoch: 0, steps: [], pos: 0,
    q: "epochs", color: "class", hidden: new Set(), errors: false, bundles: false, strength: .5,
    photos: true, links: true, hovered: null, pinned: null, highlight: null,
    t: d3.zoomIdentity, cur: null, anim: null, playing: false, playTimer: null, trails: null,
    quad: null, quadDirty: true, neuronLayer: null,
  };

  /* ================================================================ datos */
  let ASPECT = 1.6;
  /* t-SNE no tiene escala absoluta: cada etapa se centra y se ajusta al área de dibujo (percentiles 1 y 99). */
  function fitFrame(xy) {
    const n = xy.length, out = new Float32Array(n * 2);
    const xs = xy.map(p => p[0]).sort(d3.ascending), ys = xy.map(p => p[1]).sort(d3.ascending);
    const x0 = d3.quantile(xs, .01), x1 = d3.quantile(xs, .99), y0 = d3.quantile(ys, .01), y1 = d3.quantile(ys, .99);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const s = .94 / Math.max((x1 - x0) / 2 / ASPECT, (y1 - y0) / 2, 1e-9);
    for (let i = 0; i < n; i++) { out[i * 2] = (xy[i][0] - cx) * s; out[i * 2 + 1] = (xy[i][1] - cy) * s; }
    return out;
  }
  async function load(key) {
    if (cache[key]) return cache[key];
    const base = `/data/${key}/`;
    const [meta, nodes, frames, edges, neurons] = await Promise.all(
      ["meta", "nodes", "frames", "edges", "neurons"].map(n => d3.json(`${base}${n}.json`)));
    const atlas = new Image();
    atlas.src = `${base}atlas.png`;
    await atlas.decode();
    for (const f of Object.values(frames)) f.xyF = fitFrame(f.xy);
    cache[key] = { meta, nodes, frames, edges, neurons, atlas, n: nodes.length, aspect: ASPECT };
    return cache[key];
  }
  function refit() {
    if (!S.d || Math.abs(S.d.aspect - ASPECT) < .02) return;
    for (const f of Object.values(S.d.frames)) f.xyF = fitFrame(f.xy);
    S.d.aspect = ASPECT;
    if (S.cur && !S.anim) S.cur = Float32Array.from(cell().xyF);
    computeTrails();
  }

  const cellKey = (s, l) => `${s}:${l}`;
  const cellAt = i => S.d.frames[S.steps[i]?.key];
  const cell = () => cellAt(S.pos);
  const step = () => S.steps[S.pos];
  const predOf = c => (S.axis === "layers" && c.probe) ? c.probe : c.pred;
  const layerName = name => name.replace(" (píxeles)", "");

  function buildSteps() {
    const { meta, frames } = S.d;
    S.steps = S.axis === "epochs"
      ? meta.epochs.map((e, s) => ({ s, l: S.fixedLayer, key: cellKey(s, S.fixedLayer), label: e.label,
          tick: String(e.epoch), epoch: e.epoch, best: e.epoch === meta.bestEpoch })).filter(x => frames[x.key])
      : meta.layers.map((name, l) => ({ s: S.fixedEpoch, l, key: cellKey(S.fixedEpoch, l), label: name,
          tick: shortLayer(name), best: false })).filter(x => frames[x.key]);
  }
  function shortLayer(name) {
    if (name.startsWith("Entrada")) return "Entrada";
    if (name.startsWith("Salida")) return "Salida";
    return name.replace(/\s*\(.*\)/, "").replace("Bloque conv.", "Bloque");
  }
  function fixedOptions(axis) {
    const { meta, frames } = S.d;
    if (axis === "epochs") {
      return meta.layers.map((name, l) => ({ v: l, label: layerName(name),
        ok: meta.epochs.filter((_, s) => frames[cellKey(s, l)]).length > 1 }));
    }
    return meta.epochs.map((e, s) => ({ v: s, label: e.label + (e.epoch === meta.bestEpoch ? " (mejor)" : ""),
      ok: meta.layers.filter((_, l) => frames[cellKey(s, l)]).length > 1 }));
  }

  /* ================================================================ escena */
  const holo = $("#holo"), ctx = holo.getContext("2d");
  const overlay = d3.select("#overlay");
  const deco = overlay.append("g");
  const gLabels = overlay.append("g");
  const gTrail = overlay.append("g");
  const gLinks = overlay.append("g");
  const gMarks = overlay.append("g");
  let W = 0, H = 0, DPR = 1, PLOT = { cx: 0, cy: 0, rx: 0, ry: 0 };
  const bx = d3.scaleLinear(), by = d3.scaleLinear();
  const sx = x => S.t.applyX(bx(x));
  const sy = y => S.t.applyY(by(y));

  const zoom = d3.zoom().scaleExtent([.7, 14]).on("zoom", ev => {
    S.t = ev.transform; S.quadDirty = true;
    $("#viewport").classList.toggle("zoomed", S.t.k > 1.6);
    $("#zoom-read").textContent = `${LOCALE.format(".1f")(S.t.k)}×`;
    draw();
  });
  overlay.call(zoom).on("dblclick.zoom", null);

  function resize() {
    const r = $("#viewport").getBoundingClientRect();
    W = Math.max(240, r.width); H = Math.max(240, r.height);
    DPR = Math.min(2, window.devicePixelRatio || 1);
    holo.width = W * DPR; holo.height = H * DPR;
    const wide = W > 820;
    const m = { l: wide ? 168 : 24, r: wide ? 178 : 24, t: wide ? 122 : 110, b: 74 };
    PLOT = { cx: (m.l + W - m.r) / 2, cy: (m.t + H - m.b) / 2, rx: (W - m.l - m.r) / 2, ry: (H - m.t - m.b) / 2 };
    ASPECT = Math.max(.8, PLOT.rx / PLOT.ry);
    bx.domain([-ASPECT, ASPECT]).range([PLOT.cx - PLOT.rx, PLOT.cx + PLOT.rx]);
    by.domain([-1, 1]).range([PLOT.cy + PLOT.ry, PLOT.cy - PLOT.ry]);
    zoom.extent([[0, 0], [W, H]]).translateExtent([[-W, -H], [2 * W, 2 * H]]);
    drawDeco();
    buildSprites();
    refit();
    S.quadDirty = true;
    draw();
    if (S.d) renderTrack();
  }

  function drawDeco() {
    deco.selectAll("*").remove();
    const R = Math.min(PLOT.ry + 34, PLOT.rx + 34), c = deco.append("g").attr("transform", `translate(${PLOT.cx},${PLOT.cy})`);
    c.append("circle").attr("class", "ring").attr("r", R);
    c.append("circle").attr("class", "ring dash").attr("r", R * .62);
    const spin = c.append("g").attr("class", "spin");
    spin.selectAll("line").data(d3.range(0, 360, 6)).join("line").attr("class", "tick")
      .attr("x1", a => Math.cos(a * Math.PI / 180) * R).attr("y1", a => Math.sin(a * Math.PI / 180) * R)
      .attr("x2", a => Math.cos(a * Math.PI / 180) * (R + (a % 90 === 0 ? 10 : 4)))
      .attr("y2", a => Math.sin(a * Math.PI / 180) * (R + (a % 90 === 0 ? 10 : 4)));
  }

  let sprites = {};
  function sprite(color) {
    const s = 48, cv = document.createElement("canvas");
    cv.width = cv.height = s;
    const g = cv.getContext("2d"), grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2), c = d3.color(color);
    grd.addColorStop(0, c.copy({ opacity: .85 }).formatRgb());
    grd.addColorStop(.22, c.copy({ opacity: .32 }).formatRgb());
    grd.addColorStop(1, c.copy({ opacity: 0 }).formatRgb());
    g.fillStyle = grd; g.fillRect(0, 0, s, s);
    return cv;
  }
  function buildSprites() { sprites = {}; [...PALETTE, HOLO, CORE, "#2b4458", "#9fb6c4"].forEach(c => { sprites[c] = sprite(c); }); }

  /* ================================================================ color y visibilidad */
  const keepScale = d3.scaleLinear().domain([0, 5, 10]).range([CORE, "#9fb6c4", HOLO]).clamp(true);
  function colorOf(i, c) {
    const node = S.d.nodes[i];
    if (S.color === "correct") return predOf(c)[i] === node.label ? HOLO : CORE;
    if (S.color === "keep") { const k = c.keep ? c.keep[i] : 5; return k >= 7 ? HOLO : k <= 3 ? CORE : "#9fb6c4"; }
    return PALETTE[node.label];
  }
  function visible(i, c) {
    const node = S.d.nodes[i];
    if (S.hidden.has(node.label)) return false;
    return !(S.errors && predOf(c)[i] === node.label);
  }
  const emphasized = i => !S.highlight || S.highlight.set.has(i);

  /* ================================================================ haces (KDE edge bundling) */
  function computeTrails() {
    if (!S.bundles || !S.d || S.steps.length < 2) { S.trails = null; return; }
    const c = cell(), K = S.steps.length, SEG = 6, L = (K - 1) * SEG + 1;
    const ids = d3.range(S.d.n).filter(i => visible(i, c));
    const P = new Float32Array(ids.length * L * 2);
    ids.forEach((i, p) => {
      for (let k = 0; k < K - 1; k++) {
        const a = cellAt(k).xyF, b = cellAt(k + 1).xyF;
        for (let j = 0; j < SEG; j++) {
          const t = j / SEG, o = (p * L + k * SEG + j) * 2;
          P[o] = a[i * 2] + (b[i * 2] - a[i * 2]) * t;
          P[o + 1] = a[i * 2 + 1] + (b[i * 2 + 1] - a[i * 2 + 1]) * t;
        }
      }
      const z = cellAt(K - 1).xyF, o = (p * L + L - 1) * 2;
      P[o] = z[i * 2]; P[o + 1] = z[i * 2 + 1];
    });
    const G = 160, lo = -Math.max(1.25, ASPECT * 1.1), size = -2 * lo / G, dens = new Float32Array(G * G);
    for (let it = 0; it < 9 && S.strength > 0; it++) {
      const r = Math.max(1.2, 5 - it * .45);
      dens.fill(0);
      for (let q = 0; q < P.length; q += 2) {
        const gx = ((P[q] - lo) / size) | 0, gy = ((P[q + 1] - lo) / size) | 0;
        if (gx >= 0 && gy >= 0 && gx < G && gy < G) dens[gy * G + gx] += 1;
      }
      d3.blur2({ data: dens, width: G, height: G }, r);
      const len = S.strength * size * r * .45;
      for (let p = 0; p < ids.length; p++) {
        for (let j = 1; j < L - 1; j++) {
          if (j % SEG === 0) continue;
          const o = (p * L + j) * 2;
          const gx = Math.min(G - 2, Math.max(1, ((P[o] - lo) / size) | 0));
          const gy = Math.min(G - 2, Math.max(1, ((P[o + 1] - lo) / size) | 0));
          const dx = dens[gy * G + gx + 1] - dens[gy * G + gx - 1], dy = dens[(gy + 1) * G + gx] - dens[(gy - 1) * G + gx];
          const h = Math.hypot(dx, dy);
          if (h > 1e-9) { P[o] += len * dx / h; P[o + 1] += len * dy / h; }
        }
      }
      for (let pass = 0; pass < 2; pass++) {
        for (let p = 0; p < ids.length; p++) {
          for (let j = 1; j < L - 1; j++) {
            if (j % SEG === 0) continue;
            const o = (p * L + j) * 2;
            P[o] = P[o] * .5 + (P[o - 2] + P[o + 2]) * .25;
            P[o + 1] = P[o + 1] * .5 + (P[o - 1] + P[o + 3]) * .25;
          }
        }
      }
    }
    S.trails = { ids, P, L, SEG, K, index: new Map(ids.map((i, p) => [i, p])) };
  }

  /* ================================================================ dibujo */
  function draw(motion = null) {
    if (!S.d || !S.cur) return;
    const c = cell(), n = S.d.n;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (S.trails) drawTrails();
    const k = S.t.k, photo = S.photos && k >= 3.2, r = Math.min(6.5, 2.3 + Math.sqrt(k) * .85);

    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < n; i++) {
      if (!visible(i, c)) continue;
      const x = sx(S.cur[i * 2]), y = sy(S.cur[i * 2 + 1]);
      if (x < -20 || y < -20 || x > W + 20 || y > H + 20) continue;
      const em = emphasized(i), col = em ? colorOf(i, c) : "#2b4458", g = r * 4.2;
      ctx.globalAlpha = em ? .5 : .22;
      ctx.drawImage(sprites[col], x - g / 2, y - g / 2, g, g);
    }
    ctx.globalCompositeOperation = "source-over";
    if (motion) {
      ctx.lineWidth = 1.1;
      for (let i = 0; i < n; i++) {
        if (!visible(i, c) || !emphasized(i)) continue;
        ctx.strokeStyle = colorOf(i, c); ctx.globalAlpha = .42;
        ctx.beginPath(); ctx.moveTo(sx(motion[i * 2]), sy(motion[i * 2 + 1])); ctx.lineTo(sx(S.cur[i * 2]), sy(S.cur[i * 2 + 1])); ctx.stroke();
      }
    }
    const A = S.d.meta.atlas;
    ctx.imageSmoothingEnabled = false;
    for (let i = 0; i < n; i++) {
      if (!visible(i, c)) continue;
      const x = sx(S.cur[i * 2]), y = sy(S.cur[i * 2 + 1]);
      if (x < -60 || y < -60 || x > W + 60 || y > H + 60) continue;
      const em = emphasized(i);
      if (photo && em) {
        const s = Math.min(60, 5.2 * k);
        ctx.globalAlpha = 1;
        ctx.drawImage(S.d.atlas, (i % A.cols) * A.w, Math.floor(i / A.cols) * A.h, A.w, A.h, x - s / 2, y - s / 2, s, s);
        ctx.strokeStyle = colorOf(i, c); ctx.lineWidth = 1.5; ctx.strokeRect(x - s / 2, y - s / 2, s, s);
      } else {
        ctx.globalAlpha = em ? .95 : .28;
        ctx.fillStyle = em ? colorOf(i, c) : "#35526a";
        ctx.beginPath(); ctx.arc(x, y, em ? r * .55 : r * .38, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    drawLabels();
    drawMarks();
  }

  function drawTrails() {
    const T = S.trails, c = cell(), K = T.K, base = Math.min(.15, 26 / T.ids.length + .03);
    ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = 1;
    T.ids.forEach((i, p) => {
      if (S.highlight && !emphasized(i)) return;
      ctx.strokeStyle = colorOf(i, c);
      for (let k = 0; k < K - 1; k++) {
        ctx.globalAlpha = base * (S.highlight ? 2.2 : 1) * (.3 + .7 * (k + 1) / (K - 1));
        ctx.beginPath();
        for (let j = k * T.SEG; j <= (k + 1) * T.SEG; j++) {
          const o = (p * T.L + j) * 2, x = sx(T.P[o]), y = sy(T.P[o + 1]);
          j === k * T.SEG ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    });
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }

  /* rótulos directos: el nombre de la clase aparece sobre su grupo cuando el grupo es compacto */
  function drawLabels() {
    const c = cell(), names = S.d.meta.classes, out = [];
    if (S.t.k < 2.6) {
      for (let k = 0; k < 10; k++) {
        if (S.hidden.has(k)) continue;
        const ids = S.d.nodes.filter(nd => nd.label === k && visible(nd.i, c)).map(nd => nd.i);
        if (ids.length < 12) continue;
        const mx = d3.median(ids, i => S.cur[i * 2]), my = d3.median(ids, i => S.cur[i * 2 + 1]);
        const spread = d3.median(ids, i => Math.hypot(S.cur[i * 2] - mx, S.cur[i * 2 + 1] - my));
        if (spread > .2) continue;
        const core = ids.filter(i => Math.hypot(S.cur[i * 2] - mx, S.cur[i * 2 + 1] - my) <= spread * 2.2);
        const top = d3.quantile(core.map(i => S.cur[i * 2 + 1]).sort(d3.ascending), .92);
        out.push({ k, x: sx(mx), y: sy(top) - 9, spread, n: ids.length });
      }
      out.sort((a, b) => a.spread - b.spread);
    }
    const placed = [];
    for (const l of out) {
      if (placed.some(p => Math.abs(p.x - l.x) < 74 && Math.abs(p.y - l.y) < 17)) continue;
      placed.push(l);
    }
    gLabels.selectAll("text").data(placed, d => d.k).join("text").attr("class", "cls-label")
      .attr("text-anchor", "middle").attr("x", d => d.x).attr("y", d => d.y)
      .attr("fill", d => PALETTE[d.k]).attr("opacity", S.highlight || S.pinned != null ? .3 : 1).text(d => names[d.k]);
  }

  function drawMarks() {
    if (!S.d || !S.cur) return;
    const c = cell(), P = S.pinned, Hn = S.hovered;
    const pt = i => [sx(S.cur[i * 2]), sy(S.cur[i * 2 + 1])];
    let trail = [];
    if (P != null && S.steps.length > 1 && !S.anim) {
      const T = S.trails && S.trails.index.has(P) ? S.trails : null;
      if (T) {
        const p = T.index.get(P);
        for (let j = 0; j < T.L; j++) trail.push([sx(T.P[(p * T.L + j) * 2]), sy(T.P[(p * T.L + j) * 2 + 1])]);
      } else trail = S.steps.map((_, k) => [sx(cellAt(k).xyF[P * 2]), sy(cellAt(k).xyF[P * 2 + 1])]);
    }
    const col = P != null ? PALETTE[S.d.nodes[P].label] : HOLO;
    gTrail.selectAll("path").data(trail.length ? [trail] : []).join("path")
      .attr("d", d3.line().curve(d3.curveCatmullRom.alpha(.5))).attr("fill", "none")
      .attr("stroke", col).attr("stroke-width", 2).attr("stroke-opacity", .9);
    const stops = trail.length ? S.steps.map((st, k) => ({ st, k, xy: [sx(cellAt(k).xyF[P * 2]), sy(cellAt(k).xyF[P * 2 + 1])] })) : [];
    gTrail.selectAll("circle").data(stops).join("circle").attr("class", "trail-dot")
      .attr("cx", d => d.xy[0]).attr("cy", d => d.xy[1]).attr("r", d => d.k === S.pos ? 0 : 3.3)
      .attr("fill", d => predOf(cellAt(d.k))[P] === S.d.nodes[P].label ? HOLO : CORE);
    const placedT = [];
    d3.sort(stops.filter(d => d.k !== S.pos), d => d.k === 0 ? 0 : d.k === S.steps.length - 1 ? 1 : 2 + Math.abs(d.k - S.pos))
      .forEach(d => { if (!placedT.some(o => Math.abs(o.xy[0] - d.xy[0]) < 30 && Math.abs(o.xy[1] - d.xy[1]) < 13)) placedT.push(d); });
    gTrail.selectAll("text").data(placedT).join("text").attr("class", "trail-label")
      .attr("x", d => d.xy[0] + 6).attr("y", d => d.xy[1] - 6).text(d => S.axis === "epochs" ? `É${d.st.tick}` : d.st.tick);

    const nb = P != null && S.links && c.neighbors ? c.neighbors[P] : [];
    gLinks.selectAll("line").data(nb).join("line")
      .attr("x1", P != null ? pt(P)[0] : 0).attr("y1", P != null ? pt(P)[1] : 0)
      .attr("x2", j => pt(j)[0]).attr("y2", j => pt(j)[1]).attr("stroke-width", 1)
      .attr("stroke", j => S.d.nodes[j].label === S.d.nodes[P].label ? HOLO : CORE).attr("stroke-opacity", .7)
      .attr("stroke-dasharray", j => S.d.nodes[j].label === S.d.nodes[P].label ? null : "3 3");

    const marks = [];
    if (P != null) marks.push({ type: "pin", i: P });
    if (Hn != null && Hn !== P) marks.push({ type: "hover", i: Hn });
    gMarks.selectAll("g.mark").data(marks, d => d.type).join(enter => enter.append("g").attr("class", "mark").each(function (d) {
      const s = d3.select(this);
      if (d.type === "pin") { s.append("circle").attr("class", "reticle").attr("r", 8); s.append("circle").attr("class", "reticle rot").attr("r", 13); }
      else { s.append("line").attr("class", "cross h"); s.append("line").attr("class", "cross v"); s.append("circle").attr("class", "hover-ring").attr("r", 7.5); }
    })).each(function (d) {
      const [x, y] = pt(d.i), s = d3.select(this);
      s.selectAll("circle").attr("cx", x).attr("cy", y).style("transform-origin", `${x}px ${y}px`);
      s.select(".h").attr("x1", 0).attr("x2", W).attr("y1", y).attr("y2", y);
      s.select(".v").attr("y1", 0).attr("y2", H).attr("x1", x).attr("x2", x);
    });
  }

  /* ================================================================ etapas y animación */
  function goTo(pos, animate = true) {
    pos = Math.max(0, Math.min(S.steps.length - 1, pos));
    const from = S.cur ? Float32Array.from(S.cur) : null;
    S.pos = pos;
    const target = cell().xyF;
    if (S.anim) { S.anim.stop(); S.anim = null; }
    computeTrails();
    if (!from || !animate || REDUCED) {
      S.cur = Float32Array.from(target); S.quadDirty = true;
      draw(); afterStage();
      if (S.playing) S.playTimer = setTimeout(nextPlay, 900);
      return;
    }
    const dur = 950, n = S.d.n, lag = new Float32Array(n * 2);
    S.anim = d3.timer(el => {
      const t = Math.min(1, el / dur), e = d3.easeCubicInOut(t), e2 = d3.easeCubicInOut(Math.max(0, t - .1));
      for (let q = 0; q < n * 2; q++) { S.cur[q] = from[q] + (target[q] - from[q]) * e; lag[q] = from[q] + (target[q] - from[q]) * e2; }
      draw(t < 1 ? lag : null);
      if (t >= 1) {
        S.anim.stop(); S.anim = null; S.quadDirty = true; draw();
        if (S.playing) S.playTimer = setTimeout(nextPlay, 700);
      }
    });
    afterStage();
  }

  function afterStage() {
    const st = step(), c = cell(), meta = S.d.meta;
    const other = S.axis === "epochs" ? `Capa ${layerName(meta.layers[S.fixedLayer])}` : meta.epochs[S.fixedEpoch].label;
    $("#stage-kicker").innerHTML = `<span>${meta.name}</span><span>${other}</span><span>${S.pos + 1}/${S.steps.length}</span>`;
    $("#stage-title").innerHTML = `${st.label}${st.best ? "<small>mejor validación</small>" : ""}`;
    const prev = S.pos > 0 ? cellAt(S.pos - 1) : null;
    let msg = `El <b>${pct(c.purity)}</b> de los vecinos de cada imagen comparte su clase`;
    if (prev) {
      const dp = (c.purity - prev.purity) * 100;
      msg += ` (${dp >= 0 ? "+" : "−"}${LOCALE.format(".1f")(Math.abs(dp))} puntos respecto a ${S.axis === "epochs" ? "la " + S.steps[S.pos - 1].label.toLowerCase() : layerName(S.steps[S.pos - 1].label)})`;
    }
    msg += ".";
    if (c.probeAcc != null) msg += ` Una regresión lineal sobre esta capa acierta el <b>${pct(c.probeAcc)}</b>.`;
    $("#finding").innerHTML = msg;
    renderReadouts(); renderTrack(); renderQuestion(); renderDetail(); renderLegend();
  }

  function nextPlay() {
    if (!S.playing) return;
    if (S.pos >= S.steps.length - 1) { setPlaying(false); return; }
    goTo(S.pos + 1);
  }
  function setPlaying(on) {
    S.playing = on;
    clearTimeout(S.playTimer);
    d3.select("#play-icon").attr("d", on ? "M5.5 4h3v12h-3zM11.5 4h3v12h-3z" : "M6 4 L16 10 L6 16 Z");
    $("#play").setAttribute("aria-label", on ? "Pausar" : "Reproducir");
    if (on) {
      if (S.pos >= S.steps.length - 1) { goTo(0, false); }
      else nextPlay();
    }
  }

  /* ================================================================ línea de tiempo */
  function thumb(canvas, c, count = 420) {
    const g = canvas.getContext("2d"), w = canvas.width, h = canvas.height, R = h / 2 - 4, every = Math.max(1, Math.round(S.d.n / count));
    g.clearRect(0, 0, w, h);
    for (let i = 0; i < S.d.n; i += every) {
      g.fillStyle = PALETTE[S.d.nodes[i].label];
      g.fillRect(w / 2 + c.xyF[i * 2] / Math.max(1, ASPECT) * (w / 2 - 4) - 1, h / 2 - c.xyF[i * 2 + 1] * R - 1, 2.4, 2.4);
    }
  }
  function trackScale(width) {
    const m = { l: 12, r: 16 };
    if (S.axis === "epochs") {
      const maxE = d3.max(S.steps, s => s.epoch);
      return d3.scaleSymlog().constant(1).domain([0, maxE]).range([m.l, width - m.r]);
    }
    const p = d3.scalePoint().domain(S.steps.map((_, k) => k)).range([m.l + 20, width - m.r]);
    return v => p(v);
  }
  function renderTrack() {
    const svg = d3.select("#track"), width = svg.node().getBoundingClientRect().width;
    if (!width) return;
    svg.attr("viewBox", `0 0 ${width} 62`);
    const x = trackScale(width), xs = (st, k) => S.axis === "epochs" ? x(st.epoch) : x(k);
    const yR = 38, y = d3.scaleLinear([0, 1], [yR, 15]);
    const pts = S.steps.map((st, k) => ({ st, k, x: xs(st, k), p: S.d.frames[st.key].purity }));
    svg.selectAll("*").remove();
    svg.append("path").attr("class", "area").attr("d", d3.area().x(d => d.x).y0(yR).y1(d => y(d.p)).curve(d3.curveMonotoneX)(pts));
    svg.append("path").attr("class", "spark").attr("d", d3.line().x(d => d.x).y(d => y(d.p)).curve(d3.curveMonotoneX)(pts));
    svg.append("line").attr("class", "rail").attr("x1", pts[0].x).attr("x2", pts[pts.length - 1].x).attr("y1", yR).attr("y2", yR);
    const g = svg.selectAll("g.stop").data(pts).join("g")
      .attr("class", d => `stop${d.k < S.pos ? " done" : ""}${d.k === S.pos ? " on" : ""}${d.st.best ? " best" : ""}`)
      .attr("transform", d => `translate(${d.x},0)`);
    g.append("circle").attr("cy", yR).attr("r", d => d.k === S.pos ? 5 : 3.2);
    // etiquetas por prioridad (actual, extremos, mejor, resto) sin que se monten
    const order = d3.sort(pts, d => d.k === S.pos ? 0 : d.k === 0 || d.k === pts.length - 1 ? 1 : d.st.best ? 2 : 3);
    const shown = [];
    order.forEach(d => { const wTxt = d.st.tick.length * 6.4 + 8; if (!shown.some(o => Math.abs(o.x - d.x) < (wTxt + o.w) / 2)) shown.push({ x: d.x, w: wTxt, k: d.k }); });
    const showK = new Set(shown.map(o => o.k));
    g.append("text").attr("y", 56).text(d => showK.has(d.k) ? d.st.tick : "");
    const peak = d3.max(pts, d => d.p);
    svg.append("text").attr("class", "scale-note").attr("x", pts[0].x).attr("y", 9).style("text-anchor", "start").style("fill", CORE)
      .text(`pureza, de ${pct(pts[0].p, 0)} a ${pct(pts[pts.length - 1].p, 0)}${S.axis === "epochs" ? "   época en escala logarítmica" : ""}`);
    void peak;
    svg.on("pointermove", ev => preview(nearest(pts, d3.pointer(ev)[0]), pts))
      .on("pointerleave", () => { $("#preview").hidden = true; })
      .on("pointerdown", ev => { svg.node().setPointerCapture(ev.pointerId); scrub(pts, ev); })
      .on("pointerup", ev => svg.node().releasePointerCapture(ev.pointerId));
    svg.on("pointermove.drag", ev => { if (ev.buttons === 1) scrub(pts, ev); });
  }
  const nearest = (pts, px) => d3.least(pts, d => Math.abs(d.x - px));
  function scrub(pts, ev) {
    const d = nearest(pts, d3.pointer(ev)[0]);
    if (d && d.k !== S.pos) { setPlaying(false); goTo(d.k); }
  }
  function preview(d, pts) {
    const box = $("#preview"), tl = $(".timeline").getBoundingClientRect(), tr = $("#track").getBoundingClientRect();
    thumb(box.querySelector("canvas"), S.d.frames[d.st.key]);
    box.querySelector("span").textContent = `${d.st.label}: ${pct(d.p)} de vecinos de su clase`;
    box.style.left = `${Math.max(90, Math.min(tl.width - 90, tr.left - tl.left + d.x))}px`;
    box.hidden = false;
  }

  /* ================================================================ interacción con el lienzo */
  function rebuildQuad() {
    const c = cell(), pts = [];
    for (let i = 0; i < S.d.n; i++) if (visible(i, c)) pts.push([sx(S.cur[i * 2]), sy(S.cur[i * 2 + 1]), i]);
    S.quad = d3.quadtree().x(p => p[0]).y(p => p[1]).addAll(pts);
    S.quadDirty = false;
  }
  function pick(ev) {
    if (S.quadDirty) rebuildQuad();
    const [x, y] = d3.pointer(ev, overlay.node()), hit = S.quad.find(x, y, 14);
    return hit ? hit[2] : null;
  }
  overlay.on("mousemove", ev => {
    if (S.anim || !S.d) return;
    const i = pick(ev);
    if (i !== S.hovered) { S.hovered = i; drawMarks(); renderDetail(); }
    showTip(i, ev);
  }).on("mouseleave", () => { S.hovered = null; showTip(null); drawMarks(); renderDetail(); })
    .on("click", ev => { if (ev.defaultPrevented || !S.d) return; const i = pick(ev); pin(i === S.pinned ? null : i); });

  function showTip(i, ev) {
    const tip = $("#tip");
    if (i == null) { tip.hidden = true; return; }
    const node = S.d.nodes[i], names = S.d.meta.classes, p = predOf(cell())[i];
    tip.innerHTML = "";
    const cv = imgCanvas(i), txt = document.createElement("div");
    txt.innerHTML = `<b style="color:${PALETTE[node.label]};font-weight:500">${names[node.label]}</b><span>${p === node.label ? "acierto" : `predicho como ${names[p]}`}</span>`;
    tip.append(cv, txt);
    tip.hidden = false;
    const [x, y] = d3.pointer(ev, $("#viewport"));
    tip.style.left = `${Math.min(W - 180, x + 16)}px`;
    tip.style.top = `${Math.max(6, y - 54)}px`;
  }
  function pin(i) {
    S.pinned = i;
    draw(); renderDetail();
    if (S.q === "hard") renderQuestion();
  }
  function setHighlight(set, label) {
    S.highlight = set ? { set: new Set(set), label } : null;
    renderHint();
    draw();
  }
  function renderHint() {
    const h = $("#color-hint");
    if (S.highlight) {
      h.innerHTML = `Resaltado: <b>${S.highlight.label}</b><button type="button" id="clear-hl">Limpiar</button>`;
      $("#clear-hl").onclick = () => setHighlight(null);
    } else {
      h.textContent = S.color === "class" ? "" : S.color === "correct"
        ? (S.axis === "layers" ? "Cian: la sonda lineal de esta capa acierta. Ámbar: se equivoca." : "Cian: el modelo acierta en esta época. Ámbar: se equivoca.")
        : "Cian: el mapa 2D conserva sus vecinos reales. Ámbar: su posición en 2D engaña.";
    }
  }

  /* ================================================================ imágenes */
  function drawImg(canvas, i) {
    const A = S.d.meta.atlas, g = canvas.getContext("2d");
    if (canvas.width !== A.w) { canvas.width = A.w; canvas.height = A.h; }
    g.imageSmoothingEnabled = false;
    g.clearRect(0, 0, A.w, A.h);
    g.drawImage(S.d.atlas, (i % A.cols) * A.w, Math.floor(i / A.cols) * A.h, A.w, A.h, 0, 0, A.w, A.h);
  }
  function imgCanvas(i, onclick) {
    const cv = document.createElement("canvas");
    drawImg(cv, i);
    cv.style.setProperty("--c", PALETTE[S.d.nodes[i].label]);
    cv.title = S.d.meta.classes[S.d.nodes[i].label];
    if (onclick) cv.onclick = onclick;
    return cv;
  }

  /* ================================================================ detalle */
  function renderDetail() {
    const i = S.hovered ?? S.pinned, names = S.d.meta.classes;
    const body = $("#d-body");
    if (i == null) {
      $("#d-state").textContent = "";
      body.hidden = true;
      let e = $("#d-empty");
      if (!e) { e = document.createElement("p"); e.id = "d-empty"; e.className = "d-empty"; $("#detail").append(e); }
      e.textContent = "Pasa el cursor sobre un punto para ver la imagen. Haz clic para fijarla y seguir su recorrido.";
      e.hidden = false;
      return;
    }
    if ($("#d-empty")) $("#d-empty").hidden = true;
    body.hidden = false;
    $("#d-state").textContent = i === S.pinned ? "fijada, clic de nuevo para soltar" : "vista previa";
    const node = S.d.nodes[i], c = cell(), p = predOf(c)[i], ok = p === node.label;
    drawImg($("#d-img"), i);
    const learned = node.learned == null ? "Nunca se estabiliza" : `Aprendida desde la época ${node.learned}`;
    $("#d-info").innerHTML =
      `<div class="name" style="color:${PALETTE[node.label]}">${names[node.label]}</div>` +
      `<div><span class="k">${S.axis === "layers" && c.probe ? "Sonda en esta capa" : "Predicción aquí"}:</span> <span class="${ok ? "" : "bad"}">${names[p]}</span></div>` +
      `<div class="k">${learned}</div>`;
    // probabilidades del modelo final: una barra apilada y las tres clases principales
    const order = d3.sort(d3.range(10), k => -node.probs[k]), top = order.slice(0, 3);
    const pr = d3.select("#d-probs").html("");
    pr.append("span").attr("class", "lbl").text("Probabilidades del modelo final");
    const bar = pr.append("div").attr("class", "stack");
    order.forEach(k => { if (node.probs[k] >= .005) bar.append("i").style("flex", node.probs[k]).style("background", PALETTE[k]).attr("title", `${names[k]}: ${pct(node.probs[k], 0)}`); });
    pr.append("div").attr("class", "stack-key").html(top.filter(k => node.probs[k] >= .01)
      .map(k => `<span><i style="background:${PALETTE[k]}"></i>${names[k]} <b>${pct(node.probs[k], 0)}</b></span>`).join(""));
    // historial de decisiones por época
    const hist = d3.select("#d-history").html("");
    const frames = S.d.meta.epochs.map((e, s) => ({ e, f: S.d.frames[cellKey(s, S.d.meta.rowLayer)] })).filter(x => x.f);
    hist.append("span").attr("class", "lbl").text(`Predicción en cada época, ${node.flips} ${node.flips === 1 ? "cambio" : "cambios"}`);
    const w = 304, bw = (w - (frames.length - 1) * 2) / frames.length;
    const svg = hist.append("svg").attr("viewBox", `0 0 ${w} 20`).attr("width", "100%");
    const g = svg.selectAll("g").data(frames).join("g").attr("transform", (_, k) => `translate(${k * (bw + 2)},0)`);
    g.append("rect").attr("width", bw).attr("height", 11).attr("rx", 2).attr("fill", d => PALETTE[d.f.pred[i]]).attr("opacity", .85)
      .append("title").text(d => `${d.e.label}: ${names[d.f.pred[i]]}`);
    g.append("rect").attr("y", 14).attr("width", bw).attr("height", 2).attr("fill", d => d.f.pred[i] === node.label ? HOLO : CORE);
    const nb = d3.select("#d-neighbors").html("");
    if (c.neighbors) {
      const same = c.neighbors[i].filter(j => S.d.nodes[j].label === node.label).length;
      nb.append("span").attr("class", "lbl").text(`Sus 10 vecinos en esta capa: ${same} de su misma clase`);
      c.neighbors[i].forEach(j => nb.node().append(imgCanvas(j, () => pin(j))));
    }
  }

  /* ================================================================ lecturas */
  const READOUTS = [
    { key: "val", label: "Validación", tip: "Exactitud del modelo en validación en esta época" },
    { key: "purity", label: "Vecinos de su clase", tip: "De los 10 vecinos más cercanos en las activaciones, cuántos comparten la clase" },
    { key: "probe", label: "Sonda lineal", tip: "Exactitud de una regresión logística entrenada solo con esta capa (validación cruzada)" },
    { key: "retention", label: "Fidelidad 2D", tip: "Cuántos de esos 10 vecinos siguen siendo vecinos en el mapa 2D" },
  ];
  function renderReadouts() {
    const c = cell(), st = step();
    const vals = { val: S.d.meta.epochs[st.s].val, purity: c.purity, probe: c.probeAcc, retention: c.retention };
    const arc = d3.arc().innerRadius(11).outerRadius(14).startAngle(-2.4);
    d3.select("#gauges").selectAll("div.readout").data(READOUTS.filter(r => vals[r.key] != null), r => r.key).join(enter => {
      const d = enter.append("div").attr("class", "readout").attr("title", r => r.tip);
      const t = d.append("div");
      t.append("span").attr("class", "v");
      t.append("span").attr("class", "l").text(r => r.label);
      const svg = d.append("svg").attr("viewBox", "-15 -15 30 30");
      svg.append("path").attr("class", "track-arc").attr("d", d3.arc().innerRadius(11).outerRadius(14).startAngle(-2.4).endAngle(2.4)());
      svg.append("path").attr("class", "val-arc").each(function () { this._v = 0; });
      return d;
    }).each(function (r) {
      const v = vals[r.key], s = d3.select(this);
      s.select(".v").text(pct(v));
      s.select(".val-arc").transition().duration(REDUCED ? 0 : 600).attrTween("d", function () {
        const it = d3.interpolate(this._v, v); this._v = v;
        return t => arc.endAngle(-2.4 + 4.8 * it(t))();
      });
    });
  }

  /* ================================================================ preguntas */
  function renderQuestion() {
    const q = QUESTIONS[S.q];
    $("#q-code").textContent = q.code;
    $("#q-title").textContent = q.title;
    const body = d3.select("#q-body").html("");
    ({ epochs: qEpochs, layers: qLayers, confusion: qConfusion, hard: qHard })[S.q](body);
    d3.select("#q-insights").selectAll("li").data(S.d.meta.insights[S.q] || []).join("li").text(d => d);
  }

  /* posiciona etiquetas al final de cada línea sin que se monten */
  function endLabels(items, minGap = 12) {
    items.sort((a, b) => a.y - b.y);
    for (let k = 1; k < items.length; k++) if (items[k].y - items[k - 1].y < minGap) items[k].y = items[k - 1].y + minGap;
    return items;
  }

  function epochTicks(values, x) {
    const last = values[values.length - 1], keep = [last];
    for (const v of values.slice(0, -1)) if (keep.every(k => Math.abs(x(k) - x(v)) > 15)) keep.push(v);
    return keep.sort(d3.ascending);
  }

  function qEpochs(body) {
    const { meta, frames } = S.d, w = 304, h = 168, m = { t: 8, r: 70, b: 22, l: 26 };
    const L = S.axis === "epochs" ? S.fixedLayer : meta.rowLayer;
    const pts = meta.epochs.map((e, s) => ({ e: e.epoch, s, c: frames[cellKey(s, L)] })).filter(p => p.c);
    const x = d3.scaleSymlog().constant(1).domain([0, d3.max(meta.curve, d => d.epoch)]).range([m.l, w - m.r]);
    const y = d3.scaleLinear([0, 1], [h - m.b, m.t]);
    const svg = body.append("svg").attr("viewBox", `0 0 ${w} ${h}`);
    svg.append("g").attr("class", "grid").selectAll("line").data([.25, .5, .75, 1]).join("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y).attr("y2", y);
    svg.append("g").attr("class", "ax").attr("transform", `translate(0,${h - m.b})`)
      .call(d3.axisBottom(x).tickValues(epochTicks(pts.map(p => p.e), x)).tickFormat(d3.format("d")).tickSize(3).tickSizeOuter(0));
    svg.append("g").attr("class", "ax").attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).tickValues([0, .5, 1]).tickFormat(d => `${d * 100}`).tickSize(3).tickSizeOuter(0)).call(g => g.select(".domain").remove());
    const sel = S.axis === "epochs" ? meta.epochs[step().s].epoch : meta.epochs[S.fixedEpoch].epoch;
    svg.append("line").attr("x1", x(sel)).attr("x2", x(sel)).attr("y1", m.t).attr("y2", h - m.b).attr("stroke", CORE).attr("stroke-opacity", .5).attr("stroke-dasharray", "2 3");
    const series = [
      { label: "Validación", color: HOLO, data: meta.curve.map(d => ({ e: d.epoch, v: d.val })), dash: null, w: 1.6 },
      { label: "Vecinos de su clase", color: CORE, data: pts.map(p => ({ e: p.e, v: p.c.purity })), dash: null, w: 1.6 },
      { label: "Sonda lineal", color: INK, data: pts.filter(p => p.c.probeAcc != null).map(p => ({ e: p.e, v: p.c.probeAcc })), dash: "4 3", w: 1.2 },
      { label: "Fidelidad 2D", color: MUTE, data: pts.map(p => ({ e: p.e, v: p.c.retention })), dash: "1 3", w: 1.2 },
    ].filter(s => s.data.length > 1);
    const line = d3.line().x(d => x(d.e)).y(d => y(d.v)).curve(d3.curveMonotoneX);
    series.forEach(s => svg.append("path").attr("d", line(s.data)).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", s.w).attr("stroke-dasharray", s.dash));
    endLabels(series.map(s => ({ s, y: y(s.data[s.data.length - 1].v) }))).forEach(o =>
      svg.append("text").attr("class", "end").attr("x", w - m.r + 6).attr("y", o.y + 3.5).attr("fill", o.s.color).text(o.s.label.replace("Vecinos de su clase", "Pureza")));
    svg.append("g").selectAll("circle").data(pts).join("circle")
      .attr("cx", d => x(d.e)).attr("cy", d => y(d.c.purity)).attr("r", d => d.e === sel ? 4.5 : 3)
      .attr("fill", d => d.e === sel ? CORE : VOID).attr("stroke", CORE).attr("stroke-width", 1.3).style("cursor", "pointer")
      .on("click", (_, d) => { setAxis("epochs", L); goTo(S.steps.findIndex(st => st.s === d.s)); })
      .append("title").text(d => `Época ${d.e}: pureza ${pct(d.c.purity)}`);
    body.append("p").attr("class", "note").text("Eje horizontal: época en escala logarítmica. Eje vertical: %. Clic en un punto para ir a esa época.");
  }

  function qLayers(body) {
    const { meta, frames } = S.d, E = S.axis === "layers" ? S.fixedEpoch : meta.colEpoch;
    const rows = meta.layers.map((name, l) => ({ l, name, c: frames[cellKey(E, l)], c0: E !== 0 ? frames[cellKey(0, l)] : null })).filter(r => r.c);
    const showZero = rows.filter(r => r.c0).length > 1;
    const w = 304, rh = 25, h = rows.length * rh + 24, x = d3.scaleLinear([0, 1], [92, w - 30]);
    const svg = body.append("svg").attr("viewBox", `0 0 ${w} ${h + 16}`);
    const key = svg.append("g").attr("transform", "translate(0,6)");
    [["Pureza", CORE, 0], ["Sonda lineal", HOLO, 64], ...(showZero ? [["Pureza sin entrenar", INK, 152]] : [])].forEach(([t, c, dx]) => {
      key.append("rect").attr("x", dx).attr("y", -4).attr("width", t.includes("sin") ? 2 : 10).attr("height", t.includes("sin") ? 9 : 3).attr("fill", c);
      key.append("text").attr("class", "lab").attr("x", dx + 14).attr("y", 0).attr("dy", ".32em").style("font-size", "10.5px").text(t);
    });
    const cur = S.axis === "layers" ? step().l : null;
    const g = svg.selectAll("g.row").data(rows).join("g").attr("class", "row").style("cursor", "pointer")
      .attr("transform", (_, k) => `translate(0,${k * rh + 20})`)
      .on("click", (_, r) => { setAxis("layers", E); goTo(S.steps.findIndex(st => st.l === r.l)); });
    g.append("rect").attr("x", -4).attr("width", w + 4).attr("height", rh - 2).attr("rx", 3).attr("fill", r => r.l === cur ? "rgba(255,180,84,.09)" : "transparent");
    g.append("text").attr("class", r => `lab${r.l === cur ? " on" : ""}`).attr("x", 0).attr("y", 15).text(r => layerName(r.name));
    g.append("rect").attr("x", x(0)).attr("y", 6).attr("height", 4).attr("rx", 1).attr("width", r => x(r.c.purity) - x(0)).attr("fill", CORE);
    g.append("rect").attr("x", x(0)).attr("y", 13).attr("height", 4).attr("rx", 1).attr("width", r => r.c.probeAcc == null ? 0 : x(r.c.probeAcc) - x(0)).attr("fill", HOLO);
    if (showZero) g.filter(r => r.c0).append("rect").attr("x", r => x(r.c0.purity) - 1).attr("y", 3).attr("width", 2).attr("height", 17).attr("fill", INK);
    g.append("text").attr("class", "lab").attr("x", w).attr("y", 15).attr("text-anchor", "end").style("font-family", "var(--mono)").style("font-size", "10.5px")
      .text(r => Math.round(r.c.purity * 100));
    svg.append("g").attr("class", "ax").attr("transform", `translate(0,${rows.length * rh + 22})`)
      .call(d3.axisBottom(x).tickValues([0, .5, 1]).tickFormat(d => `${d * 100} %`).tickSize(3).tickSizeOuter(0));

    // mapa de neuronas
    const hiddenLayers = S.d.neurons.map(nm => nm.layer);
    const curName = S.axis === "layers" ? meta.layers[step().l] : meta.layers[S.fixedLayer];
    if (hiddenLayers.includes(curName) && S.axis === "layers") S.neuronLayer = curName;
    if (!hiddenLayers.includes(S.neuronLayer)) S.neuronLayer = hiddenLayers.includes(curName) ? curName : hiddenLayers[hiddenLayers.length - 1];
    const NM = S.d.neurons.find(nm => nm.layer === S.neuronLayer);
    const sub = body.append("div").attr("class", "sub");
    sub.append("span").text("Mapa de neuronas");
    sub.append("select").attr("aria-label", "Capa del mapa de neuronas")
      .on("change", ev => { S.neuronLayer = ev.target.value; renderQuestion(); })
      .selectAll("option").data(hiddenLayers).join("option").attr("value", d => d).text(d => layerName(d)).property("selected", d => d === S.neuronLayer);
    const nw = 304, nh = 170, xN = d3.scaleLinear([-1.1, 1.1], [8, nw - 8]), yN = d3.scaleLinear([-1.1, 1.1], [nh - 8, 8]);
    const rS = d3.scaleSqrt([0, d3.max(NM.neurons, d => d.sel) || 1], [2, 7.5]);
    const svgN = body.append("svg").attr("class", "neuron-map").attr("viewBox", `0 0 ${nw} ${nh}`);
    svgN.append("rect").attr("width", nw).attr("height", nh).attr("rx", 6).attr("fill", "rgba(255,255,255,.015)").attr("stroke", "rgba(125,205,240,.13)");
    const info = body.append("p").attr("class", "note");
    const thumbs = body.append("div").attr("class", "thumbs").style("margin-top", "8px");
    const defaultInfo = `${NM.neurons.filter(d => d.alive).length} neuronas activas. Cada punto es una neurona, ubicada según a qué imágenes responde; el color es la clase que más la enciende y el tamaño, qué tan selectiva es.`;
    info.text(defaultInfo);
    svgN.selectAll("circle").data(NM.neurons).join("circle").attr("class", "nrn")
      .attr("cx", d => xN(d.x)).attr("cy", d => yN(d.y)).attr("r", d => d.alive ? rS(d.sel) : 2)
      .attr("fill", d => d.alive ? PALETTE[d.cls] : "none").attr("fill-opacity", .85).attr("stroke", d => d.alive ? null : FAINT)
      .on("mouseenter", (_, d) => {
        if (!d.alive) { info.text("Neurona inactiva: no se enciende con ninguna de estas imágenes."); return; }
        setHighlight(d.top, `neurona ${d.i}, ${layerName(NM.layer)}`);
        thumbs.html("");
        d.top.slice(0, 16).forEach(j => thumbs.node().append(imgCanvas(j, () => pin(j))));
        const share = d.top.filter(j => S.d.nodes[j].label === d.cls).length;
        info.text(`Neurona ${d.i}: responde sobre todo a ${meta.classes[d.cls]} (${share} de sus 24 imágenes más activas).`);
      });
  }

  function confusionOf(c) {
    const pred = predOf(c), M = d3.range(10).map(() => new Array(10).fill(0));
    S.d.nodes.forEach((nd, i) => { M[nd.label][pred[i]] += 1; });
    return M;
  }

  function qConfusion(body) {
    const c = cell(), names = S.d.meta.classes, M = confusionOf(c), usingProbe = S.axis === "layers" && c.probe;
    const sub = body.append("div").attr("class", "sub").style("margin-top", "0");
    sub.append("span").text("Matriz de confusión");
    sub.append("small").text(usingProbe ? "sonda lineal de esta capa" : "predicciones del modelo");
    const w = 304, ox = 74, cs = (w - ox) / 10, oy = 2;
    const svg = body.append("svg").attr("class", "cm").attr("viewBox", `0 0 ${w} ${cs * 10 + 24}`);
    const max = d3.max(M.flatMap((r, a) => r.filter((_, b) => a !== b))) || 1;
    const off = d3.scaleSqrt([0, max], [0, 1]), diag = d3.scaleLinear([0, 100], [0, 1]);
    names.forEach((nm, a) => svg.append("text").attr("class", "lab").attr("x", ox - 6).attr("y", oy + a * cs + cs / 2).attr("dy", ".32em")
      .attr("text-anchor", "end").style("font-size", "10.5px").attr("fill", PALETTE[a]).text(nm));
    names.forEach((_, b) => svg.append("rect").attr("x", ox + b * cs + 2).attr("y", oy + cs * 10 + 3).attr("width", cs - 5).attr("height", 2.5).attr("rx", 1).attr("fill", PALETTE[b]));
    svg.append("text").attr("class", "lab").attr("x", ox + cs * 5).attr("y", oy + cs * 10 + 18).attr("text-anchor", "middle").style("font-size", "10.5px").text("clase predicha");
    const cells = M.flatMap((r, a) => r.map((v, b) => ({ a, b, v })));
    const alpha = d => d.v === 0 ? .03 : d.a === d.b ? .12 + .8 * diag(d.v) : .1 + .85 * off(d.v);
    svg.selectAll("rect.cell").data(cells).join("rect").attr("class", "cell")
      .attr("x", d => ox + d.b * cs).attr("y", d => oy + d.a * cs).attr("width", cs - 1.5).attr("height", cs - 1.5).attr("rx", 2)
      .attr("fill", d => d.a === d.b ? HOLO : CORE).attr("fill-opacity", d => alpha(d))
      .on("click", (_, d) => {
        const pred = predOf(c), set = S.d.nodes.filter((nd, i) => nd.label === d.a && pred[i] === d.b).map(nd => nd.i);
        setHighlight(set, `${names[d.a]} predicho como ${names[d.b]} (${set.length})`);
      })
      .append("title").text(d => `${names[d.a]} predicho como ${names[d.b]}: ${d.v}`);
    svg.selectAll("text.v").data(cells.filter(d => d.a === d.b || d.v >= Math.max(3, max * .35))).join("text").attr("class", "v")
      .attr("x", d => ox + d.b * cs + cs / 2 - .75).attr("y", d => oy + d.a * cs + cs / 2).attr("dy", ".34em")
      .attr("fill", d => alpha(d) > .55 ? VOID : INK).text(d => d.v);

    // red de clases
    const s2 = body.append("div").attr("class", "sub");
    s2.append("span").text("Red de clases");
    s2.append("small").text("vecinos compartidos en esta etapa");
    const pair = d3.range(10).map(() => new Array(10).fill(0));
    if (c.neighbors) c.neighbors.forEach((row, i) => row.forEach(j => { pair[S.d.nodes[i].label][S.d.nodes[j].label] += 1; }));
    const tot = pair.map(r => d3.sum(r) || 1), links = [];
    for (let a = 0; a < 10; a++) for (let b = a + 1; b < 10; b++) {
      const v = (pair[a][b] / tot[a] + pair[b][a] / tot[b]) / 2;
      if (v > .01) links.push({ a, b, v });
    }
    links.sort((p, q) => q.v - p.v);
    const top = links.slice(0, 12), nw = 304, nh = 214, R = 72;
    const pos = d3.range(10).map(k => [nw / 2 + R * 1.25 * Math.cos(k / 10 * 2 * Math.PI - Math.PI / 2), nh / 2 + R * Math.sin(k / 10 * 2 * Math.PI - Math.PI / 2)]);
    const net = body.append("svg").attr("class", "net").attr("viewBox", `0 0 ${nw} ${nh}`);
    const wS = d3.scaleLinear([0, top[0]?.v || 1], [.6, 8]);
    net.selectAll("line").data(top).join("line").attr("class", "edge")
      .attr("x1", l => pos[l.a][0]).attr("y1", l => pos[l.a][1]).attr("x2", l => pos[l.b][0]).attr("y2", l => pos[l.b][1])
      .attr("stroke", CORE).attr("stroke-linecap", "round").attr("stroke-opacity", l => .25 + .6 * l.v / top[0].v).attr("stroke-width", l => wS(l.v))
      .on("click", (_, l) => {
        const set = c.neighbors ? S.d.nodes.filter((nd, i) => (nd.label === l.a && c.neighbors[i].some(j => S.d.nodes[j].label === l.b)) ||
          (nd.label === l.b && c.neighbors[i].some(j => S.d.nodes[j].label === l.a))).map(nd => nd.i) : [];
        setHighlight(set, `frontera entre ${names[l.a]} y ${names[l.b]} (${set.length})`);
      })
      .append("title").text(l => `${names[l.a]} y ${names[l.b]}: ${pct(l.v)} de vecinos compartidos`);
    const nodes = net.selectAll("g.node").data(d3.range(10)).join("g").attr("class", "node").attr("transform", k => `translate(${pos[k]})`)
      .on("click", (_, k) => setHighlight(S.d.nodes.filter(nd => nd.label === k).map(nd => nd.i), `clase ${names[k]}`));
    nodes.append("circle").attr("r", k => 3.5 + 7 * pair[k][k] / tot[k]).attr("fill", k => PALETTE[k])
      .append("title").text(k => `${names[k]}: ${pct(pair[k][k] / tot[k])} de vecinos propios`);
    nodes.append("text").attr("class", "lab").style("font-size", "11px")
      .attr("text-anchor", k => Math.abs(pos[k][0] - nw / 2) < 6 ? "middle" : pos[k][0] < nw / 2 ? "end" : "start")
      .attr("dx", k => Math.abs(pos[k][0] - nw / 2) < 6 ? 0 : pos[k][0] < nw / 2 ? -13 : 13)
      .attr("dy", k => Math.abs(pos[k][0] - nw / 2) < 6 ? (pos[k][1] < nh / 2 ? -14 : 22) : 4).text(k => names[k]);
    body.append("p").attr("class", "note").text("Grosor de la arista: proporción de vecinos compartidos. Tamaño del nodo: cohesión de la clase. Clic en una arista para ver su frontera en el holograma.");

    const s3 = body.append("div").attr("class", "sub");
    s3.append("span").text("Fotos puente");
    s3.append("small").text("mayor intermediación");
    const th = body.append("div").attr("class", "thumbs");
    S.d.edges.bridges.slice(0, 16).forEach(b => {
      const cv = imgCanvas(b.node, () => pin(b.node));
      cv.title = `${names[S.d.nodes[b.node].label]}: ${pct(b.otherShare, 0)} de sus vecinos son de otra clase, sobre todo ${names[b.towards]}`;
      th.node().append(cv);
    });
  }

  function qHard(body) {
    const names = S.d.meta.classes, ranked = d3.sort(S.d.nodes, n => -n.score).slice(0, 30);
    const final = S.d.frames[cellKey(S.d.meta.colEpoch, S.d.meta.rowLayer)], maxF = d3.max(S.d.nodes, n => n.flips);
    body.append("p").attr("class", "note").style("margin", "0 0 8px")
      .text("Ordenadas por dificultad (confianza, error, vecinos de otra clase y cambios de opinión). Las barras cuentan los cambios entre épocas.");
    const list = body.append("div").attr("class", "rank");
    ranked.forEach((n, k) => {
      const p = final.pred[n.i];
      const b = list.append("button").attr("type", "button").attr("aria-current", n.i === S.pinned)
        .on("click", () => { if (S.axis !== "epochs") setAxis("epochs", S.d.meta.rowLayer); pin(n.i); })
        .on("mouseenter", () => { S.hovered = n.i; drawMarks(); renderDetail(); })
        .on("mouseleave", () => { S.hovered = null; drawMarks(); renderDetail(); });
      b.append("span").attr("class", "pos").text(k + 1);
      b.node().append(imgCanvas(n.i));
      b.append("span").html(`<span style="color:${PALETTE[n.label]}">${names[n.label]}</span><span class="meta">${p === n.label ? "acierta" : `predice ${names[p]}`}, confianza ${pct(n.conf, 0)}</span>`);
      const f = b.append("span").attr("class", "flips").attr("title", `${n.flips} cambios de predicción`);
      d3.range(maxF).forEach(q => f.append("i").style("height", `${6 + q * 1.4}px`).style("opacity", q < n.flips ? 1 : .12));
    });
  }

  /* ================================================================ leyenda y controles */
  function renderLegend() {
    const names = S.d.meta.classes, c = cell(), pred = predOf(c);
    const leg = d3.select("#legend");
    leg.selectAll("button").data(names).join(enter => enter.append("button").attr("type", "button").each(function (_, k) {
      let timer = null;
      this.onclick = () => { clearTimeout(timer); timer = setTimeout(() => { S.hidden.has(k) ? S.hidden.delete(k) : S.hidden.add(k); refreshFilters(); }, 220); };
      this.ondblclick = () => {
        clearTimeout(timer);
        const only = S.hidden.size === 9 && !S.hidden.has(k);
        S.hidden = new Set(only ? [] : d3.range(10).filter(x => x !== k));
        refreshFilters();
      };
    }))
      .attr("aria-pressed", (_, k) => !S.hidden.has(k))
      .attr("title", (_, k) => `${names[k]}: aciertos en esta etapa. Clic para ocultar, doble clic para aislar.`)
      .html((nm, k) => {
        const of = S.d.nodes.filter(n => n.label === k), acc = of.filter(n => pred[n.i] === k).length / of.length;
        return `<span class="dot" style="background:${PALETTE[k]}"></span><span>${nm}</span><span class="acc">${Math.round(acc * 100)}%</span>`;
      });
    leg.selectAll(".legend-note").data([0]).join("span").attr("class", "legend-note").text("% de aciertos por clase");
  }
  function refreshFilters() { S.quadDirty = true; computeTrails(); renderLegend(); draw(); }

  function setAxis(axis, fixed) {
    const changed = S.axis !== axis && S.steps.length;
    const was = changed ? step() : null;
    S.axis = axis;
    if (axis === "epochs") S.fixedLayer = fixed ?? S.fixedLayer; else S.fixedEpoch = fixed ?? S.fixedEpoch;
    d3.selectAll("#axis button").attr("aria-pressed", function () { return this.dataset.axis === axis; });
    $("#fixed-label").textContent = axis === "epochs" ? "capa" : "época";
    const opts = fixedOptions(axis), cur = axis === "epochs" ? S.fixedLayer : S.fixedEpoch;
    d3.select("#fixed").selectAll("option").data(opts).join("option").attr("value", o => o.v)
      .text(o => o.ok ? o.label : `${o.label} (sin datos)`).property("disabled", o => !o.ok).property("selected", o => o.v === cur);
    if (was) { if (axis === "epochs") S.fixedLayer = fixed ?? was.l; else S.fixedEpoch = fixed ?? was.s; }
    if (was) {
      const fopt = fixedOptions(axis).find(o => o.v === (axis === "epochs" ? S.fixedLayer : S.fixedEpoch));
      if (!fopt?.ok) { if (axis === "epochs") S.fixedLayer = S.d.meta.rowLayer; else S.fixedEpoch = S.d.meta.colEpoch; }
      d3.select("#fixed").selectAll("option").property("selected", o => o.v === (axis === "epochs" ? S.fixedLayer : S.fixedEpoch));
    }
    buildSteps();
    if (was) {
      const k = S.steps.findIndex(st => axis === "epochs" ? st.s === was.s : st.l === was.l);
      S.pos = k >= 0 ? k : S.steps.length - 1;
    }
    S.pos = Math.min(S.pos, S.steps.length - 1);
    renderHint();
    goTo(S.pos, true);
  }
  function setQuestion(q) {
    S.q = q;
    d3.selectAll("#questions button").attr("aria-pressed", function () { return this.dataset.q === q; });
    setHighlight(null);
    renderQuestion();
    $(".question").scrollTop = 0;
  }
  function setChip(id, on) { $(id).setAttribute("aria-pressed", on); }

  /* ================================================================ vista general */
  function openMatrix() {
    const { meta, frames } = S.d, grid = d3.select("#matrix-grid").html("");
    $("#matrix-title").textContent = `Vista general de ${meta.name}: época × capa`;
    $("#matrix-sub").textContent = meta.partial
      ? "Filas: capas, de la entrada a la salida. Columnas: épocas. El entrenamiento original de CIFAR-10 guardó el bloque 3 en cada época y todas las capas en la época seleccionada, por eso la matriz tiene forma de cruz. El número es el % de vecinos de su clase."
      : "Filas: capas, de la entrada a la salida. Columnas: épocas. Cada celda es una proyección t-SNE y el número es el % de vecinos de su clase. Clic en una celda para abrirla.";
    grid.style("grid-template-columns", `112px repeat(${meta.epochs.length}, minmax(48px, 1fr))`);
    grid.append("div");
    meta.epochs.forEach(e => grid.append("div").attr("class", `h${e.epoch === meta.bestEpoch ? " best" : ""}`).text(e.epoch));
    meta.layers.forEach((name, l) => {
      grid.append("div").attr("class", "rh").text(layerName(name));
      meta.epochs.forEach((e, s) => {
        const f = frames[cellKey(s, l)];
        if (!f) { grid.append("div").attr("class", "missing").attr("title", "Sin activaciones guardadas"); return; }
        const isCur = step().key === cellKey(s, l);
        const b = grid.append("button").attr("type", "button").attr("class", `cellbtn${isCur ? " cur" : ""}`).attr("title", `${e.label}, ${layerName(name)}`)
          .on("click", () => {
            closeMatrix();
            const rowOk = fixedOptions("epochs").find(o => o.v === l)?.ok;
            if (rowOk) { setAxis("epochs", l); goTo(S.steps.findIndex(st => st.s === s)); }
            else { setAxis("layers", s); goTo(S.steps.findIndex(st => st.l === l)); }
          });
        b.append("canvas").attr("width", 120).attr("height", 84).each(function () { thumb(this, f, 320); });
        b.append("span").attr("class", "pv").text(Math.round(f.purity * 100));
      });
    });
    $("#matrix").hidden = false;
    $("#close-matrix").focus();
  }
  function closeMatrix() { $("#matrix").hidden = true; }

  /* ================================================================ recorrido guiado */
  let tourIdx = 0;
  function tourSteps() {
    const m = S.d.meta, ins = m.insights, names = m.classes, hardest = d3.sort(S.d.nodes, n => -n.score)[0];
    const frontier = () => {
      const [a, b] = m.topPair, c = cell();
      return S.d.nodes.filter((nd, i) => (nd.label === a && c.neighbors[i].some(j => S.d.nodes[j].label === b)) ||
        (nd.label === b && c.neighbors[i].some(j => S.d.nodes[j].label === a))).map(nd => nd.i);
    };
    return [
      { title: `Qué ve una red neuronal mientras aprende ${m.name}`,
        text: "Cada punto es una de 1.000 imágenes de validación. Su posición resume lo que la red ve en una capa (proyección t-SNE) y su color es la clase real. Cuatro preguntas guían el recorrido.",
        run: () => { setQuestion("epochs"); bundles(false); setAxis("epochs", m.rowLayer); pin(null); goTo(S.steps.length - 1, false); } },
      { title: "1. ¿En qué época aparece la estructura?", text: `${ins.epochs[0]} ${ins.epochs[1]}`,
        run: () => { setQuestion("epochs"); bundles(false); setAxis("epochs", m.rowLayer); pin(null); goTo(0, false); setTimeout(() => setPlaying(true), 700); } },
      { title: "2. ¿En qué capa se separan las clases?", text: `${ins.layers[1]} ${ins.layers[ins.layers.length - 1]} Los haces dibujan el recorrido de cada imagen de capa en capa.`,
        run: () => { setPlaying(false); setQuestion("layers"); bundles(true); setAxis("layers", m.colEpoch); goTo(0, false); setTimeout(() => setPlaying(true), 700); } },
      { title: "3. ¿Qué clases confunde?", text: `${ins.confusion[0]} ${ins.confusion[1]} En el holograma quedan resaltadas las imágenes de esa frontera.`,
        run: () => { setPlaying(false); bundles(false); setQuestion("confusion"); setAxis("epochs", m.rowLayer); goTo(S.steps.length - 1, false);
          const [a, b] = m.topPair; setHighlight(frontier(), `frontera entre ${names[a]} y ${names[b]}`); } },
      { title: "4. ¿Qué imágenes le cuestan?", text: `${ins.hard[0]} La más difícil es ${m.articles[hardest.label]}: su línea muestra cómo se movió de época en época; los puntos ámbar marcan cuándo se equivocó.`,
        run: () => { setQuestion("hard"); setAxis("epochs", m.rowLayer); goTo(S.steps.length - 1, false); pin(hardest.i); } },
    ];
  }
  function showTour(k) {
    const steps = tourSteps();
    tourIdx = Math.max(0, Math.min(steps.length - 1, k));
    const st = steps[tourIdx];
    $("#tour").hidden = false;
    d3.select("#tour-progress").selectAll("i").data(steps).join("i").attr("class", (_, q) => q <= tourIdx ? "on" : null);
    $("#tour-title").textContent = st.title;
    $("#tour-text").textContent = st.text;
    $("#tour-prev").disabled = tourIdx === 0;
    $("#tour-next").textContent = tourIdx === steps.length - 1 ? "Terminar" : "Siguiente";
    st.run();
  }
  function bundles(on) { S.bundles = on; setChip("#t-bundles", on); $("#bundle-row").hidden = !on; }

  /* ================================================================ arranque */
  async function selectDataset(key, first = false) {
    setPlaying(false);
    const d = await load(key);
    S.key = key; S.d = d;
    S.hidden = new Set(); S.pinned = null; S.hovered = null; S.highlight = null;
    S.fixedLayer = d.meta.rowLayer; S.fixedEpoch = d.meta.colEpoch; S.neuronLayer = null; S.cur = null;
    d3.selectAll("#datasets button").attr("aria-pressed", function () { return this.dataset.key === key; });
    S.axis = S.q === "layers" ? "layers" : "epochs";
    buildSteps();
    S.pos = S.steps.length - 1;
    overlay.call(zoom.transform, d3.zoomIdentity);
    setAxis(S.axis);
    if (first) await boot(d);
  }

  async function boot(d) {
    const log = d3.select("#boot-log"), box = $("#boot");
    const lines = [
      `${d.meta.name}: ${num(d.n)} imágenes de validación, ${d.meta.classes.length} clases`,
      d.meta.model,
      `${num(Object.keys(d.frames).length)} proyecciones t-SNE enlazadas`,
      `${num(d.edges.edges.length)} aristas en la red de vecinos`,
      `exactitud en prueba: ${pct(d.meta.testAccuracy)}`,
    ];
    let skip = false;
    box.onclick = () => { skip = true; };
    if (!REDUCED) {
      for (const line of lines) {
        if (skip) break;
        log.append("li").text(line);
        await new Promise(r => setTimeout(r, 220));
      }
      if (!skip) { log.append("li").attr("class", "ok").text("listo"); await new Promise(r => setTimeout(r, 420)); }
    }
    box.classList.add("out");
    setTimeout(() => box.remove(), 600);
  }

  function wire() {
    d3.selectAll("#datasets button").on("click", function () { selectDataset(this.dataset.key); });
    d3.selectAll("#questions button").on("click", function () {
      const q = this.dataset.q;
      setQuestion(q);
      if (q === "epochs" && S.axis !== "epochs") setAxis("epochs", S.d.meta.rowLayer);
      if (q === "layers" && S.axis !== "layers") setAxis("layers", S.d.meta.colEpoch);
    });
    d3.selectAll("#axis button").on("click", function () { setPlaying(false); setAxis(this.dataset.axis); });
    $("#fixed").onchange = ev => { setPlaying(false); setAxis(S.axis, Number(ev.target.value)); };
    d3.selectAll("#color-mode button").on("click", function () {
      S.color = this.dataset.mode;
      d3.selectAll("#color-mode button").attr("aria-pressed", function () { return this.dataset.mode === S.color; });
      renderHint(); draw();
    });
    $("#t-bundles").onclick = () => { bundles(!S.bundles); computeTrails(); draw(); };
    $("#t-errors").onclick = () => { S.errors = !S.errors; setChip("#t-errors", S.errors); refreshFilters(); };
    $("#t-links").onclick = () => { S.links = !S.links; setChip("#t-links", S.links); drawMarks(); };
    $("#t-photos").onclick = () => { S.photos = !S.photos; setChip("#t-photos", S.photos); draw(); };
    let bt = null;
    $("#bundle-strength").oninput = ev => { S.strength = Number(ev.target.value); clearTimeout(bt); bt = setTimeout(() => { computeTrails(); draw(); }, 60); };
    $("#play").onclick = () => setPlaying(!S.playing);
    $("#reset-zoom").onclick = () => overlay.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    $("#open-matrix").onclick = openMatrix;
    $("#close-matrix").onclick = closeMatrix;
    $("#matrix").onclick = ev => { if (ev.target.id === "matrix") closeMatrix(); };
    $("#open-tour").onclick = () => showTour(0);
    $("#tour-next").onclick = () => { if (tourIdx === tourSteps().length - 1) { $("#tour").hidden = true; setPlaying(false); } else showTour(tourIdx + 1); };
    $("#tour-prev").onclick = () => showTour(tourIdx - 1);
    $("#tour-close").onclick = () => { $("#tour").hidden = true; setPlaying(false); };
    document.addEventListener("keydown", ev => {
      if (ev.target.matches("input[type=range], select, textarea") || !S.d) return;
      if (ev.key === "ArrowRight") { setPlaying(false); goTo(S.pos + 1); }
      else if (ev.key === "ArrowLeft") { setPlaying(false); goTo(S.pos - 1); }
      else if (ev.key === " " && !ev.target.matches("button")) { ev.preventDefault(); setPlaying(!S.playing); }
      else if (ev.key === "Escape") { if (!$("#matrix").hidden) closeMatrix(); else { pin(null); setHighlight(null); } }
    });
    new ResizeObserver(() => resize()).observe($("#viewport"));
  }

  async function init() {
    wire();
    resize();
    const first = $("#datasets button");
    if (!first) { $("#stage-title").textContent = "No hay datos generados"; $("#boot")?.remove(); return; }
    try {
      await selectDataset(first.dataset.key, true);
    } catch (err) {
      console.error(err);
      $("#boot")?.remove();
      $("#empty").hidden = false;
      $("#empty").textContent = `No se pudieron cargar los datos (${err.message}). Ejecuta "python app.py" y abre http://127.0.0.1:5000.`;
    }
  }
  init();
})();
