// VLA Arch: renders the atlas from data/models.js.
// Every diagram is data (nodes on a grid, edges between node sides, frames around
// groups) drawn here as inline SVG, so all figures share one visual grammar.

(() => {
  const DATA = window.VLA_ARCH;
  if (!DATA) return;

  const NS = 'http://www.w3.org/2000/svg';
  const ENGINES = { cpp: 'vla.cpp', simd: 'vla.simd' };
  let uidSeq = 0;

  const svgEl = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const h = (tag, attrs = {}, html) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (html !== undefined) n.innerHTML = html;
    return n;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const plain = (s) => String(s).replace(/<[^>]+>/g, '');

  // ---------------------------------------------------------------------------
  // Block diagrams
  // Grid units: node {c, r} is its column / row, {w, h} its span in cells.
  // Coordinates for routing (mx, my, via) accept a number in grid units, or
  // 'gN' for the centre of the gutter before column / row N, with an optional
  // px offset ('g3+8').
  // ---------------------------------------------------------------------------
  const BASE = { cw: 150, gx: 44, rh: 54, gy: 32, px: 22, py: 30 };

  function diagram(d, opts = {}) {
    const G = { ...BASE, ...(d.grid || {}) };
    const uid = `d${++uidSeq}`;
    const colX = (c) => G.px + c * (G.cw + G.gx);
    const rowY = (r) => G.py + r * (G.rh + G.gy);
    const span = (n, cell, gap) => (n >= 1 ? n * cell + (n - 1) * gap : n * cell);
    const coord = (v, axis) => {
      if (typeof v === 'number') return axis === 'x' ? colX(v) : rowY(v);
      const m = /^g(-?\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?$/.exec(v);
      if (!m) return 0;
      const base = axis === 'x' ? colX(+m[1]) - G.gx / 2 : rowY(+m[1]) - G.gy / 2;
      return base + (m[2] ? +m[2] : 0);
    };

    const boxes = {};
    for (const n of d.nodes) {
      boxes[n.id] = {
        x: colX(n.c) + (n.dx || 0), y: rowY(n.r) + (n.dy || 0),
        w: n.pw || span(n.w ?? 1, G.cw, G.gx), h: n.ph || span(n.h ?? 1, G.rh, G.gy),
      };
    }

    const svg = svgEl('svg', { class: `dg${opts.mini ? ' dg--mini' : ''}`, role: 'img', 'aria-label': plain(d.label || opts.label || '') });
    const defs = svgEl('defs', {}, svg);
    for (const k of ['data', 'attn', 'cond', 'loop', 'skip']) {
      const m = svgEl('marker', {
        id: `${uid}-${k}`, viewBox: '0 0 10 10', refX: 9, refY: 5,
        markerWidth: 8, markerHeight: 8, markerUnits: 'userSpaceOnUse', orient: 'auto-start-reverse',
      }, defs);
      svgEl('path', { d: 'M0,0.5 L10,5 L0,9.5 z', class: `m-${k}` }, m);
    }

    let maxX = 0, maxY = 0, minY = Infinity;
    const grow = (x, y) => { maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); minY = Math.min(minY, y); };

    // Frames first so they sit behind everything
    for (const g of d.groups || []) {
      const ids = g.of || [];
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const id of ids) {
        const b = boxes[id];
        x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
        x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.h);
      }
      const p = g.pad ?? 12, top = g.top ?? 26;
      const [l, r, t, bm] = g.ext || [0, 0, 0, 0];
      const x = x0 - p - l, y = y0 - top - t, w = x1 - x0 + 2 * p + l + r, hh = y1 - y0 + top + p + t + bm;
      const gg = svgEl('g', { class: `g g-${g.k || 'plain'}` }, svg);
      svgEl('rect', { x, y, width: w, height: hh, rx: 12 }, gg);
      const lab = svgEl('text', { class: 'gl', x: g.right ? x + w - 10 : x + 10, y: y + 14, 'text-anchor': g.right ? 'end' : 'start' }, gg);
      lab.textContent = g.label;
      grow(x + w, y + hh); grow(x, y);
    }

    const edgeLayer = svgEl('g', {}, svg);
    const nodeLayer = svgEl('g', {}, svg);
    const labelLayer = svgEl('g', {}, svg);

    // Nodes
    for (const n of d.nodes) {
      const b = boxes[n.id];
      const g = svgEl('g', { class: `nd nd-${n.k || 'op'}` }, nodeLayer);
      if (n.stack) {
        for (const o of [6, 3]) svgEl('rect', { class: 'k', x: b.x + o, y: b.y - o, width: b.w, height: b.h, rx: 9 }, g);
      }
      svgEl('rect', { class: 'b', x: b.x, y: b.y, width: b.w, height: b.h, rx: n.round ? b.h / 2 : 9 }, g);
      const titles = String(n.t || '').split('\n').filter(Boolean);
      const subs = (Array.isArray(n.s) ? n.s : n.s ? [n.s] : []);
      const block = titles.length * 15 + subs.length * 12.5;
      let y = b.y + b.h / 2 - block / 2 + 11;
      const cx = b.x + b.w / 2;
      for (const t of titles) {
        const tx = svgEl('text', { class: 't', x: cx, y, 'text-anchor': 'middle' }, g);
        tx.textContent = t; y += 15;
      }
      y -= 1.5;
      for (const s of subs) {
        const tx = svgEl('text', { class: 's', x: cx, y, 'text-anchor': 'middle' }, g);
        tx.textContent = s; y += 12.5;
      }
      if (n.badge) {
        const bw = n.badge.length * 6 + 12;
        const bx = b.x + b.w - bw + 6 + (n.stack ? 6 : 0), by = b.y - 9 - (n.stack ? 6 : 0);
        const bg = svgEl('g', {}, labelLayer);
        svgEl('rect', { x: bx, y: by, width: bw, height: 17, rx: 8.5, fill: n.k === 'act' ? '#F4C15D' : '#4B2E0F' }, bg);
        const bt = svgEl('text', { x: bx + bw / 2, y: by + 12.2, 'text-anchor': 'middle', class: 'el', style: `fill:${n.k === 'act' ? '#4B2E0F' : '#FAF7F2'};font-weight:500` }, bg);
        bt.textContent = n.badge;
      }
      grow(b.x + b.w + (n.stack ? 6 : 0), b.y + b.h); grow(b.x, b.y - (n.stack ? 6 : 0) - (n.badge ? 10 : 0));
    }

    // Edges
    const port = (b, side, at = 0.5) => {
      switch (side) {
        case 'r': return [b.x + b.w, b.y + b.h * at];
        case 'l': return [b.x, b.y + b.h * at];
        case 't': return [b.x + b.w * at, b.y];
        default: return [b.x + b.w * at, b.y + b.h];
      }
    };
    const autoSides = (a, b) => {
      if (b.x >= a.x + a.w) return ['r', 'l'];
      if (b.x + b.w <= a.x) return ['l', 'r'];
      if (b.y >= a.y + a.h) return ['b', 't'];
      return ['t', 'b'];
    };
    const isH = (s) => s === 'l' || s === 'r';

    for (const e of d.edges || []) {
      const A = boxes[e.from], B = boxes[e.to];
      if (!A || !B) { console.warn('edge to unknown node', e); continue; }
      let [fs, ts] = autoSides(A, B);
      if (e.fs) fs = e.fs;
      if (e.ts) ts = e.ts;
      const S = port(A, fs, e.fa), E = port(B, ts, e.ta);
      let pts;
      if (e.via) {
        pts = [S];
        for (const v of e.via) pts.push([v.x !== undefined ? coord(v.x, 'x') : pts[pts.length - 1][0], v.y !== undefined ? coord(v.y, 'y') : pts[pts.length - 1][1]]);
        pts.push(E);
      } else if (isH(fs) && isH(ts)) {
        const fwd = (fs === 'r' && ts === 'l' && E[0] > S[0]) || (fs === 'l' && ts === 'r' && E[0] < S[0]);
        if (fwd) {
          const mx = e.mx !== undefined ? coord(e.mx, 'x') : (S[0] + E[0]) / 2;
          pts = Math.abs(S[1] - E[1]) < 0.5 ? [S, E] : [S, [mx, S[1]], [mx, E[1]], E];
        } else if (fs === ts) {
          const mx = e.mx !== undefined ? coord(e.mx, 'x') : (fs === 'r' ? Math.max(S[0], E[0]) + 20 : Math.min(S[0], E[0]) - 20);
          pts = [S, [mx, S[1]], [mx, E[1]], E];
        } else {
          const st = 16, my = coord(e.my ?? 'g99', 'y');
          const sx = S[0] + (fs === 'r' ? st : -st), ex = E[0] + (ts === 'l' ? -st : st);
          pts = [S, [sx, S[1]], [sx, my], [ex, my], [ex, E[1]], E];
        }
      } else if (!isH(fs) && !isH(ts)) {
        const fwd = (fs === 'b' && ts === 't' && E[1] > S[1]) || (fs === 't' && ts === 'b' && E[1] < S[1]);
        if (fwd) {
          const my = e.my !== undefined ? coord(e.my, 'y') : (S[1] + E[1]) / 2;
          pts = Math.abs(S[0] - E[0]) < 0.5 ? [S, E] : [S, [S[0], my], [E[0], my], E];
        } else if (fs === ts) {
          const my = e.my !== undefined ? coord(e.my, 'y') : (fs === 'b' ? Math.max(S[1], E[1]) + 20 : Math.min(S[1], E[1]) - 20);
          pts = [S, [S[0], my], [E[0], my], E];
        } else {
          const st = 14, mx = coord(e.mx ?? 'g99', 'x');
          const sy = S[1] + (fs === 'b' ? st : -st), ey = E[1] + (ts === 't' ? -st : st);
          pts = [S, [S[0], sy], [mx, sy], [mx, ey], [E[0], ey], E];
        }
      } else if (isH(fs)) {
        pts = e.mx !== undefined ? [S, [coord(e.mx, 'x'), S[1]], [coord(e.mx, 'x'), E[1] - (ts === 't' ? 14 : -14)], [E[0], E[1] - (ts === 't' ? 14 : -14)], E] : [S, [E[0], S[1]], E];
      } else {
        pts = e.my !== undefined ? [S, [S[0], coord(e.my, 'y')], [E[0] - (ts === 'l' ? 14 : -14), coord(e.my, 'y')], [E[0] - (ts === 'l' ? 14 : -14), E[1]], E] : [S, [S[0], E[1]], E];
      }

      const k = e.k || 'data';
      svgEl('path', {
        class: `e e-${k}`, d: roundedPath(pts, 7),
        'marker-end': e.arrow === false ? null : `url(#${uid}-${k})`,
        'marker-start': e.both ? `url(#${uid}-${k})` : null,
      }, edgeLayer);
      for (const p of pts) grow(p[0], p[1]);

      if (e.l) {
        // Label on the chosen (or longest) segment, over a paper-coloured chip
        let si = e.ls;
        if (si === undefined) {
          let best = -1;
          for (let i = 0; i < pts.length - 1; i++) {
            const len = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
            if (len > best) { best = len; si = i; }
          }
        }
        if (si < 0) si = pts.length - 1 + si;
        const t = e.lt ?? 0.5;
        const lx = pts[si][0] + (pts[si + 1][0] - pts[si][0]) * t + (e.lx || 0);
        const ly = pts[si][1] + (pts[si + 1][1] - pts[si][1]) * t + (e.ly || 0);
        const lines = String(e.l).split('\n');
        const wpx = Math.max(...lines.map((s) => s.length)) * (opts.mini ? 6.4 : 5.75) + 8;
        const hpx = lines.length * 11.5 + 3;
        const lg = svgEl('g', {}, labelLayer);
        svgEl('rect', { class: 'el-bg', x: lx - wpx / 2, y: ly - hpx / 2, width: wpx, height: hpx, rx: 3 }, lg);
        lines.forEach((s, i) => {
          const tx = svgEl('text', { class: 'el', x: lx, y: ly - hpx / 2 + 10.5 + i * 11.5, 'text-anchor': 'middle' }, lg);
          tx.textContent = s;
        });
        grow(lx + wpx / 2, ly + hpx / 2); grow(lx - wpx / 2, ly - hpx / 2);
      }
    }

    for (const t of d.texts || []) {
      const tx = svgEl('text', { class: t.cls || 'ax', x: coord(t.x, 'x') + (t.dx || 0), y: coord(t.y, 'y') + (t.dy || 0), 'text-anchor': t.anchor || 'start' }, labelLayer);
      tx.textContent = t.t;
    }

    const W = Math.ceil(maxX + G.px), H = Math.ceil(maxY + 14);
    const top = Math.floor(minY - 12);
    svg.setAttribute('viewBox', `0 ${top} ${W} ${H - top}`);
    if (opts.natural) natural(svg, W);
    return svg;
  }

  // Secondary figures render at their drawn size (never upscaled), shrinking only to fit
  function natural(svg, w) {
    svg.style.width = `${w}px`;
    svg.style.maxWidth = '100%';
    svg.style.minWidth = `${Math.min(w, 620)}px`;
  }

  function roundedPath(pts, r) {
    if (pts.length < 3) return `M${pts.map((p) => p.join(',')).join(' L')}`;
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
      const l1 = Math.hypot(x1 - x0, y1 - y0), l2 = Math.hypot(x2 - x1, y2 - y1);
      const rr = Math.min(r, l1 / 2, l2 / 2);
      if (rr < 0.5) { d += ` L${x1},${y1}`; continue; }
      const ax = x1 - ((x1 - x0) / l1) * rr, ay = y1 - ((y1 - y0) / l1) * rr;
      const bx = x1 + ((x2 - x1) / l2) * rr, by = y1 + ((y2 - y1) / l2) * rr;
      d += ` L${ax},${ay} Q${x1},${y1} ${bx},${by}`;
    }
    const last = pts[pts.length - 1];
    return `${d} L${last[0]},${last[1]}`;
  }

  // ---------------------------------------------------------------------------
  // Attention masks: rows are query groups, columns key groups.
  // rule[i][j]: 1 = attends, 0 = blocked, 'c' = causal inside the block,
  // 'd' = each token only sees itself.
  // ---------------------------------------------------------------------------
  function maskFigure(m) {
    const maxN = Math.max(...m.groups.map((g) => g.n));
    const big = m.size || 150;
    const size = (n) => Math.max(18, Math.round(big * Math.sqrt(n / maxN)));
    const sz = m.groups.map((g) => size(g.n));
    const left = 92, top = 58, total = sz.reduce((a, b) => a + b, 0);
    const svg = svgEl('svg', { class: 'dg', role: 'img', 'aria-label': plain(m.label || m.caption || 'Attention mask'), viewBox: `0 0 ${left + total + 40} ${top + total + 30}` });
    const off = [];
    sz.reduce((acc, s, i) => { off[i] = acc; return acc + s; }, 0);

    m.groups.forEach((g, i) => {
      const ty = svgEl('text', { class: 'ax-b', x: left - 8, y: top + off[i] + sz[i] / 2 + (sz[i] >= 30 ? -1 : 3.5), 'text-anchor': 'end' }, svg);
      ty.textContent = g.t;
      if (sz[i] >= 30) {
        const tn = svgEl('text', { class: 'ax', x: left - 8, y: top + off[i] + sz[i] / 2 + 11, 'text-anchor': 'end' }, svg);
        tn.textContent = g.nl || String(g.n);
      }
      const cx = left + off[i] + sz[i] / 2;
      const tx = svgEl('text', { class: 'ax-b', x: cx, y: top - 8, transform: `rotate(-35 ${cx} ${top - 8})` }, svg);
      tx.textContent = g.t;
    });
    const ql = svgEl('text', { class: 'ax', x: left - 8, y: top - 8, 'text-anchor': 'end' }, svg);
    ql.textContent = 'query ↓  key →';

    m.groups.forEach((_, i) => m.groups.forEach((__, j) => {
      const x = left + off[j], y = top + off[i], w = sz[j], hh = sz[i];
      const r = m.rule[i][j];
      svgEl('rect', { class: `mk-cell ${r === 1 ? 'mk-on' : 'mk-off'}`, x, y, width: w, height: hh }, svg);
      if (r === 'c') svgEl('polygon', { class: 'mk-on', points: `${x},${y} ${x + w},${y + hh} ${x},${y + hh}` }, svg);
      if (r === 'd') svgEl('line', { x1: x, y1: y, x2: x + w, y2: y + hh, stroke: '#4B2E0F', 'stroke-width': 2.5 }, svg);
    }));
    svgEl('rect', { class: 'mk-frame', x: left, y: top, width: total, height: total }, svg);
    const leg = svgEl('g', {}, svg);
    svgEl('rect', { x: left, y: top + total + 12, width: 10, height: 10, class: 'mk-on' }, leg);
    const l1 = svgEl('text', { class: 'ax', x: left + 15, y: top + total + 21 }, leg); l1.textContent = 'attends';
    svgEl('rect', { x: left + 72, y: top + total + 12, width: 10, height: 10, class: 'mk-off' }, leg);
    const l2 = svgEl('text', { class: 'ax', x: left + 87, y: top + total + 21 }, leg); l2.textContent = 'masked';
    natural(svg, left + total + 40);
    return svg;
  }

  // ---------------------------------------------------------------------------
  // Token strips: one row of sequence segments, with brackets marking the rows
  // a head reads. segs: [{t, n, k, w}], marks: [{a: [seg, frac], b: [seg, frac], l, k, side}]
  // ---------------------------------------------------------------------------
  function stripFigure(st) {
    const X0 = 16, Y0 = 26, H = 44;
    const xs = [];
    let x = X0;
    st.segs.forEach((sg) => { xs.push(x); x += sg.w; });
    const W = x + 16;
    const svg = svgEl('svg', { class: 'dg', role: 'img', 'aria-label': plain(st.label || 'Token layout') });
    st.segs.forEach((sg, i) => {
      const g = svgEl('g', { class: `nd nd-${sg.k || 'op'}` }, svg);
      svgEl('rect', { class: 'b', x: xs[i] + 1, y: Y0, width: sg.w - 2, height: H, rx: 6 }, g);
      const t = svgEl('text', { class: 't', x: xs[i] + sg.w / 2, y: Y0 + (sg.n ? 19 : 26), 'text-anchor': 'middle', style: 'font-size:11.5px' }, g);
      t.textContent = sg.t;
      if (sg.n) {
        const n = svgEl('text', { class: 's', x: xs[i] + sg.w / 2, y: Y0 + 33, 'text-anchor': 'middle' }, g);
        n.textContent = sg.n;
      }
    });
    let maxY = Y0 + H;
    const at = ([i, f]) => xs[i] + st.segs[i].w * f;
    (st.marks || []).forEach((m, j) => {
      const xa = at(m.a), xb = at(m.b);
      const below = m.side !== 't';
      const y0 = below ? Y0 + H + 6 + (m.lvl || 0) * 30 : Y0 - 6;
      const y1 = below ? y0 + 8 : y0 - 8;
      svgEl('path', { class: `e e-${m.k || 'attn'}`, d: `M${xa},${y0} L${xa},${y1} L${xb},${y1} L${xb},${y0}`, style: 'stroke-dasharray:none' }, svg);
      const lt = svgEl('text', { class: 'el', x: (xa + xb) / 2, y: below ? y1 + 13 : y1 - 5, 'text-anchor': 'middle', style: 'font-weight:500;fill:#4B2E0F' }, svg);
      lt.textContent = m.l;
      maxY = Math.max(maxY, y1 + 18);
    });
    svg.setAttribute('viewBox', `0 0 ${W} ${maxY + 6}`);
    natural(svg, W);
    return svg;
  }

  // ---------------------------------------------------------------------------
  // Layer schedules: one cell per layer, coloured by what it attends to.
  // layers: string of codes, kinds: {code: [node kind, label]}
  // ---------------------------------------------------------------------------
  function scheduleFigure(sc) {
    const cw = sc.cell || 22, ch = 28, gap = 3, X0 = 16, Y0 = 10;
    const codes = [...sc.layers];
    const svg = svgEl('svg', { class: 'dg', role: 'img', 'aria-label': plain(sc.label || 'Layer schedule') });
    codes.forEach((c, i) => {
      const g = svgEl('g', { class: `nd nd-${sc.kinds[c][0]}` }, svg);
      svgEl('rect', { class: 'b', x: X0 + i * (cw + gap), y: Y0, width: cw, height: ch, rx: 4 }, g);
      if (i % (sc.every || 2) === 0) {
        const n = svgEl('text', { class: 'ax', x: X0 + i * (cw + gap) + cw / 2, y: Y0 + ch + 13, 'text-anchor': 'middle' }, svg);
        n.textContent = i;
      }
    });
    let lx = X0;
    const ly = Y0 + ch + 34;
    for (const [code, [kind, label]] of Object.entries(sc.kinds)) {
      if (!codes.includes(code)) continue;
      const g = svgEl('g', { class: `nd nd-${kind}` }, svg);
      svgEl('rect', { class: 'b', x: lx, y: ly - 10, width: 12, height: 12, rx: 3 }, g);
      const t = svgEl('text', { class: 'ax', x: lx + 17, y: ly }, svg);
      t.textContent = label;
      lx += 17 + label.length * 5.75 + 22;
    }
    const W = Math.max(X0 + codes.length * (cw + gap) + 13, lx);
    svg.setAttribute('viewBox', `0 0 ${W} ${ly + 8}`);
    natural(svg, W);
    return svg;
  }

  const FIGS = {
    diagram: (f) => diagram(f.dg, { label: f.caption, natural: true }),
    mask: (f) => maskFigure(f),
    strip: (f) => stripFigure(f),
    schedule: (f) => scheduleFigure(f),
  };

  const figure = (svg, caption, cls = '') => {
    const f = h('figure', { class: `fig ${cls}`.trim() });
    const sc = h('div', { class: 'fig__scroll' });
    sc.appendChild(svg);
    f.appendChild(sc);
    if (caption) f.appendChild(h('figcaption', {}, caption));
    return f;
  };

  // ---------------------------------------------------------------------------
  // Legend
  // ---------------------------------------------------------------------------
  function legend(root) {
    const sw = (k) => {
      const s = svgEl('svg', { class: 'dg', width: 26, height: 16, viewBox: '0 0 26 16', 'aria-hidden': 'true' });
      const g = svgEl('g', { class: `nd nd-${k}` }, s);
      svgEl('rect', { class: 'b', x: 1, y: 1, width: 24, height: 14, rx: 4 }, g);
      return s;
    };
    const ln = (k) => {
      const s = svgEl('svg', { class: 'dg', width: 38, height: 12, viewBox: '0 0 38 12', 'aria-hidden': 'true' });
      const id = `lg-${k}`;
      const m = svgEl('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, markerUnits: 'userSpaceOnUse', orient: 'auto' }, svgEl('defs', {}, s));
      svgEl('path', { d: 'M0,0.5 L10,5 L0,9.5 z', class: `m-${k}` }, m);
      svgEl('path', { class: `e e-${k}`, d: 'M2,6 L35,6', 'marker-end': `url(#${id})` }, s);
      return s;
    };
    const fr = (k) => {
      const s = svgEl('svg', { class: 'dg', width: 26, height: 16, viewBox: '0 0 26 16', 'aria-hidden': 'true' });
      const g = svgEl('g', { class: `g g-${k}` }, s);
      svgEl('rect', { x: 1, y: 1, width: 24, height: 14, rx: 4 }, g);
      return s;
    };
    const item = (svg, text) => {
      const s = h('span', { class: 'legend__item' });
      s.appendChild(svg);
      s.appendChild(document.createTextNode(text));
      return s;
    };
    root.appendChild(h('span', { class: 'legend__title' }, 'Read the figures'));
    [['in', 'raw input'], ['vis', 'vision'], ['lang', 'language'], ['state', 'proprio state'], ['noise', 'noise · timestep'], ['core', 'fusion backbone'], ['act', 'action head'], ['out', 'actions'], ['skip', 'not run at inference']]
      .forEach(([k, t]) => root.appendChild(item(sw(k), t)));
    root.appendChild(h('span', { class: 'legend__sep' }));
    [['data', 'activations'], ['attn', 'attention read (KV · cross-attn)'], ['cond', 'conditioning (time · FiLM · AdaLN)'], ['loop', 'denoising feedback']]
      .forEach(([k, t]) => root.appendChild(item(ln(k), t)));
    root.appendChild(h('span', { class: 'legend__sep' }));
    root.appendChild(item(fr('once'), 'runs once per observation'));
    root.appendChild(item(fr('loop'), 'runs every denoising step'));
  }

  // ---------------------------------------------------------------------------
  // Page
  // ---------------------------------------------------------------------------
  const engTags = (engines) => `<span class="eng-tags">${engines.map((e) => `<span class="eng-tag eng-tag--${e}">${ENGINES[e]}</span>`).join('')}</span>`;
  const typeById = Object.fromEntries(DATA.archetypes.map((t) => [t.id, t]));
  const models = DATA.models;

  // Hero stats
  const setCount = (k, v) => document.querySelectorAll(`[data-count="${k}"]`).forEach((n) => { n.textContent = v; });
  setCount('models', models.length);
  setCount('cpp', models.filter((m) => m.engines.includes('cpp')).length);
  setCount('simd', models.filter((m) => m.engines.includes('simd')).length);
  setCount('types', DATA.archetypes.length);

  const legendRoot = document.querySelector('[data-legend]');
  if (legendRoot) legend(legendRoot);

  // Archetypes
  const typesRoot = document.querySelector('[data-types]');
  DATA.archetypes.forEach((t, i) => {
    const card = h('article', { class: 'type reveal', id: `type-${t.id}` });
    card.appendChild(h('span', { class: 'type__n' }, `${String(i + 1).padStart(2, '0')} · ${esc(t.short)}`));
    card.appendChild(h('h3', {}, esc(t.name)));
    const f = h('figure');
    f.appendChild(diagram(t.dg, { label: t.blurb, mini: true }));
    card.appendChild(f);
    card.appendChild(h('p', {}, t.blurb));
    const chips = h('div', { class: 'chips' });
    models.filter((m) => m.type === t.id).forEach((m) => chips.appendChild(h('a', { class: 'chip', href: `#${m.id}` }, esc(m.name))));
    card.appendChild(chips);
    typesRoot.appendChild(card);
  });

  // Compare table
  const tbody = document.querySelector('[data-compare]');
  for (const t of DATA.archetypes) {
    for (const m of models.filter((x) => x.type === t.id)) {
      const r = m.row;
      const tr = h('tr', { 'data-engines': m.engines.join(' ') });
      tr.innerHTML = `<td><a href="#${m.id}">${esc(m.name)}</a></td><td>${engTags(m.engines)}</td>`
        + `<td>${r.vision}</td><td>${r.backbone}</td><td>${r.head}</td><td>${r.decode}</td><td class="num">${r.chunk}</td>`;
      tbody.appendChild(tr);
    }
  }

  // Picker + model sections
  const side = document.querySelector('[data-side]');
  const list = document.querySelector('[data-models]');
  for (const t of DATA.archetypes) {
    const grp = h('div', { class: 'picker__group', 'data-type': t.id });
    grp.appendChild(h('span', { class: 'picker__group-title' }, esc(t.short)));
    for (const m of models.filter((x) => x.type === t.id)) {
      grp.appendChild(h('a', { href: `#${m.id}`, 'data-nav': m.id, 'data-engines': m.engines.join(' ') }, esc(m.name)));
      list.appendChild(modelSection(m, t));
    }
    side.appendChild(grp);
  }

  function modelSection(m, t) {
    const sec = h('article', { class: 'model', id: m.id, 'data-engines': m.engines.join(' ') });
    const head = h('header', { class: 'model__head' });
    head.appendChild(h('div', {}, `<p class="model__kicker">// ${esc(t.short)} · ${esc(m.kicker)}</p><h3 class="model__name">${esc(m.name)}${m.alt ? `<small>${esc(m.alt)}</small>` : ''}</h3>`));
    head.appendChild(h('div', {}, engTags(m.engines)));
    head.appendChild(h('p', { class: 'model__summary' }, m.summary));
    sec.appendChild(head);

    if (m.facts) {
      const ul = h('ul', { class: 'facts' });
      m.facts.forEach(([k, v]) => ul.appendChild(h('li', {}, `<span>${esc(k)}</span><b>${esc(v)}</b>`)));
      sec.appendChild(ul);
    }

    sec.appendChild(figure(diagram(m.dg, { label: m.caption }), m.caption));

    const extras = (m.figs || []).map((f) => figure(FIGS[f.type](f), f.caption, 'fig--natural'));
    if (extras.length > 1) {
      const pair = h('div', { class: `fig fig--pair${m.figsLayout ? ` fig--${m.figsLayout}` : ''}` });
      extras.forEach((e) => pair.appendChild(e));
      sec.appendChild(pair);
    } else if (extras.length === 1) {
      sec.appendChild(extras[0]);
    }

    if (m.specs) {
      const det = h('details', { class: 'specs' });
      det.appendChild(h('summary', {}, 'Layer-level specification'));
      const wrap = h('div', { class: 'spec-wrap' });
      const tbl = h('table', { class: 'spec' });
      tbl.innerHTML = '<thead><tr><th>Block</th><th>Architecture</th><th>Layers</th><th>Width</th><th>Heads (q / kv)</th><th>MLP</th><th>Notes</th></tr></thead>'
        + `<tbody>${m.specs.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="num">${r[2]}</td><td class="num">${r[3]}</td><td class="num">${r[4]}</td><td class="num">${r[5]}</td><td class="note">${r[6] || ''}</td></tr>`).join('')}</tbody>`;
      wrap.appendChild(tbl);
      det.appendChild(wrap);
      sec.appendChild(det);
    }

    if (m.notes) {
      const notes = h('div', { class: 'notes' });
      for (const [title, items] of m.notes) {
        const box = h('div', { class: 'note-box' });
        box.appendChild(h('h4', {}, esc(title)));
        box.appendChild(h('ul', {}, items.map((x) => `<li>${x}</li>`).join('')));
        notes.appendChild(box);
      }
      sec.appendChild(notes);
    }

    if (m.src) {
      const p = h('p', { class: 'src' });
      p.innerHTML = Object.entries(m.src).map(([e, files]) => `<span>${ENGINES[e]}</span>${files.map((f) => `<code>${esc(f)}</code>`).join('')}`).join('');
      sec.appendChild(p);
    }
    return sec;
  }

  // Engine filter
  const filterBtns = document.querySelectorAll('[data-filter]');
  filterBtns.forEach((btn) => btn.addEventListener('click', () => {
    const f = btn.dataset.filter;
    filterBtns.forEach((b) => { const on = b === btn; b.classList.toggle('is-active', on); b.setAttribute('aria-pressed', String(on)); });
    const show = (n) => f === 'all' || n.dataset.engines.split(' ').includes(f);
    document.querySelectorAll('.model, [data-nav], [data-compare] tr').forEach((n) => { n.hidden = !show(n); });
    document.querySelectorAll('.picker__group').forEach((g) => { g.hidden = !g.querySelector('a:not([hidden])'); });
  }));

  // Scrollspy for the sidebar
  const navLinks = Object.fromEntries([...document.querySelectorAll('[data-nav]')].map((a) => [a.dataset.nav, a]));
  const seen = new Map();
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((en) => seen.set(en.target.id, en.isIntersecting ? en.boundingClientRect.top : null));
    let best = null, bestTop = Infinity;
    for (const [id, top] of seen) if (top !== null && Math.abs(top) < bestTop) { best = id; bestTop = Math.abs(top); }
    if (!best) return;
    Object.values(navLinks).forEach((a) => a.classList.remove('is-current'));
    const cur = navLinks[best];
    if (!cur) return;
    cur.classList.add('is-current');
    // Keep the current chip in view inside the horizontally scrolling picker
    const bar = cur.closest('.picker__groups');
    const l = cur.offsetLeft - bar.offsetLeft, r = l + cur.offsetWidth;
    if (l < bar.scrollLeft || r > bar.scrollLeft + bar.clientWidth) bar.scrollTo({ left: l - 40, behavior: 'smooth' });
  }, { rootMargin: '-90px 0px -55% 0px' });
  document.querySelectorAll('.model').forEach((s) => spy.observe(s));

  // Mobile nav, footer year, reveal-on-scroll (as on the main site)
  const toggle = document.querySelector('.nav__toggle');
  const links = document.querySelector('.nav__links');
  toggle?.addEventListener('click', () => {
    const open = links.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  links?.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => {
    links.classList.remove('is-open');
    toggle.setAttribute('aria-expanded', 'false');
  }));
  const year = document.getElementById('year');
  if (year) year.textContent = new Date().getFullYear();
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) { en.target.classList.add('is-visible'); io.unobserve(en.target); }
  }), { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach((n) => io.observe(n));
})();
