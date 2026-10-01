# VLA Arch

Block diagrams of every vision-language-action policy served by
[vla.cpp](https://github.com/VinRobotics/vla.cpp) and
[vla.simd](https://github.com/cair-vinuni/vla.simd), drawn from the engines' C++
forward passes. It uses the theme of the main site ([havi.fit](https://havi.fit/)).
There is no build step: open `index.html`, or serve the folder with any static server.

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Files

| Path | What it holds |
|---|---|
| `index.html` | Page shell: hero, archetypes, comparison table, model sections |
| `assets/site.css` | Main-site palette and fonts, plus the diagram styles |
| `assets/arch.js` | Renders every figure from data as inline SVG, and runs the page (picker, engine filter, scrollspy) |
| `data/archetypes.js` | The six ways an action head reads the scene, with their mini diagrams |
| `data/{flow,dit,parallel,readout,query,unet}.js` | One file per archetype; each pushes its models |

## Diagram data

A model's `dg` places nodes on a grid and connects their sides:

```js
{ id: 'vit', k: 'vis', t: 'SigLIP So400m/14', s: ['d 1152 · 16 heads'], c: 1, r: 0, stack: true, badge: '×27' }
{ from: 'vlm', to: 'aex', k: 'attn', fs: 'b', ts: 't', l: 'layer-i K, V' }
```

- Nodes: `c`, `r` are column and row (fractions allowed); `w`, `h` are spans in cells; `k` is the role
  (`in`, `vis`, `lang`, `state`, `noise`, `core`, `act`, `out`, `op`, `skip`).
- Edges: `fs` and `ts` pick the sides (`l`, `r`, `t`, `b`), and `fa` and `ta` the position along them (0–1).
  `mx` and `my` set the routing channel, in grid units or `'gN'` for the gutter before column or row N.
  `k` is `data`, `attn`, `cond`, `loop` or `skip`.
- Groups: `once` and `loop` frames mark what runs once per observation and what repeats every denoising step.

A model's `figs` add secondary figures: `mask` (attention mask), `strip` (token layout), `schedule`
(per-layer attention pattern) or `diagram`.
