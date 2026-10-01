// VLA Arch data: the six ways an action head reads the scene.
// Each model file (flow.js, dit.js, ...) pushes into VLA_ARCH.models with a `type`
// naming one of these archetypes. Diagram grammar: see assets/arch.js.

const MINI = { cw: 96, gx: 34, rh: 34, gy: 46, px: 12, py: 18 };

window.VLA_ARCH = {
  models: [],
  archetypes: [
    {
      id: 'flow', short: 'flow expert',
      name: "Flow-matching expert on the VLM's KV cache",
      blurb: 'A smaller expert runs beside the VLM layer for layer and attends to the cached keys and values of the prefix. The prefix runs once; only the expert repeats for each Euler step.',
      dg: {
        grid: MINI,
        nodes: [
          { id: 'o', k: 'in', t: 'Obs + prompt', c: 0, r: 0 },
          { id: 'v', k: 'core', t: 'VLM', c: 1, r: 0, stack: true },
          { id: 'x', k: 'noise', t: 'xₜ', c: 0, r: 1 },
          { id: 'e', k: 'act', t: 'Expert', c: 1, r: 1, stack: true },
          { id: 'u', k: 'out', t: 'actions', c: 2, r: 1 },
        ],
        edges: [
          { from: 'o', to: 'v' },
          { from: 'v', to: 'e', k: 'attn', fs: 'b', ts: 't', l: 'K,V per layer' },
          { from: 'x', to: 'e' },
          { from: 'e', to: 'x', k: 'loop', fs: 'b', ts: 'b', l: '× N steps' },
          { from: 'e', to: 'u' },
        ],
      },
    },
    {
      id: 'dit', short: 'DiT head',
      name: 'Diffusion transformer with cross-attention',
      blurb: 'A separate DiT denoises the chunk. It reads VLM features through cross-attention and the timestep through AdaLN; the cross-attention K, V are computed once and reused at every step.',
      dg: {
        grid: MINI,
        nodes: [
          { id: 'o', k: 'in', t: 'Obs + prompt', c: 0, r: 0 },
          { id: 'v', k: 'core', t: 'VLM', c: 1, r: 0, stack: true },
          { id: 't', k: 'noise', t: 'time t', c: 2, r: 0 },
          { id: 'x', k: 'noise', t: 'xₜ', c: 0, r: 1 },
          { id: 'd', k: 'act', t: 'DiT', c: 1, r: 1, stack: true },
          { id: 'u', k: 'out', t: 'actions', c: 2, r: 1 },
        ],
        edges: [
          { from: 'o', to: 'v' },
          { from: 'v', to: 'd', k: 'attn', fs: 'b', ts: 't', fa: 0.35, ta: 0.35, l: 'cross-attn' },
          { from: 't', to: 'd', k: 'cond', fs: 'b', ts: 't', ta: 0.85 },
          { from: 'x', to: 'd' },
          { from: 'd', to: 'x', k: 'loop', fs: 'b', ts: 'b', l: '× N steps' },
          { from: 'd', to: 'u' },
        ],
      },
    },
    {
      id: 'parallel', short: 'parallel decode',
      name: 'Parallel decoding inside the LLM',
      blurb: "Empty action slots ride in the LLM's own sequence, and the mask lets them see the whole prompt. One forward pass, then an MLP regresses the chunk from the slots' hidden states. No loop.",
      dg: {
        grid: MINI,
        nodes: [
          { id: 'o', k: 'in', t: 'Obs + prompt', c: 0, r: 0 },
          { id: 's', k: 'in', t: 'empty slots', c: 0, r: 1 },
          { id: 'l', k: 'core', t: 'LLM', s: 'one pass', c: 1, r: 0, h: 2, stack: true },
          { id: 'u', k: 'out', t: 'actions', c: 2, r: 0 },
          { id: 'hd', k: 'act', t: 'MLP head', c: 2, r: 1 },
        ],
        edges: [
          { from: 'o', to: 'l', ta: 0.149 },
          { from: 's', to: 'l', ta: 0.851 },
          { from: 'l', to: 'hd', fa: 0.851 },
          { from: 'hd', to: 'u', fs: 't', ts: 'b' },
        ],
      },
    },
    {
      id: 'readout', short: 'readout + diffusion',
      name: 'Readout token feeding a diffusion head',
      blurb: 'A block-masked transformer writes into a readout token that no other token can read. That one vector conditions a small MLP denoiser, so each DDPM step is a handful of matmuls.',
      dg: {
        grid: MINI,
        nodes: [
          { id: 'o', k: 'in', t: 'Obs + task', c: 0, r: 0 },
          { id: 'tr', k: 'core', t: 'Transformer', c: 1, r: 0, stack: true },
          { id: 'x', k: 'noise', t: 'xₜ', c: 0, r: 1 },
          { id: 'n', k: 'act', t: 'MLP ε-net', c: 1, r: 1 },
          { id: 'u', k: 'out', t: 'actions', c: 2, r: 1 },
        ],
        edges: [
          { from: 'o', to: 'tr' },
          { from: 'tr', to: 'n', fs: 'b', ts: 't', l: 'readout' },
          { from: 'x', to: 'n' },
          { from: 'n', to: 'x', k: 'loop', fs: 'b', ts: 'b', l: '× 20 steps' },
          { from: 'n', to: 'u' },
        ],
      },
    },
    {
      id: 'query', short: 'learned queries',
      name: 'Encoder-decoder with learned action queries',
      blurb: "ACT-style: an encoder fuses camera, state and text tokens; a decoder's fixed query tokens cross-attend to that memory and become the chunk in one deterministic pass.",
      dg: {
        grid: MINI,
        nodes: [
          { id: 'o', k: 'in', t: 'Obs + text', c: 0, r: 0 },
          { id: 'en', k: 'core', t: 'Encoder', c: 1, r: 0, stack: true },
          { id: 'q', k: 'in', t: 'queries', c: 0, r: 1 },
          { id: 'de', k: 'act', t: 'Decoder', c: 1, r: 1, stack: true },
          { id: 'u', k: 'out', t: 'chunk', c: 2, r: 1 },
        ],
        edges: [
          { from: 'o', to: 'en' },
          { from: 'en', to: 'de', k: 'attn', fs: 'b', ts: 't', l: 'cross-attn' },
          { from: 'q', to: 'de' },
          { from: 'de', to: 'u' },
        ],
      },
    },
    {
      id: 'unet', short: 'conv U-Net',
      name: 'Convolutional U-Net diffusion',
      blurb: 'Diffusion Policy: image keypoints and state collapse into one global condition that FiLM-modulates every block of a 1-D temporal U-Net, iterated with DDIM.',
      dg: {
        grid: MINI,
        nodes: [
          { id: 'o', k: 'in', t: 'Obs history', c: 0, r: 0 },
          { id: 'en', k: 'vis', t: 'ResNet', c: 1, r: 0 },
          { id: 'x', k: 'noise', t: 'xₜ', c: 0, r: 1 },
          { id: 'un', k: 'act', t: '1-D U-Net', c: 1, r: 1 },
          { id: 'u', k: 'out', t: 'actions', c: 2, r: 1 },
        ],
        edges: [
          { from: 'o', to: 'en' },
          { from: 'en', to: 'un', k: 'cond', fs: 'b', ts: 't', l: 'FiLM' },
          { from: 'x', to: 'un' },
          { from: 'un', to: 'x', k: 'loop', fs: 'b', ts: 'b', l: '× 10 steps' },
          { from: 'un', to: 'u' },
        ],
      },
    },
  ],
};
