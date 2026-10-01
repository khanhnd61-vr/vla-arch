// Readout token + diffusion head: Octo (octo-small).
// Sources: vla.cpp src/models/octo.cpp; vla.simd src/models/octo/*, src/nn/t5_encoder.cpp.

window.VLA_ARCH.models.push({
  id: 'octo', name: 'Octo', alt: 'octo-small', engines: ['cpp', 'simd'], type: 'readout',
  kicker: 'block-masked ViT-S, DDPM head',
  summary: "A generalist transformer policy trained on Open X-Embodiment. Conv stems and a frozen T5 feed a ViT-S whose <b>block-wise mask</b> funnels everything into one readout token per timestep; a 3-block MLP denoiser turns that vector into a 4-step chunk over <b>20 DDPM steps</b>.",
  facts: [['Cameras', 'primary 256² + wrist 128²'], ['Window', '1 (cpp) · 2 (simd)'], ['Tokens / step', '256 + 64 + 16 + 1'], ['Decode', 'DDPM · 20 steps'], ['Chunk', '4 × 7 / 4 × 6'], ['SO-101 ckpt', '136.7M params']],
  row: { vision: 'SmallStem16 conv · 256 + 64 tok', backbone: 'T5-base + ViT-S 12 L, block mask', head: 'MLP-ResNet ε-net on the readout', decode: 'DDPM · 20 steps', chunk: '4 × 7 / 4 × 6' },
  dg: {
    nodes: [
      { id: 'cam1', k: 'in', t: 'Primary camera', s: ['256 × 256 · [-1, 1]', '+ 3 goal channels = −1'], c: 0, r: 0 },
      { id: 'stem1', k: 'vis', t: 'SmallStem16', s: ['4 × conv s2 · GN · ReLU', '→ 256 tok × 512'], c: 1, r: 0 },
      { id: 'cam2', k: 'in', t: 'Wrist camera', s: ['128 × 128 · optional'], c: 0, r: 1 },
      { id: 'stem2', k: 'vis', t: 'SmallStem16', s: ['separate weights', '→ 64 tok × 512'], c: 1, r: 1 },
      { id: 'txt', k: 'in', t: 'Instruction', s: ['T5 SPM · 16 tokens'], c: 0, r: 2 },
      { id: 't5', k: 'lang', t: 'T5-base encoder', s: ['d 768 · frozen', 'cached per instruction'], c: 1, r: 2, stack: true, badge: '×12' },
      { id: 'proj', k: 'op', t: 'Project + position', s: ['each stream → 384', 'learned pos per', 'timestep and group', '+ 1 readout / step'], c: 2, r: 0, h: 3 },
      { id: 'tf', k: 'core', t: 'ViT-S transformer', s: ['d 384 · 6 heads', 'MLP 1536 · pre-LN', 'block-wise mask', 'readout-only last layer'], c: 3, r: 0, h: 3, stack: true, badge: '×12' },
      { id: 'ro', k: 'op', t: 'Readout token', s: ['last timestep', '384-d'], c: 4, r: 1 },
      { id: 'xt', k: 'noise', t: 'Noisy chunk xₜ', s: ['4 × 7 or 4 × 6', 'starts at N(0, I)'], c: 0, r: 3.7 },
      { id: 'cat', k: 'op', t: 'Concat', s: ['[t 32 ∥ readout 384', '∥ xₜ 28] = 444'], c: 2, r: 3.7 },
      { id: 'den', k: 'act', t: 'MLP-ResNet ε-net', s: ['→ 256 · 3 blocks', 'LN → 1024 → SiLU'], c: 3, r: 3.7, stack: true, badge: '×3' },
      { id: 'ddpm', k: 'op', t: 'DDPM update', s: ['cosine β · clip ±5', 't = 19 → 0'], c: 4, r: 3.7 },
      { id: 'out', k: 'out', t: 'Action chunk', s: ['4 × 7 (cpp) · 4 × 6', 'mean / std un-norm'], c: 5, r: 2 },
      { id: 'temb', k: 'noise', t: 'Time step t', s: ['Fourier → MLP → 32', 'table built at load'], c: 1, r: 4.75 },
    ],
    edges: [
      { from: 'cam1', to: 'stem1' }, { from: 'stem1', to: 'proj', ta: 0.119 },
      { from: 'cam2', to: 'stem2' }, { from: 'stem2', to: 'proj' },
      { from: 'txt', to: 't5' }, { from: 't5', to: 'proj', ta: 0.881 },
      { from: 'proj', to: 'tf' }, { from: 'tf', to: 'ro' },
      { from: 'ro', to: 'cat', fs: 'b', fa: 0.3, ts: 't', my: 'g3.7-35', l: 'readout embedding (computed once)' },
      { from: 'xt', to: 'cat', l: 'xₜ' },
      { from: 'temb', to: 'cat', k: 'cond', fs: 'r', ts: 'b' },
      { from: 'cat', to: 'den' }, { from: 'den', to: 'ddpm' },
      { from: 'ddpm', to: 'out', fs: 't', fa: 0.7, ts: 'l', l: 'after 20 steps' },
      { from: 'ddpm', to: 'xt', k: 'loop', fs: 'b', ts: 'b', my: 'g6', l: 'xₜ₋₁ · repeat × 20' },
    ],
    groups: [
      { k: 'once', label: 'observation · runs once per call', of: ['cam1', 'stem1', 'cam2', 'stem2', 'txt', 't5', 'proj', 'tf', 'ro'] },
      { k: 'loop', label: 'diffusion head · every DDPM step (× 20)', of: ['cat', 'den', 'ddpm', 'temb'] },
    ],
  },
  caption: '<b>The transformer runs once; the head loops.</b> Each denoising step concatenates the time embedding, the 384-d readout vector and the noisy 4-step chunk, then runs three MLP-ResNet blocks. Twenty of those steps cost far less than one transformer pass.',
  figsLayout: null,
  figs: [
    {
      type: 'strip',
      segs: [
        { t: 'task', n: '16', w: 52, k: 'lang' },
        { t: 'primary', n: '256', w: 118, k: 'vis' }, { t: 'wrist', n: '64', w: 62, k: 'vis' }, { t: 'lang', n: '16', w: 50, k: 'lang' }, { t: 'readout', n: '1', w: 56, k: 'act' },
        { t: 'primary', n: '256', w: 118, k: 'vis' }, { t: 'wrist', n: '64', w: 62, k: 'vis' }, { t: 'lang', n: '16', w: 50, k: 'lang' }, { t: 'readout', n: '1', w: 56, k: 'act' },
      ],
      marks: [
        { a: [1, 0], b: [4, 1], side: 't', k: 'data', l: 'timestep 0' },
        { a: [5, 0], b: [8, 1], side: 't', k: 'data', l: 'timestep 1' },
        { a: [8, 0.05], b: [8, 0.95], k: 'attn', l: 'to the head' },
      ],
      caption: '<b>Sequence for a two-step window</b> (vla.simd, SO-101): 16 + 2 × 337 = 690 tokens. The language tokens are repeated inside every timestep without a new position embedding.',
    },
    {
      type: 'mask', size: 110,
      groups: [{ t: 'task', n: 16 }, { t: 'obs₀', n: 336, nl: '336' }, { t: 'ro₀', n: 1 }, { t: 'obs₁', n: 336, nl: '336' }, { t: 'ro₁', n: 1 }],
      rule: [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 1, 1, 1]],
      caption: '<b>Block-wise mask.</b> Task tokens see only themselves; observations see the task and observations up to their step; readouts also see earlier readouts, and nothing else reads a readout.',
    },
  ],
  specs: [
    ['Image stem', 'SmallStem16, per camera', '4 conv', '6 → 32 → 96 → 192 → 384', '–', 'ReLU', 'weight-standardised conv s2 + GroupNorm 32; 1×1 → 512; patch 16'],
    ['Text', 'T5-base encoder', '12', '768', '12 / 12 × 64', '3072 ReLU', 'frozen, pre-RMSNorm, shared relative position bias; 16 tokens'],
    ['Projections', 'Linear + learned position', '–', '→ 384', '–', '–', 'per stream; positions indexed by timestep (max 10)'],
    ['Transformer', 'ViT-S', '12', '384', '6 / 6 × 64', '1536 GELU', 'pre-LN, block-wise mask; the last layer computes only the readout row'],
    ['Time', 'learned Fourier + MLP', '2', '32 → 64 → 32', '–', 'SiLU', 'table of the 20 steps built at load'],
    ['Denoiser', 'MLP-ResNet', '3 blocks', '444 → 256', '–', '1024 SiLU', 'LN → 1024 → SiLU → 256 + residual; out 28 (24 on SO-101)'],
  ],
  notes: [
    ['In vla.cpp (LIBERO checkpoint)', [
      'JAX fine-tune: window 1, primary camera only, 4 × 7, no proprio. Only octo-small is supported.',
      'Each stage is its own cached graph; the 20-step reverse chain is one graph, one submission per frame.',
      'Padded timesteps are dropped rather than masked: a window-2 sequence shrinks from 690 to 370 tokens at cold start.',
      'Un-normalises server-side with the dataset statistics in the GGUF. RTX 3090: 4.4 ms.',
    ]],
    ['In vla.simd (SO-101 checkpoint)', [
      'Window 2, primary + wrist, 4 × 6; 136.7M parameters, FP32.',
      'Stem outputs are reused when a frame is byte-identical to one already encoded, so a sliding window encodes only the new frame.',
      'T5 output cached per instruction (64 entries); the mask is cached per window shape.',
      '<code>OCTO_INT8</code> picks W8A8 per block: 52.7 ms FP32 → 29.8 ms INT8 on an i7-14700F.',
    ]],
  ],
  src: { cpp: ['src/models/octo.cpp'], simd: ['src/models/octo/small_stem.cpp', 'octo_transformer.cpp', 'diffusion_head.cpp', 'octo_model.cpp'] },
});
