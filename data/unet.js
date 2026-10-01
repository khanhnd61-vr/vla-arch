// Convolutional U-Net diffusion: Diffusion Policy.
// Sources: vla.simd src/models/diffusion/* (rgb_encoder, unet1d, scheduler, diffusion_model).

window.VLA_ARCH.models.push({
  id: 'diffusion-policy', name: 'Diffusion Policy', alt: 'diffusion', engines: ['simd'], type: 'unet',
  kicker: '1-D temporal U-Net, DDIM',
  summary: "Diffusion Policy with a convolutional backbone. Two frames of keypoints and state collapse into a 268-d condition that <b>FiLM-modulates a 1-D temporal U-Net</b>; DDIM refines a 64-step noise chunk in <b>10 steps</b>.",
  facts: [['Cameras', '2 × 480 × 640'], ['History', '2 frames'], ['Condition', '268 + 128 = 396'], ['Decode', 'DDIM · 10 steps'], ['Chunk', '64 × 6 (32 run)'], ['i5-12400F', '500 / 178 ms (FP32 / INT8)']],
  row: { vision: 'ResNet-18 × 2 + spatial softmax', backbone: 'none: one global condition vector', head: 'conditional 1-D U-Net · FiLM', decode: 'DDIM · 10 steps', chunk: '64 × 6 (32 run)' },
  dg: {
    nodes: [
      { id: 'cam', k: 'in', t: 'Cameras × 2 steps', s: ['front + wrist', '480 × 640 · no crop'], c: 0, r: 0 },
      { id: 'enc', k: 'vis', t: 'ResNet-18 × 2', s: ['one per camera', 'BN folded'], c: 1, r: 0 },
      { id: 'kp', k: 'op', t: 'Spatial softmax', s: ['32 keypoints → 64', 'Linear + ReLU'], c: 2, r: 0 },
      { id: 'state', k: 'in', t: 'Robot state × 2 steps', s: ['6-d · min / max'], c: 0, r: 1 },
      { id: 'gc', k: 'op', t: 'Global condition', s: ['per step 6 + 64 + 64', '× 2 steps = 268'], c: 3, r: 0, h: 2 },
      { id: 'tt', k: 'noise', t: 'Timestep t', s: ['sin 128 → MLP → 128'], c: 1, r: 2.9 },
      { id: 'cond', k: 'op', t: 'cond = [t ∥ obs]', s: ['396 → Mish', 'per block → scale, bias'], c: 2, r: 2.9 },
      { id: 'xt', k: 'noise', t: 'Noisy chunk xₜ', s: ['64 × 6', 'starts at N(0, I)'], c: 0, r: 3.9 },
      { id: 'unet', k: 'act', t: 'Conditional 1-D U-Net', s: ['512 → 1024 → 2048', 'kernel 5 · GN 8 · Mish', '12 FiLM res blocks', 'predicts ε'], c: 3, r: 2.9, h: 2 },
      { id: 'ddim', k: 'op', t: 'DDIM step', s: ['t = 90, 80 … 0', 'η = 0 · clip x̂₀'], c: 4, r: 3.9 },
      { id: 'out', k: 'out', t: 'Action chunk', s: ['run 32 of 64 steps', 'min / max un-norm'], c: 5, r: 1 },
    ],
    edges: [
      { from: 'cam', to: 'enc' }, { from: 'enc', to: 'kp' }, { from: 'kp', to: 'gc', ta: 0.193 },
      { from: 'state', to: 'gc', ta: 0.807, l: '2 × 6' },
      { from: 'gc', to: 'cond', k: 'cond', fs: 'b', fa: 0.12, ts: 't', my: 'g2.9-40', l: 'obs 268' },
      { from: 'tt', to: 'cond' },
      { from: 'cond', to: 'unet', k: 'cond', ta: 0.193 },
      { from: 'xt', to: 'unet', ta: 0.807, l: 'xₜ' },
      { from: 'unet', to: 'ddim', fa: 0.807 },
      { from: 'ddim', to: 'out', fs: 't', ts: 'l', l: 'after 10 steps' },
      { from: 'ddim', to: 'xt', k: 'loop', fs: 'b', ts: 'b', my: 'g5', l: 'xₜ₋₁₀ · repeat × 10' },
    ],
    groups: [
      { k: 'once', label: 'observation · once per call (frame features cached)', of: ['cam', 'enc', 'kp', 'state', 'gc'] },
      { k: 'loop', label: 'denoiser · every DDIM step (× 10)', of: ['tt', 'cond', 'unet', 'ddim'] },
    ],
  },
  caption: '<b>The observation is encoded once; the U-Net runs ten times.</b> Keypoint features from both cameras over two frames, plus state, form one global condition. With the timestep embedding it FiLM-modulates every residual block of the U-Net at each DDIM step.',
  figs: [{
    type: 'diagram',
    dg: {
      grid: { cw: 104, gx: 24, rh: 46, gy: 30, px: 16, py: 16 },
      nodes: [
        { id: 'd0', k: 'act', t: 'Down 0', s: ['6 → 512 · T 64'], c: 0, r: 0 },
        { id: 'd1', k: 'act', t: 'Down 1', s: ['→ 1024 · T 32'], c: 1, r: 1 },
        { id: 'd2', k: 'act', t: 'Down 2', s: ['→ 2048 · T 16'], c: 2, r: 2 },
        { id: 'mid', k: 'act', t: 'Mid', s: ['2048 · T 16'], c: 3, r: 2 },
        { id: 'u0', k: 'act', t: 'Up 0', s: ['4096 → 1024'], c: 4, r: 2 },
        { id: 'u1', k: 'act', t: 'Up 1', s: ['2048 → 512'], c: 5, r: 1 },
        { id: 'fin', k: 'op', t: 'Final conv', s: ['512 → 6 · T 64'], c: 6, r: 0 },
      ],
      edges: [
        { from: 'd0', to: 'd1', l: '↓2' }, { from: 'd1', to: 'd2', l: '↓2' }, { from: 'd2', to: 'mid' }, { from: 'mid', to: 'u0' },
        { from: 'u0', to: 'u1', l: '↑2' }, { from: 'u1', to: 'fin', l: '↑2' },
        { from: 'd2', to: 'u0', k: 'data', fs: 'b', ts: 'b', l: 'skip: concat 2048' },
        { from: 'd1', to: 'u1', k: 'data', l: 'skip: concat 1024' },
        { from: 'd0', to: 'fin', k: 'skip', l: 'Down 0 skip unused' },
      ],
    },
    caption: '<b>The U-Net runs along the chunk\'s time axis.</b> Each stage is two FiLM-modulated residual blocks. Two stride-2 convolutions shrink the 64-step horizon to 16 while channels widen to 2048; the up path concatenates the matching down features back in.',
  }],
  specs: [
    ['Backbone', 'ResNet-18 × 2 (per camera)', '8 blocks', '64 → 512', '–', 'ReLU', 'BN folded; 480 × 640 → 15 × 20 × 512; no crop'],
    ['Keypoints', '1×1 conv + spatial softmax', '1', '512 → 32 → 64', '–', '–', 'expected (x, y) per keypoint, then Linear 64 → 64 + ReLU'],
    ['Condition', 'concat', '–', '2 × (6 + 64 + 64) = 268', '–', '–', 'state min / max to [-1, 1]'],
    ['Time', 'sinusoid + MLP', '2', '128 → 512 → 128', '–', 'Mish', ''],
    ['U-Net down', 'Conv1d res blocks', '6', '512 / 1024 / 2048', '–', 'Mish', 'kernel 5, GroupNorm 8, stride-2 conv between stages'],
    ['U-Net mid', 'Conv1d res blocks', '2', '2048', '–', 'Mish', 'T = 16'],
    ['U-Net up', 'Conv1d res blocks', '4', '1024 / 512', '–', 'Mish', 'concat skips; transposed conv ×2'],
    ['Output', 'conv block + 1×1', '2', '512 → 6', '–', '–', 'ε prediction over 64 steps'],
  ],
  notes: [
    ['In vla.simd', [
      'Each residual block has its own Linear 396 → 2C for FiLM scale and bias (applied as scale·h + bias, no "1 +").',
      'Encoder features are reused when a frame is byte-identical to one already encoded, so a sliding 2-frame window re-encodes only the newest frame per camera.',
      '<code>DIFFUSION_INT8</code> quantises U-Net and ResNet convs: 499.6 ms FP32 → 178.1 ms INT8 on an i5-12400F; 4.27 s FP32 on a Raspberry Pi 5.',
    ]],
    ['Inputs and outputs', [
      'Trained as DDPM with 100 squared-cosine steps; served with DDIM, 10 steps, η = 0 (<code>DP_SCHEDULER</code> / <code>DP_STEPS</code> switch back).',
      'History of 2 observations; the server repeats the oldest frame at episode start.',
      'Horizon 64; executes 32 actions starting at index 1, min / max un-normalised.',
    ]],
  ],
  src: { simd: ['src/models/diffusion/rgb_encoder.cpp', 'unet1d.cpp', 'scheduler.cpp', 'diffusion_model.cpp'] },
});
