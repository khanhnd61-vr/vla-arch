// One-pass parallel decoding inside the LLM: OpenVLA-OFT, VLA-Adapter, BitVLA.
// Sources: vla.cpp src/models/{openvla_oft,vla_adapter,bitvla}.cpp, src/modules/dual_tower.h,
// src/kernels/bitvla/*, scripts/convert_*.py, scripts/tokenize_prompt.py.

// Fractions of a 5-row (h: 5) and 4-row (h: 4) node at which each row's centre sits
const R5 = [0.068, 0.284, 0.5, 0.716, 0.932];
const R4 = [0.087, 0.362, 0.638, 0.913];

window.VLA_ARCH.models.push(
  // ---------------------------------------------------------------------------
  {
    id: 'openvla-oft', name: 'OpenVLA-OFT', alt: 'openvla_oft', engines: ['cpp'], type: 'parallel',
    kicker: 'Llama-2 7B, 56 slots, L1 regression',
    summary: "OpenVLA fine-tuned for speed: a Prismatic dual vision tower and Llama-2 7B, with autoregressive action tokens replaced by <b>56 empty slots decoded in one bidirectional pass</b> and an MLP that regresses an 8-step chunk with L1.",
    facts: [['Views', '2 × 224²'], ['Sequence', '512 img + 1 + L + 56'], ['Passes', '1'], ['Head', 'MLPResNet · L1'], ['Chunk', '8 × 7'], ['RTX 3090', '135 ms · 14.8 GB']],
    row: { vision: 'DINOv2-L + SigLIP so400m · 256 tok/view', backbone: 'Llama-2 7B · 32 L, bidirectional', head: 'MLPResNet on 56 slot states', decode: 'one pass · L1', chunk: '8 × 7' },
    dg: {
      nodes: [
        { id: 'cam', k: 'in', t: 'Cameras', s: ['2 views · 224 × 224', 'centre-crop 90 %'], c: 0, r: 0 },
        { id: 'dino', k: 'vis', t: 'DINOv2 ViT-L/14', s: ['d 1024 · 4 registers', '23 of 24 blocks'], c: 1, r: 0, stack: true, badge: '×23' },
        { id: 'sig', k: 'vis', t: 'SigLIP so400m/14', s: ['d 1152 · 26 of 27', '256 patches / view'], c: 1, r: 1, stack: true, badge: '×26' },
        { id: 'proj', k: 'op', t: 'Concat + MLP', s: ['2176-d per patch', '→ 8704 → 4096 → 4096', '256 tok / view'], c: 2, r: 0, h: 2 },
        { id: 'txt', k: 'in', t: 'Instruction', s: ['"In: What action should', 'the robot take to …?"'], c: 0, r: 2 },
        { id: 'emb', k: 'lang', t: 'Token embedding', s: ['Llama-2 SPM · 32k'], c: 1, r: 2 },
        { id: 'state', k: 'in', t: 'Proprio', s: ['8-d · q01 / q99'], c: 0, r: 3 },
        { id: 'pp', k: 'state', t: 'Proprio projector', s: ['8 → 4096 → 4096', '1 token'], c: 1, r: 3 },
        { id: 'slots', k: 'in', t: 'Action slots', s: ['56 = 8 × 7', 'zero embeddings'], c: 1, r: 4 },
        { id: 'llm', k: 'core', t: 'Llama-2 7B', s: ['d 4096 · 32 heads × 128', 'SwiGLU 11008 · RoPE 1e4', 'causal mask removed:', 'fully bidirectional', 'lm_head not loaded'], c: 3, r: 0, h: 5, stack: true, badge: '×32' },
        { id: 'head', k: 'act', t: 'MLPResNet head', s: ['8 × 28672 → 4096', '2 res blocks → 7'], c: 4, r: 4 },
        { id: 'out', k: 'out', t: 'Action chunk', s: ['8 × 7', 'q01 / q99 un-norm'], c: 5, r: 4 },
      ],
      edges: [
        { from: 'cam', to: 'dino' }, { from: 'cam', to: 'sig', mx: 'g1' },
        { from: 'dino', to: 'proj', ta: 0.193 }, { from: 'sig', to: 'proj', ta: 0.807 },
        { from: 'proj', to: 'llm', ta: 0.176 },
        { from: 'txt', to: 'emb' }, { from: 'emb', to: 'llm', ta: R5[2], l: 'prompt' },
        { from: 'state', to: 'pp' }, { from: 'pp', to: 'llm', ta: R5[3], l: '1 tok' },
        { from: 'slots', to: 'llm', ta: R5[4], l: '56 slots' },
        { from: 'llm', to: 'head', fa: R5[4] }, { from: 'head', to: 'out' },
      ],
    },
    caption: '<b>One pass, no loop.</b> Every input becomes a row of the Llama-2 sequence: 256 fused patch tokens per view, one proprio token, the prompt and 56 zero slots. With the causal mask removed, the slots see the whole sequence, and their final hidden states feed the regression head.',
    figs: [{
      type: 'strip',
      segs: [
        { t: 'BOS', w: 46, k: 'lang' }, { t: 'patches', n: '256 · V', w: 240, k: 'vis' }, { t: 'proprio', n: '1', w: 66, k: 'state' },
        { t: 'prompt', n: 'L − 1', w: 150, k: 'lang' }, { t: 'action slots', n: '56 zero rows', w: 230, k: 'in' }, { t: 'stop', w: 48, k: 'lang' },
      ],
      marks: [
        { a: [0, 0], b: [5, 1], side: 't', k: 'data', l: 'one bidirectional pass over every row' },
        { a: [3, 0.94], b: [4, 0.982], k: 'attn', l: '56 final hidden states → MLPResNet' },
      ],
      caption: '<b>Where the head reads.</b> The 56 rows start one position early, at the last prompt token, and end at slot 54: the next-token shift of the original OpenVLA, kept by OFT and reproduced in vla.cpp.',
    }],
    specs: [
      ['Vision A', 'DINOv2 ViT-L/14 + 4 registers', '23 of 24', '1024', '16 / 16', '4096 GELU-erf', 'LayerScale; CLS and registers dropped after the blocks'],
      ['Vision B', 'SigLIP so400m/14', '26 of 27', '1152', '16 / 16', '4304 GELU-erf', 'no CLS, no post-LN'],
      ['Projector', 'fused GELU MLP', '3', '2176 → 8704 → 4096 → 4096', '–', 'GELU', 'channel concat per patch; 256 tokens per view'],
      ['Proprio', 'MLP', '2', '8 → 4096 → 4096', '–', 'GELU-erf', 'one LLM token after the patches'],
      ['LLM', 'Llama-2 7B', '32', '4096', '32 / 32 × 128', '11008 SwiGLU', 'RoPE θ 1e4, no mask, final RMSNorm'],
      ['Head', 'MLPResNet', '2 blocks', '28672 → 4096 → 7', '–', 'ReLU', 'LayerNorm on input and blocks; 8 rows of 7 × 4096 states'],
    ],
    notes: [
      ['In vla.cpp', [
        'Vision is one cached graph per view count; LLM and head are one fused graph cached on (sequence, views, prompt length).',
        'QKᵀ accumulates in F32 and attention is explicit softmax.',
        'RTX 3090, 1 view: 135 ms and 14.8 GB. LIBERO-Object 98.5 %.',
      ]],
      ['Inputs and outputs', [
        'Prompt: "In: What action should the robot take to {task}?\\nOut:" plus token 29871; the C++ appends the stop token.',
        'Action slots enter as zero vectors, not token embeddings.',
        'Output 8 × 7, un-normalised in C++ with q01 / q99 (suite picked by <code>VLA_OPENVLA_OFT_UNNORM_KEY</code>).',
      ]],
    ],
    src: { cpp: ['src/models/openvla_oft.cpp', 'src/modules/dual_tower.h'] },
  },

  // ---------------------------------------------------------------------------
  {
    id: 'vla-adapter', name: 'VLA-Adapter', alt: 'vla_adapter (Pro)', engines: ['cpp'], type: 'parallel',
    kicker: 'bridge attention over every LM layer',
    summary: "A 0.5B alternative to OFT that keeps the dual vision tower but reads <b>every LM layer</b>: a 24-block bridge-attention policy pairs block i with layer i, attending to its vision rows through a learned tanh gate and to 64 ActionQuery rows.",
    facts: [['Views', '2 × 224²'], ['LLM', 'Qwen2.5-0.5B · 24 L'], ['Head reads', 'all 24 layers'], ['Passes', '1'], ['Chunk', '8 × 7'], ['RTX 3090', '34.5 ms · 3.0 GB']],
    row: { vision: 'DINOv2-L + SigLIP so400m · 256 tok/view', backbone: 'Qwen2.5-0.5B · 24 L, causal', head: 'bridge attention · 24 blocks on all layers', decode: 'one pass · L1', chunk: '8 × 7' },
    dg: {
      nodes: [
        { id: 'cam', k: 'in', t: 'Cameras', s: ['2 views · 224 × 224', 'centre-crop 90 %'], c: 0, r: 0 },
        { id: 'dino', k: 'vis', t: 'DINOv2 ViT-L/14', s: ['d 1024 · 4 registers', '23 of 24 blocks'], c: 1, r: 0, stack: true, badge: '×23' },
        { id: 'sig', k: 'vis', t: 'SigLIP so400m/14', s: ['d 1152 · 26 of 27', '256 patches / view'], c: 1, r: 1, stack: true, badge: '×26' },
        { id: 'proj', k: 'op', t: 'Concat + MLP', s: ['2176-d per patch', '→ 896', '256 tok / view'], c: 2, r: 0, h: 2 },
        { id: 'txt', k: 'in', t: 'Instruction', s: ['Qwen chat template'], c: 0, r: 2 },
        { id: 'emb', k: 'lang', t: 'Token embedding', s: ['Qwen2.5 · 152k'], c: 1, r: 2 },
        { id: 'aq', k: 'op', t: 'ActionQuery tokens', s: ['64 learned × 896'], c: 1, r: 3 },
        { id: 'llm', k: 'core', t: 'Qwen2.5-0.5B', s: ['d 896 · 14 q / 2 kv', 'causal · RoPE θ 1e6', 'all 24 layers tapped'], c: 3, r: 0, h: 4, stack: true, badge: '×24' },
        { id: 'state', k: 'in', t: 'Proprio', s: ['8-d · q01 / q99'], c: 0, r: 4.6 },
        { id: 'pp', k: 'state', t: 'Proprio projector', s: ['8 → 896 → 896'], c: 1, r: 4.6 },
        { id: 'head', k: 'act', t: 'Bridge-attention policy', s: ['8 latents · d 896', 'block i ↔ LM layer i', 'tanh-gated task attn'], c: 3, r: 4.6, ph: 66, dy: -6, stack: true, badge: '×24' },
        { id: 'out', k: 'out', t: 'Action chunk', s: ['8 × 7', 'q01 / q99 un-norm'], c: 5, r: 4.6 },
      ],
      edges: [
        { from: 'cam', to: 'dino' }, { from: 'cam', to: 'sig', mx: 'g1' },
        { from: 'dino', to: 'proj', ta: 0.193 }, { from: 'sig', to: 'proj', ta: 0.807 },
        { from: 'proj', to: 'llm', ta: 0.224 },
        { from: 'txt', to: 'emb' }, { from: 'emb', to: 'llm', ta: R4[2], l: 'prompt' },
        { from: 'aq', to: 'llm', ta: R4[3], l: '64 queries' },
        { from: 'llm', to: 'head', k: 'attn', fs: 'b', ts: 't', l: 'layer i: vision rows +\nActionQuery rows → block i' },
        { from: 'state', to: 'pp' }, { from: 'pp', to: 'head', l: '65th adapter K, V' },
        { from: 'head', to: 'out', l: '8 × 7 · L1' },
      ],
    },
    caption: '<b>The head reads the LM layer by layer.</b> Unlike OFT, the LM keeps its causal mask and the head reads more than the last layer: bridge-attention block i takes keys and values from LM layer i. Proprio bypasses the LM and joins as an extra adapter key.',
    figs: [{
      type: 'strip',
      segs: [
        { t: 'first', n: 'BOS slot', w: 60, k: 'lang' }, { t: 'patches', n: '256 · V', w: 240, k: 'vis' },
        { t: 'prompt', n: 'L − 1', w: 170, k: 'lang' }, { t: 'ActionQuery', n: '64 learned', w: 210, k: 'op' }, { t: 'stop', w: 48, k: 'lang' },
      ],
      marks: [
        { a: [0, 0], b: [1, 0.98], k: 'attn', l: 'task K, V · rows [0, 256V)' },
        { a: [2, 0.93], b: [3, 0.985], k: 'attn', l: 'adapter K, V · 64 rows + proprio' },
      ],
      caption: '<b>Two slices of every LM layer feed block i.</b> Vision-position rows become task keys, whose logits a learned tanh gate scales; the ActionQuery rows, shifted one position early, plus the proprio token become adapter keys. Both join the 8 latents\' own keys in one softmax.',
    }],
    specs: [
      ['Vision', 'DINOv2-L (23) + SigLIP (26)', '23 + 26', '1024 + 1152', '16 + 16', '–', 'identical towers to OpenVLA-OFT'],
      ['Projector', 'GELU MLP', '3', '2176 → 896', '–', 'GELU', '256 tokens per view'],
      ['LLM', 'Qwen2.5-0.5B', '24', '896', '14 / 2 × 64', '4864 SwiGLU', 'QKV bias, RoPE θ 1e6, causal; every layer\'s output kept'],
      ['ActionQuery', 'learned embeddings', '–', '64 × 896', '–', '–', 'replace 64 placeholder ids'],
      ['Proprio', 'MLP', '2', '8 → 896 → 896', '–', 'GELU-erf', '65th adapter key / value, not in the LLM'],
      ['Policy', 'bridge-attention MLPResNet', '24 blocks', '896', '8 × 112', '896 ReLU', 'joint softmax over self (8) + adapter (65) + gated task (256V) keys; pairwise RoPE'],
      ['Output', 'LN + Linear', '1', '896 → 7', '–', '–', '8 latents → 8 × 7'],
    ],
    notes: [
      ['In vla.cpp', [
        'The LLM with all 24 layer taps and the 24-block head form one cached graph of 65,536 nodes.',
        'RoPE tables, the causal mask and the zero input are uploaded once.',
        'RTX 3090, 1 view: 34.5 ms and 3.0 GB. LIBERO-Object 95.5 %.',
      ]],
      ['Inputs and outputs', [
        'Qwen chat template without special tokens; the first prompt token takes the BOS slot in front of the patches.',
        'The head starts from zeros: 8 × 6272 → LN → Linear → ReLU gives the 8 latents.',
        'Output 8 × 7, q01 / q99 un-normalised in C++.',
      ]],
    ],
    src: { cpp: ['src/models/vla_adapter.cpp', 'src/modules/dual_tower.h', 'src/layers/rope.h'] },
  },

  // ---------------------------------------------------------------------------
  {
    id: 'bitvla', name: 'BitVLA', alt: 'bitvla', engines: ['cpp'], type: 'parallel',
    kicker: '1.58-bit weights, int8 activations',
    summary: "OFT's recipe on a <b>1.58-bit</b> stack: ternary weights and int8 activations in both the SigLIP tower and a BitNet b1.58 LLM. vla.cpp runs the packed 2-bit weights directly with its own tensor-core kernels.",
    facts: [['Views', '2 × 224²'], ['Weights', 'ternary · 2-bit packed'], ['Activations', 'int8 per token'], ['Passes', '1'], ['Chunk', '8 × 7'], ['RTX 3090', '27.1 ms · 1.46 GB']],
    row: { vision: 'BitSigLIP (ternary) · 256 tok/view', backbone: 'BitNet b1.58 · 30 L, bidirectional', head: 'MLPResNet (FP32) on 56 slots', decode: 'one pass · L1', chunk: '8 × 7' },
    dg: {
      nodes: [
        { id: 'cam', k: 'in', t: 'Cameras', s: ['2 views · 224 × 224', 'centre-crop 90 %'], c: 0, r: 0 },
        { id: 'vit', k: 'vis', t: 'BitSigLIP', s: ['d 1152 · 16 heads', 'ternary W · int8 A'], c: 1, r: 0, stack: true, badge: '×26' },
        { id: 'proj', k: 'op', t: 'Projector (FP)', s: ['1152 → 2560 → 2560'], c: 2, r: 0 },
        { id: 'txt', k: 'in', t: 'Chat template', s: ['image_pad × 512', 'proprio_pad + task'], c: 0, r: 1 },
        { id: 'emb', k: 'lang', t: 'Token embedding', s: ['128k vocab', 'rows read from disk'], c: 1, r: 1 },
        { id: 'state', k: 'in', t: 'Proprio', s: ['8-d · q01 / q99'], c: 0, r: 2 },
        { id: 'pp', k: 'state', t: 'Proprio projector', s: ['8 → 2560 → 2560'], c: 1, r: 2 },
        { id: 'slots', k: 'in', t: 'Action slots', s: ['56 zeros + stop'], c: 1, r: 3 },
        { id: 'llm', k: 'core', t: 'BitNet b1.58 LLM', s: ['d 2560 · 20 q / 5 kv', 'ReLU² GLU 6912', 'W 1.58-bit · A int8', 'bidirectional'], c: 3, r: 0, h: 4, stack: true, badge: '×30' },
        { id: 'head', k: 'act', t: 'MLPResNet (FP32)', s: ['8 × 17920 → 2560', '2 res blocks → 7'], c: 4, r: 3 },
        { id: 'out', k: 'out', t: 'Action chunk', s: ['8 × 7', 'q01 / q99 un-norm'], c: 5, r: 3 },
      ],
      edges: [
        { from: 'cam', to: 'vit' }, { from: 'vit', to: 'proj' }, { from: 'proj', to: 'llm', ta: R4[0] },
        { from: 'txt', to: 'emb' }, { from: 'emb', to: 'llm', ta: R4[1], l: 'prompt' },
        { from: 'state', to: 'pp' }, { from: 'pp', to: 'llm', ta: R4[2], l: 'proprio_pad' },
        { from: 'slots', to: 'llm', ta: R4[3], l: '56 slots' },
        { from: 'llm', to: 'head', fa: R4[3] }, { from: 'head', to: 'out' },
      ],
    },
    caption: "<b>OFT's wiring, 1.58-bit arithmetic.</b> Projected image tokens and the proprio token are scattered into placeholder positions of the chat template, and 56 zero slots follow. The LLM attends bidirectionally, and an FP32 MLPResNet regresses the chunk from the slot states.",
    figs: [{
      type: 'diagram',
      dg: {
        grid: { cw: 148, gx: 34, rh: 46, gy: 34, px: 16, py: 16 },
        nodes: [
          { id: 'x', k: 'op', t: 'Activations x', s: ['bf16 / f32 rows'], c: 0, r: 1 },
          { id: 'q', k: 'op', t: 'Per-token absmax', s: ['→ int8 rows'], c: 1, r: 1 },
          { id: 'w', k: 'core', t: 'Ternary weights', s: ['{-1, 0, 1} · 4 per byte', 'one f32 scale s'], c: 2, r: 0 },
          { id: 'g', k: 'act', t: 'int8 × int2 GEMM', s: ['WMMA tensor cores', 'int32 accumulate'], c: 2, r: 1 },
          { id: 'rs', k: 'op', t: 'Rescale', s: ['× s · max|x| / 127'], c: 3, r: 1 },
          { id: 'y', k: 'op', t: 'Output y', s: ['next op'], c: 4, r: 1 },
        ],
        edges: [
          { from: 'x', to: 'q' }, { from: 'q', to: 'g' }, { from: 'w', to: 'g', l: 'lop3 decode' }, { from: 'g', to: 'rs' }, { from: 'rs', to: 'y' },
        ],
      },
      caption: '<b>BitLinear on the CUDA path.</b> Weights stay 2-bit packed in device memory and are decoded to int8 inside the GEMM; activations are quantised per token just before it. Every q, k, v, o and FFN projection in the ViT and the LLM runs this way; the projector and head stay full precision.',
    }],
    specs: [
      ['Vision', 'BitSigLIP', '26', '1152', '16 / 16', '4304 GELU-tanh', 'every projection BitLinear (ternary W, int8 A); biases full precision'],
      ['Projector', 'MLP, full precision', '2', '1152 → 2560 → 2560', '–', 'GELU-erf', '256 tokens per view, scattered at <|image_pad|>'],
      ['Proprio', 'MLP', '2', '8 → 2560 → 2560', '–', 'GELU-erf', 'one token at <proprio_pad>'],
      ['LLM', 'BitNet b1.58', '30', '2560', '20 / 5 × 128', '6912 ReLU²-GLU', 'sub-norms before o_proj and down_proj; RoPE θ 5e5; no mask'],
      ['Head', 'MLPResNet (FP32)', '2 blocks', '17920 → 2560 → 7', '–', 'ReLU', '56 slot states → 8 × 7'],
    ],
    notes: [
      ['In vla.cpp', [
        'The int2-packed GGUF runs only on CUDA, where the ViT, LLM and head are all custom kernels; the ggml backend stays on the CPU.',
        '<code>ladder_int8xint2</code> decodes the 2-bit weights with <code>lop3</code> and feeds WMMA int8 tensor cores.',
        'gate and up are one GEMM followed by a fused ReLU²·mul kernel; attention uses cuBLAS bf16 batched GEMMs with F32 compute.',
        'RTX 3090, 1 view: 27.1 ms and 1.46 GB. LIBERO-Object 100 %, four-suite average 94.1 %.',
      ]],
      ['Quantisation and I/O', [
        'Weights: absmean ternary per tensor. Activations: absmax int8 per token.',
        'The CPU path runs ggml matmuls on float (ternary × scale) weights with an activation fake-quant op.',
        'Output 8 × 7, q01 / q99 un-normalised in C++.',
      ]],
    ],
    src: { cpp: ['src/models/bitvla.cpp', 'src/kernels/bitvla/bitnet_kernels.cu', 'src/kernels/bitvla/bitvla_lm_cuda.cu'] },
  },
);
