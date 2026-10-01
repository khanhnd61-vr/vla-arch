// Encoder-decoders with learned action queries: ACT, IMPACT, TurboVLA.
// Sources: vla.simd src/models/{act,impact,turbovla}/*, src/nn/t5_encoder.cpp;
// vla.cpp src/models/turbovla.cpp. ACT / IMPACT values from the SO-101 benchmark GGUFs.

window.VLA_ARCH.models.push(
  // ---------------------------------------------------------------------------
  {
    id: 'act', name: 'ACT', alt: 'act', engines: ['simd'], type: 'query',
    kicker: 'CVAE decoder at z = 0',
    summary: "Action Chunking with Transformers, as served on the SO-101 arm. ResNet-18 features from two cameras, the joint state and a zeroed CVAE latent form a 602-token sequence; a DETR-style decoder turns <b>50 learned queries</b> into a 50-step chunk in <b>one pass</b>.",
    facts: [['Cameras', '2 × 480 × 640'], ['Encoder seq', '602 tokens'], ['Layers', '6 enc · 4 dec'], ['Passes', '1'], ['Chunk', '50 × 6'], ['i5-12400F', '139 / 59 ms (FP32 / INT8)']],
    row: { vision: 'ResNet-18 · 300 tok/camera', backbone: 'post-norm encoder · 6 L', head: 'decoder 4 L · 50 queries', decode: 'one pass', chunk: '50 × 6' },
    dg: {
      nodes: [
        { id: 'cam', k: 'in', t: 'Cameras', s: ['front + wrist', '480 × 640 · ImageNet'], c: 0, r: 0 },
        { id: 'rn', k: 'vis', t: 'ResNet-18 trunk', s: ['shared · BN folded', '→ 15 × 20 × 512'], c: 1, r: 0 },
        { id: 'tok', k: 'op', t: 'Camera tokens', s: ['1×1 conv 512 → 512', '2 × 300 + 2-D sine pos'], c: 2, r: 0 },
        { id: 'state', k: 'in', t: 'Robot state', s: ['6-d · mean / std'], c: 0, r: 1 },
        { id: 'sp', k: 'state', t: 'state_proj', s: ['6 → 512 · 1 token'], c: 1, r: 1 },
        { id: 'cv', k: 'skip', t: 'CVAE encoder', s: ['training only', 'not exported'], c: 0, r: 2 },
        { id: 'lat', k: 'op', t: 'Latent token', s: ['z = 0 → bias only', '1 token'], c: 1, r: 2 },
        { id: 'enc', k: 'core', t: 'Transformer encoder', s: ['d 512 · 8 heads', 'FFN 3200 ReLU', 'post-norm', '602 tokens'], c: 3, r: 0, h: 3, stack: true, badge: '×6' },
        { id: 'q', k: 'in', t: 'Decoder queries', s: ['50 zeros', '+ learned position'], c: 2, r: 3.7 },
        { id: 'dec', k: 'act', t: 'Transformer decoder', s: ['self + cross-attn', 'post-norm · d 512'], c: 3, r: 3.7, stack: true, badge: '×4' },
        { id: 'hd', k: 'op', t: 'LayerNorm + Linear', s: ['512 → 6'], c: 4, r: 3.7 },
        { id: 'out', k: 'out', t: 'Action chunk', s: ['50 × 6', 'mean / std un-norm'], c: 5, r: 3.7 },
      ],
      edges: [
        { from: 'cam', to: 'rn' }, { from: 'rn', to: 'tok' }, { from: 'tok', to: 'enc', ta: 0.119 },
        { from: 'state', to: 'sp' }, { from: 'sp', to: 'enc', l: '1 tok' },
        { from: 'cv', to: 'lat', k: 'skip' }, { from: 'lat', to: 'enc', ta: 0.881, l: '1 tok' },
        { from: 'enc', to: 'dec', k: 'attn', fs: 'b', ts: 't', l: 'cross-attn to\n602-token memory' },
        { from: 'q', to: 'dec' }, { from: 'dec', to: 'hd' }, { from: 'hd', to: 'out' },
      ],
    },
    caption: '<b>A single deterministic pass.</b> At inference the CVAE encoder is gone and the latent is zero, so ACT is a DETR-style encoder-decoder: 602 tokens in, 50 queries out. Position encodings are added to queries and keys at every layer, never to the token stream.',
    specs: [
      ['Backbone', 'ResNet-18 (to layer4)', '8 blocks', '64 → 512', '–', 'ReLU', 'BN folded into convs; 480 × 640 → 15 × 20 per camera'],
      ['Camera tokens', '1×1 conv', '1', '512 → 512', '–', '–', '+ 2-D sine position on Q and K only, every layer'],
      ['State', 'Linear', '1', '6 → 512', '–', '–', 'learned 1-D position'],
      ['Latent', 'constant', '–', '512', '–', '–', 'z = 0, so the projection collapses to its bias'],
      ['Encoder', 'post-norm transformer', '6', '512', '8 × 64', '3200 ReLU', '602 tokens; no final encoder norm'],
      ['Decoder', 'post-norm transformer', '4', '512', '8 × 64', '3200 ReLU', '50 zero queries + learned position; self + cross-attn'],
      ['Head', 'LN + Linear', '1', '512 → 6', '–', '–', ''],
    ],
    notes: [
      ['In vla.simd', [
        'Weights load into float arenas with exact-size checks; GEMMs pack once into 16-wide panels for a 6 × 16 AVX2 micro-kernel (NEON tiles on ARM, Accelerate on Apple).',
        'Residual adds fuse into the GEMM epilogue; ReLU fuses into the max-pool.',
        '<code>ACT_INT8</code> picks W8A8 per block (VNNI / sdot); the state projection and head stay FP32.',
        'i5-12400F: 139.4 ms FP32, 59.2 ms INT8. Raspberry Pi 5: 1.25 s FP32.',
      ]],
      ['Inputs and outputs', [
        'Two cameras at 480 × 640 with ImageNet mean / std; 6-d state, mean / std.',
        'Code defaults are LeRobot\'s (4 enc / 1 dec / chunk 100); the served checkpoint is 6 / 4 / 50 to match IMPACT.',
        'No temporal ensembling; the server truncates to <code>actions_per_chunk</code>.',
      ]],
    ],
    src: { simd: ['src/models/act/resnet_backbone.cpp', 'act_transformer.cpp', 'act_model.cpp'] },
  },

  // ---------------------------------------------------------------------------
  {
    id: 'impact', name: 'IMPACT', alt: 'instruction-modulated ACT', engines: ['simd'], type: 'query',
    kicker: 'language as tokens and FiLM',
    summary: "vla.simd's own policy: <b>ACT plus language</b>. A frozen T5-small encodes the instruction once, and it enters the network twice: as <b>FiLM on every ResNet stage</b> and as tokens in the encoder sequence.",
    facts: [['Cameras', '2 × 480 × 640'], ['Text', '≤ 32 T5 tokens'], ['Language enters', 'tokens + FiLM'], ['Passes', '1'], ['Chunk', '50 × 6'], ['i5-12400F', '140 / 61 ms (FP32 / INT8)']],
    row: { vision: 'ResNet-18 + FiLM · 300 tok/camera', backbone: 'T5-small + post-norm encoder 6 L', head: 'decoder 4 L · 50 queries', decode: 'one pass', chunk: '50 × 6' },
    dg: {
      nodes: [
        { id: 'txt', k: 'in', t: 'Instruction', s: ['T5 SPM · ≤ 32 tokens'], c: 0, r: 0 },
        { id: 't5', k: 'lang', t: 'T5-small encoder', s: ['d 512 · frozen'], c: 1, r: 0, stack: true, badge: '×6' },
        { id: 'tproj', k: 'lang', t: 'Text projection', s: ['512 → 512', '≤ 32 text tokens'], c: 2, r: 0 },
        { id: 'film', k: 'lang', t: 'FiLM head', s: ['masked mean-pool', '→ γ, β (960 each)'], c: 1, r: 1 },
        { id: 'cam', k: 'in', t: 'Cameras', s: ['front + wrist', '480 × 640 · ImageNet'], c: 0, r: 2 },
        { id: 'rn', k: 'vis', t: 'ResNet-18 + FiLM', s: ['(1 + γ)·x + β after', 'each of 4 stages'], c: 1, r: 2 },
        { id: 'tok', k: 'op', t: 'Camera tokens', s: ['2 × 300 + 2-D sine pos'], c: 2, r: 2 },
        { id: 'state', k: 'in', t: 'Robot state', s: ['6-d · mean / std'], c: 0, r: 3 },
        { id: 'sp', k: 'state', t: 'state_proj', s: ['6 → 512 · 1 token'], c: 1, r: 3 },
        { id: 'enc', k: 'core', t: 'Transformer encoder', s: ['[latent | state |', 'cam0 | cam1 | text]', 'd 512 · 8 heads', 'post-norm · FFN 3200'], c: 3, r: 0, h: 4, stack: true, badge: '×6' },
        { id: 'q', k: 'in', t: 'Decoder queries', s: ['50 zeros + position'], c: 2, r: 4.6 },
        { id: 'dec', k: 'act', t: 'Transformer decoder', s: ['self + cross-attn', 'post-norm · d 512'], c: 3, r: 4.6, stack: true, badge: '×4' },
        { id: 'hd', k: 'op', t: 'LayerNorm + Linear', s: ['512 → 6'], c: 4, r: 4.6 },
        { id: 'out', k: 'out', t: 'Action chunk', s: ['50 × 6', 'mean / std un-norm'], c: 5, r: 4.6 },
      ],
      edges: [
        { from: 'txt', to: 't5' }, { from: 't5', to: 'tproj' }, { from: 'tproj', to: 'enc', ta: 0.087 },
        { from: 't5', to: 'film', fs: 'b', ts: 't' },
        { from: 'film', to: 'rn', k: 'cond', fs: 'b', ts: 't' },
        { from: 'cam', to: 'rn' }, { from: 'rn', to: 'tok' }, { from: 'tok', to: 'enc', ta: 0.638 },
        { from: 'state', to: 'sp' }, { from: 'sp', to: 'enc', ta: 0.913, l: '1 tok' },
        { from: 'enc', to: 'dec', k: 'attn', fs: 'b', ts: 't', l: 'cross-attn to memory' },
        { from: 'q', to: 'dec' }, { from: 'dec', to: 'hd' }, { from: 'hd', to: 'out' },
      ],
      groups: [
        { k: 'plain', label: 'language path · cached per instruction', of: ['txt', 't5', 'tproj', 'film'] },
      ],
    },
    caption: '<b>Language enters twice.</b> The cached instruction embedding yields per-stage FiLM parameters for the shared ResNet and up to 32 text tokens for the encoder. Everything after is ACT: a post-norm encoder, a 4-layer decoder and 50 queries. The constant latent token (z = 0) is part of the encoder sequence but not drawn.',
    specs: [
      ['Text', 'T5-small encoder', '6', '512', '8 × 64', '2048 ReLU', 'frozen, pre-RMSNorm, shared relative bias; cached per instruction'],
      ['Text tokens', 'Linear', '1', '512 → 512', '–', '–', '≤ 32 tokens + learned position on Q, K; padded rows dropped'],
      ['FiLM head', 'Linear', '1', '512 → 1920', '–', '–', 'masked mean-pool → γ, β for 64 + 128 + 256 + 512 channels'],
      ['Backbone', 'ResNet-18 + FiLM', '8 blocks', '64 → 512', '–', 'ReLU', '(1 + γ)·x + β after each stage; same γ, β for both cameras'],
      ['Encoder', 'post-norm transformer', '6', '512', '8 × 64', '3200 ReLU', '602 + n_text tokens'],
      ['Decoder', 'post-norm transformer', '4', '512', '8 × 64', '3200 ReLU', '50 queries'],
      ['Head', 'LN + Linear', '1', '512 → 6', '–', '–', ''],
    ],
    notes: [
      ['In vla.simd', [
        'The whole language path (T5, projection, FiLM) is cached per instruction string; a query costs the ResNet and transformer plus at most 32 extra tokens.',
        'Padded text rows are dropped rather than masked, which keeps attention on the dense path.',
        'A QAT-trained int8 checkpoint runs W8A8: 60.5 ms against 140.3 ms FP32 on an i5-12400F; 1.32 s FP32 on a Raspberry Pi 5.',
      ]],
      ['Design', [
        'FiLM is the input that matters: next to about 600 visual tokens, 32 text tokens are easy for a policy to ignore, and FiLM puts language inside perception.',
        'The FiLM head is zero-initialised, so training starts as plain ACT.',
        'Changes from ACT: chunk 100 → 50, encoder 4 → 6, decoder 1 → 4; about 78M trainable parameters (ACT: 34M).',
      ]],
    ],
    src: { simd: ['src/models/impact/t5_text.cpp', 'impact_model.cpp', 'src/models/act/resnet_backbone.cpp', 'act_transformer.cpp'] },
  },

  // ---------------------------------------------------------------------------
  {
    id: 'turbovla', name: 'TurboVLA', alt: 'turbovla', engines: ['cpp', 'simd'], type: 'query',
    kicker: 'bi-attention fusion, ACT decoder',
    summary: "A small, fast policy: DINOv3 sees, BERT reads, six <b>Grounding-DINO-style bi-attention layers</b> fuse the two, and an ACT decoder with <b>12 learned queries</b> regresses the chunk in a single pass.",
    facts: [['Views', '2 × 256²'], ['Memory', '535 tokens'], ['Queries', '12'], ['Passes', '1'], ['Chunk', '12 × 7'], ['Params', '215.5M · FP32']],
    row: { vision: 'DINOv3 ViT-B/16 · 256 tok/view', backbone: 'BERT-base + 6 bi-attention fusion layers', head: 'ACT decoder 3 L · 12 queries', decode: 'one pass · tanh', chunk: '12 × 7' },
    dg: {
      nodes: [
        { id: 'cam', k: 'in', t: 'Cameras', s: ['2 views · 256 × 256', 'ImageNet norm'], c: 0, r: 0 },
        { id: 'vit', k: 'vis', t: 'DINOv3 ViT-B/16', s: ['d 768 · 2-D RoPE θ 100', 'CLS + 4 reg + 256'], c: 1, r: 0, stack: true, badge: '×12' },
        { id: 'vp', k: 'op', t: 'Vision projection', s: ['MLP + skip → 256', '+ view embedding'], c: 2, r: 0 },
        { id: 'txt', k: 'in', t: 'Instruction', s: ['BERT WordPiece', 'pad to 11 / 14 / 21'], c: 0, r: 1.2 },
        { id: 'bert', k: 'lang', t: 'BERT-base', s: ['d 768 · post-LN', 'sub-sentence mask'], c: 1, r: 1.2, stack: true, badge: '×12' },
        { id: 'tp', k: 'lang', t: 'Text projection', s: ['768 → 256 · 21 rows'], c: 2, r: 1.2 },
        { id: 'fus', k: 'core', t: 'Bi-attention fusion', s: ['vision ↔ text, one', 'score matrix both ways', '+ text self-attn layer'], c: 3, r: 0, h: 2.2, stack: true, badge: '×6' },
        { id: 'state', k: 'in', t: 'Robot state', s: ['8-d · mean / std'], c: 0, r: 2.2 },
        { id: 'st', k: 'state', t: 'State tokens', s: ['MLP 8 → 512', '→ 2 tok × 256'], c: 1, r: 2.2 },
        { id: 'mem', k: 'op', t: 'Decoder memory', s: ['512 vision', '+ 21 text', '+ 2 state', '= 535 × 256'], c: 4, r: 0, h: 3.2 },
        { id: 'q', k: 'in', t: 'Learned queries', s: ['12 × 256'], c: 2, r: 3.9 },
        { id: 'dec', k: 'act', t: 'ACT decoder', s: ['pre-norm · 8 heads × 32', 'self + cross · FFN 2048'], c: 3, r: 3.9, stack: true, badge: '×3' },
        { id: 'mlp', k: 'op', t: 'MLP + tanh', s: ['256 → 512 → 512 → 7'], c: 4, r: 3.9 },
        { id: 'out', k: 'out', t: 'Action chunk', s: ['12 × 7 in [-1, 1]', 'min / max un-norm'], c: 5, r: 3.9 },
      ],
      edges: [
        { from: 'cam', to: 'vit' }, { from: 'vit', to: 'vp' }, { from: 'vp', to: 'fus', ta: 0.172 },
        { from: 'txt', to: 'bert' }, { from: 'bert', to: 'tp' }, { from: 'tp', to: 'fus', ta: 0.828 },
        { from: 'fus', to: 'mem', ta: 0.323 },
        { from: 'state', to: 'st' }, { from: 'st', to: 'mem', ta: 0.889, l: '2 tok' },
        { from: 'mem', to: 'dec', k: 'attn', fs: 'b', ts: 't', l: 'cross-attn · 535 tokens' },
        { from: 'q', to: 'dec' }, { from: 'dec', to: 'mlp' }, { from: 'mlp', to: 'out' },
      ],
      groups: [
        { k: 'plain', label: 'text · re-run only when the instruction changes', of: ['txt', 'bert', 'tp'] },
      ],
    },
    caption: '<b>One pass from pixels to a 12-step chunk.</b> After DINOv3, vision tokens never self-attend: fusion layers exchange information with text through one shared score matrix, and only the text side gets a self-attention layer. The decoder\'s 12 queries then cross-attend to all 535 memory tokens, padded text rows included.',
    specs: [
      ['Vision', 'DINOv3 ViT-B/16', '12', '768', '12 / 12 × 64', '3072 GELU-erf', 'CLS + 4 registers, axial 2-D RoPE θ 100 on patches; LayerScale folded; views batched'],
      ['Vision projection', 'MLP + skip', '2', '768 → 1024 → 256', '–', 'GELU-erf', '+ skip Linear 768 → 256, LN, + learned view embedding'],
      ['Text', 'BERT-base (uncased)', '12', '768', '12 / 12 × 64', '3072 GELU-erf', 'post-LN, sub-sentence mask, positions restart per span'],
      ['Fusion', 'bi-attention + text encoder', '6', '256', '4 × 256 / 4 × 64', '1024 ReLU', 'one score matrix used vision → text and text → vision; gated residual on normed tokens'],
      ['State', 'MLP', '2', '8 → 256 → 512', '–', 'GELU-erf', 'reshaped to 2 tokens + learned position, LN'],
      ['Decoder', 'ACT, norm-first', '3', '256', '8 × 32', '2048 ReLU', '12 learned queries; unmasked self + cross-attn; no final norm'],
      ['Head', 'MLP + tanh', '3', '256 → 512 → 512 → 7', '–', 'ReLU', 'chunk in [-1, 1]'],
    ],
    notes: [
      ['In vla.cpp', [
        'Two cached graphs: BERT re-runs only when the token ids change, and the main graph reads its output tensor as a view.',
        'The cross-attention in_proj is applied twice rather than sliced, so ggml-openvino works with BF16 weights.',
        'RTX 3090: 9.3 ms (5.8 ms with f16 weights + flash attention). LIBERO-Object 100 %.',
        'Returns tanh-space actions; the client un-normalises.',
      ]],
      ['In vla.simd', [
        'Runs the same vla.cpp-format GGUF (215.5M parameters, FP32), re-laid out at load.',
        'Both views go through DINOv3\'s GEMMs as one batch, with per-view attention.',
        'A single-entry cache keeps the last instruction\'s text encoding.',
        'Un-normalises server-side (min / max, gripper sign). 132.7 ms FP32 on an i7-14700F.',
      ]],
    ],
    src: { cpp: ['src/models/turbovla.cpp'], simd: ['src/models/turbovla/dinov3_vision.cpp', 'bert_text.cpp', 'fusion.cpp', 'action_head.cpp'] },
  },
);
