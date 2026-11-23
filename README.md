# auto-distill

Copy capabilities from one model to another without a dataset. The student writes its own text, the
teacher scores every token, and the student learns the teacher's full next-token distribution (KL).
Each batch is generated, used once and thrown away, so nothing is stored.

## `distill` (Swift, MLX)

```sh
swift run -c release distill train TEACHER STUDENT OUT [--steps 300] [--batch 8] [--seq 256] [--lr 1e-5]
swift run -c release distill eval TEXTFILE TEACHER MODEL...
```

Models are local MLX directories or Hugging Face ids already in the local cache.

- **Quantised student:** only the quantisation scales and biases are trained, so the output is an ordinary
  MLX quantised model, with no adapter and no extra inference cost.
- **Full-precision student:** every weight is trained. Teacher and student must share a tokenizer.

First result, LiquidAI/LFM2-350M at 3 bits, 300 steps (about 600k self-generated tokens, about 13 minutes
on an M2 Max). Perplexity is on wikitext-2 test, which training never sees:

| model | perplexity | KL vs teacher |
|---|---|---|
| bf16 teacher | 32.37 | 0 |
| 3-bit | 56.57 | 0.866 |
| 3-bit, repaired | 49.82 | 0.802 |

An earlier Python prototype of the same loop reached 48.10 in one run; the gap is within run-to-run noise
(the per-step training KL tracks closely). Building needs Xcode's Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`).

A learning rate of 1e-4 destroys 3-bit scales within three steps; 1e-5 works.

## `ternary-probe`: ternary on the Apple Neural Engine

`ternary-probe` builds one ANE program that multiplies fp16 activations by ternary weights ({-1, 0, 1}, fed
at runtime), in the forward (`x·W`) and backward (`g·Wᵀ`) directions. It compares every output bit for bit
with the exact sum rounded once to fp16.

On an M2 Max (macOS 27.2.1), every finite output matches the exact sum rounded once **with ties away
from zero** (not IEEE ties-to-even). That holds at K = 1024 and 4608, for 8-bit activation codes and for
general fp16 activations, so a simulator can reproduce these matmuls bit for bit. With 8-bit codes, a
power-of-two prescale chosen from 127·K makes overflow impossible; without it, the worst case is all `inf`.
Receipt: [`results/ternary-probe-2026-11-23.txt`](results/ternary-probe-2026-11-23.txt).

```sh
swift run -c release ternary-probe
```

Building needs a local checkout of swift-ane (see `Package.swift`), which is not published yet.

## `lfm2-ane`: a ternary LFM2 layer on the Apple Neural Engine

```sh
swift run -c release lfm2-ane TEXTFILE [LAYER...]
```

Runs one LFM2-350M decoder layer with ternary weights (per-row absmean scale, BitNet style) on the ANE and
compares it with an MLX fp32 reference. The inputs are the real hidden states of bf16 LFM2-350M on the text.
Weights, scales and constants are runtime inputs uploaded once; after that, MLX reads the ANE's own input
and output IOSurfaces in place, so both engines compute from the same bytes.

On an M2 Max, conv and attention layers at both ends of the network match the reference to 0.05–0.07% (relative
L2) at the layer output, with no non-finite values, in 4.5–6 ms per 256 tokens per layer. Receipt:
[`results/lfm2-ane-layers-2026-11-23.txt`](results/lfm2-ane-layers-2026-11-23.txt).

How the layer is built for the ANE:
- **RMSNorm** divides by each row's max |x| before squaring and sums the squares with a matmul (wide
  accumulator), with ε rescaled to match. No finite input can overflow it.
- **Attention** uses exp(s − max) and two matmuls for numerator and denominator, with V scaled by 2⁸ against
  underflow, following swift-ane's attention study. GQA runs as 16 full heads.
- **The conv block** uses three matmuls instead of splitting one along the last axis (M2's width-offset slice
  cliff), and shifts rows for the causal conv with 0/1 matrices, which is exact.
- **SiLU** is the explicit `a / (1 + exp(-a))`; swift-ane measured the native op as inaccurate.
- **Constants are inputs.** Several `BLOBFILE` constants in one in-memory MIL program fail to load
  (`verifyBundleAtPath: invalid model`) on macOS 27.2.1; a single one works.
