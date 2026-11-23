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

**Backward pass.** A second program per layer recomputes the forward pass from the layer input and runs a
hand-written backward pass. It takes the gradient at the layer output and returns:
- `dx`, the gradient for the layer below;
- for each ternary linear, its two gradient factors: the input activations and the output gradient times the
  row scale. Their product is the gradient for the trits, so the full weight-gradient matrix never exists on the
  ANE. This is the form TTF-style training consumes;
- the gradients of the float parameters: row scales, norm gains and conv taps.

Every ternary matmul in it is exact up to the final rounding, and token and feature sums go through
ones-matmuls (wide accumulator). Against MLX fp32 autodiff on the same bytes, all gradients point the same way
(cosine ≥ 0.9989, no non-finite values, 10–11.5 ms per 256 tokens per layer). Typical errors are 0.1–0.6%. The
worst are 3–4.5%, in layers 0 and 2, where results come out of heavy cancellation: softmax's backward on the
q/k path, and norm-gain gradients summed over 256 random-sign token terms. The ANE rounds every elementwise op
to fp16; MLX's fp16 autodiff, whose fused kernels keep fp32 inside, stays 5–15× closer.

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

## `ane-train`: training a ternary model on the ANE

```sh
swift run -c release ane-train [--steps 2000] [--accum 1] [--frozen-trits]
```

Trains a tiny ternary LFM2-shaped student from scratch: d = 256, 4 layers (conv, conv, attention, conv),
3.4M trits. The teacher is bf16 LFM2-350M, which writes the training text and scores it. Each step:
- embedding lookup on the GPU;
- 4 layers forward on the ANE;
- head and KL loss on the GPU;
- 4 layers backward on the ANE;
- a TTF update for the trits and Adam for the float parameters.

Nothing passes through the CPU:
- **Owned state:** trits and float parameters live in page-aligned buffers we own. Both ANE programs of a
  layer bind the same buffer, and MLX wraps it.
- **GPU writes:** Metal kernels write updates straight into those buffers.
- **Between programs:** activations and gradients chain directly from one program's output IOSurface to the
  next program's input.
- **Self-test:** the trainer checks both directions of zero-copy at startup.

First result, 2,000 steps of one 256-token sequence each, KL on 8 held-out teacher sequences:

| step | TTF | trits frozen (only float parts learn) |
|---|---|---|
| 0 | 9.70 | 9.70 |
| 500 | 9.83 | 9.76 |
| 1,000 | 8.91 | 9.59 |
| 2,000 | 8.17 | 9.48 |

Receipts: [`results/ane-ttf-2026-11-23.txt`](results/ane-ttf-2026-11-23.txt),
[`results/ane-frozen-2026-11-23.txt`](results/ane-frozen-2026-11-23.txt).

**The trunk learns, but the update is too noisy.** Each update sees 256 tokens; TTF's paper used 32k. Each
update is normalised to about ±ρ per entry, so pure noise drives the accumulator to a typical size of
ρ·√(τ/2) ≈ 22ρ. At ρ near 1 that crosses θ = 16, so trits flip on noise. That matches the curve:
- **Steps 250–750 (ρ ≈ 1):** flips peak and held-out KL gets worse than the frozen control.
- **Steps 750–1,250 (ρ 0.85 → 0.6):** learning happens.
- **Later (ρ < 0.5):** flips almost stop and KL plateaus.

**Speed:** about 90 ms per step, or 2,900 tokens/s overall; 4,600 tokens/s excluding text generation. The
ANE accounts for 8.8 ms of each step (forward 2.9, backward 5.9); most of the rest is the 350M teacher.
