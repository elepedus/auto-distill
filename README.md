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

## `float-train`: the GPU reference

```sh
swift run -c release float-train [--ternary | --lr 3e-3 [--schedule]] [--tokens 2048000] [--round 256000] \
  [--hyper LR,THETA,TAU,RHO] [--seed 0] [--eval 8] [--jsonl PATH]
```

The same student (d 256, 4 layers, tied embedding, 20.2M parameters) and the same pipeline as `ane-train`, with
autodiff gradients in fp32 on the GPU. It exists to say what the ANE path should reach.
- **fp32 (default):** float weights, Adam at `--lr` (default 1e-3); `--schedule` adds warmup and a cosine decay to 0.1x (always on with `--ternary`).
- **`--ternary`:** the ternary linears are trained with the same TTF kernel and schedules as `ane-train` (`--hyper`
  default `2e-3,16,1000,1`); only the gradients differ (fp32 on the GPU, not fp16 on the ANE).

At 2.05M tokens, 3 seeds, held-out KL: ternary 2.286 ± 0.13; float at lr 3e-3, one seed: 2.45 (2.39 with the schedule).
Ternary TTF at 8.2M tokens: 1.832. See [`results/ane-vs-gpu-2026-11-24.md`](results/ane-vs-gpu-2026-11-24.md).

## `scripts/run-queue.sh`

```sh
scripts/run-queue.sh QUEUEFILE OUTDIR [BINARY=ane-train]
```

Runs the lines of QUEUEFILE (`name args...`) one after another from a copy of the release binary in OUTDIR, writing
`name.log` and `name.jsonl` (via `--jsonl`) for each. A run whose log already has its throughput line is skipped, so the
queue can be restarted. Blank and `#` lines are ignored. Build first: `swift build -c release --build-path .build/out`.

## `ane-train`: training a ternary model on the ANE

```sh
swift run -c release ane-train [--population 8] [--tokens 2048000] [--round 256000] [--accum 1] [--frozen-trits] \
  [--seed 0] [--hyper LR,THETA,TAU,RHO] [--loss-scale 1024] [--fixed-scale] [--eval 8] [--jsonl PATH] \
  [--grad-prescale] [--bound] [--bound-alpha 0.1]
```

- `--seed N`: initialisation, training data and PBT draws follow it. The eval set does not (fixed, generated under seed 424242).
- `--hyper LR,THETA,TAU,RHO`: Adam rate, TTF θ, τ and ρ multiplier for every member (default of `float-train`:
  `2e-3,16,1000,1`). Without it, member 0 has the hand-picked settings and the others random draws (PBT).
- `--loss-scale S`: starting fp16 loss scale (default 1024). It adapts from max |dx₀| unless `--fixed-scale`.
- `--eval N`: number of held-out teacher sequences (default 8).
- `--jsonl PATH`: one JSON object per line: `eval`, `train`, `overflow`, `params`, `done`. Non-finite numbers are null.
- `--grad-prescale` (optional): per-tensor power-of-two scales inside the ANE backward, so token-sum outputs (`ds_*`, `dk*`, `dg_*`)
  and per-element ones (`gp_*`) each fit fp16. Every output is now checked for non-finite values, and an overflowing
  sequence is skipped. See [`results/grad-numerics-2026-11-23.md`](results/grad-numerics-2026-11-23.md).
- `--bound`, `--bound-alpha A` (recommended; the stabiliser that matters, see the receipt): after each Adam step, clamp the per-row scales, gains and conv taps so the fp16 forward
  stays below about 16k. A is the practical row-sum factor (0.1; strict worst case 1, which is infeasible). The bound is
  practical, not a proof; attention's `e @ (V·2^8)` is unbounded in the worst case. See
  [`results/forward-bounds-2026-11-24.md`](results/forward-bounds-2026-11-24.md).

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

**Result.** Held-out KL on 8 fixed teacher sequences after 2.05M tokens, defaults, P = 1, 3 seeds. The same ternary
model trained on the GPU with fp32 gradients (`float-train --ternary`) is the reference:

| path | s0 | s1 | s2 | mean ± std |
|---|---|---|---|---|
| ANE ternary, `--bound` | 2.160 | 2.243 | 2.250 | **2.218 ± 0.05** (0 of 3 diverged) |
| ANE ternary, `--grad-prescale --bound` | 2.203 | 2.278 | 2.253 | 2.245 ± 0.04 (0 of 3 diverged) |
| GPU ternary TTF, fp32 gradients | 2.232 | 2.439 | 2.187 | 2.286 ± 0.13 |
| ANE ternary, `--grad-prescale` only | 2.150 | diverged ~1.0M | diverged ~1.0M | |
| ANE ternary, neither | diverged ~0.3M | | | |

The forward bound is needed (without it 3 of 4 runs diverged, from a nonfinite forward as the row scales grow); the
backward still overflows 60–93 times per run, each detected and skipped. Full curves, flags, caveats (single machine,
PBT selects on the eval set, attention unbounded) and the JSONL of every run:
[`results/ane-vs-gpu-2026-11-24.md`](results/ane-vs-gpu-2026-11-24.md). Prescale cuts gradient zeros but buys no KL, so it is
optional.

**Long runs (8.2M tokens), held-out KL:** ANE ternary (`--grad-prescale --bound`) seed 0 1.794; GPU ternary seed 0 1.832,
seed 1 1.811; GPU float fp32 seed 0 1.607. The ANE matches GPU ternary (one ANE seed, so a tie, not a win); float leads
ternary by about 0.2 nats, the first sign of ternary capacity mattering at d = 256. ANE seed 1 and float seed 1: pending.
Bound-only at 2.05M: 2.218 ± 0.05 (3 seeds).

**Earlier numbers were wrong.** The first P = 1 results (held-out KL 7.4–9.9 at 2.05M tokens, and the "trunk learns
slowly" curves below) came from a bug, not from ternary training or fp16 as such: the backward's token-sum outputs
(`ds_*`, `dk*`, `dg_*`, mostly layer 0) overflowed fp16 and went unchecked into Adam, and each overflow that was caught
halved the loss scale until the per-element gradients underflowed. P = 1 was also broken at the time by a TTF kernel
address-space bug (7fd7052). The measurements below are kept as history and should not be read as the current state.

**(Superseded, old overflow handling.) The trunk learns, but the update is too noisy.** Each update sees 256 tokens; TTF's paper used 32k. Each
update is normalised to about ±ρ per entry, so pure noise drives the accumulator to a typical size of
ρ·√(τ/2) ≈ 22ρ. At ρ near 1 that crosses θ = 16, so trits flip on noise. That matches the curve:
- **Steps 250–750 (ρ ≈ 1):** flips peak and held-out KL gets worse than the frozen control.
- **Steps 750–1,250 (ρ 0.85 → 0.6):** learning happens.
- **Later (ρ < 0.5):** flips almost stop and KL plateaus.

**Speed:** about 90 ms per step, or 2,900 tokens/s overall; 4,600 tokens/s excluding text generation. The
ANE accounts for 8.8 ms of each step (forward 2.9, backward 5.9); most of the rest is the 350M teacher.

**(Superseded, old overflow handling.) Accumulating more tokens per update hurt, at equal tokens.** Same seed, data and budget (2.05M tokens), only
tokens per update varied (held-out KL):

| tokens seen | 256 per update (8,000 updates) | 2k per update (1,000) | 8k per update (250) |
|---|---|---|---|
| 1.09M | 7.84 | 9.67 | 9.73 |
| 1.47M | 7.69 | 7.80 | 9.62 |
| 2.05M | 7.43 | 7.67 | 9.56 |

Receipts: [`results/ane-accum1-2026-11-23.txt`](results/ane-accum1-2026-11-23.txt),
[`results/ane-accum8-2026-11-23.txt`](results/ane-accum8-2026-11-23.txt),
[`results/ane-accum32-2026-11-23.txt`](results/ane-accum32-2026-11-23.txt).

TTF's step and Adam's step are both normalised, so each update moves the parameters by about the same amount
whatever its batch. Fewer, larger updates therefore move the model less per token: the 8k run flipped almost no
trits until halfway through. Better signal-to-noise does stop the noise flips (no early bump), but θ, τ, ρ and
the learning rate are all calibrated in updates. They have to be retuned for each tokens-per-update setting, not
carried over. Accumulation still raises throughput, 5,300 tokens/s against 4,100 excluding text generation,
because the per-update work is shared.

## `population-probe`: many models in one ANE program

```sh
swift run -c release population-probe
```

`LFM2Shape(population: P)` runs P independent models side by side in one ANE program. Every tensor gets a
population axis: the channel axis, which matmuls batch over. Each member has its own weights. Attention heads of
all members share that axis.

On an M2 Max, tiny ternary layers (d 256, T 256) compile at P up to 16. Every member's outputs are
**bit-identical** to running that member alone, forward and backward. The cost per member stops falling at
about P = 4, where the ANE is saturated:

| layer | solo, fwd / bwd | per member at P ≥ 4, fwd / bwd | speedup per member |
|---|---|---|---|
| conv | 0.80 / 1.50 ms | 0.26 / 0.70 ms | 3.1× / 2.2× |
| attention | 0.57 / 1.30 ms | 0.40 / 1.00 ms | 1.4× / 1.3× |

Receipt: [`results/population-probe-2026-11-23.txt`](results/population-probe-2026-11-23.txt).

**(Superseded, old overflow handling.) Population-based training, 8 members, 2.05M tokens.** Members start from the hand-picked settings (member 0)
and from random draws over Adam rate 3e-4–2e-2, θ 2–64, τ 50–4,000 and a ρ multiplier of 0.1–2. Every 256k tokens
the worse half copies the better half and perturbs its hyperparameters by ×0.8 or ×1.25:

| tokens | best held-out KL | its hyperparameters (Adam rate, θ, τ, ρ×) |
|---|---|---|
| 256k | 7.74 | 1.6e-3, 20, 1,250, 0.80 |
| 512k | 5.92 | same |
| 1.02M | 3.86 | 1.3e-3, 16, 1,562, 0.64 |
| 1.79M | 2.61 | 2.0e-3, 10.2, 2,441, 1.00 |
| 2.05M | 2.67 | 1.0e-3, 12.8, 1,953, 0.80 |

Receipt: [`results/ane-pbt8-2026-11-23.txt`](results/ane-pbt8-2026-11-23.txt). The two members drawn with Adam
rates above 9e-3 diverged by update ~200 and were replaced. Throughput was 11,650 member-tokens/s excluding text
generation.

With the fixed binary, PBT-8 with `--grad-prescale` reaches 2.243 at 2.05M tokens (4 diverged members replaced), no better than a
P = 1 bound+prescale run. The old comparison with the single runs (7.43) needs care:
- **Not reproducible.** Member 0 has the single run's exact settings, data and initialisation, yet reached 8.00
  at 256k tokens against about 9.9 alone. Members are bit-identical on the ANE; what differed is shared loss
  scaling and skipped sequences. With noise-driven flips at ρ ≈ 1, the trajectory seems to be chaotic, so
  single-run comparisons need replicates.
- **Unstable late.** Training KL swings by ±0.5 between logs late in the run, and the lineage that led at 1.79M
  tokens fell to 7.8 by 2.05M.
- **Loss scale.** The scale wandered between 8 and 2,048 across runs. fp16 gradient range is a candidate cause of
  both effects.
