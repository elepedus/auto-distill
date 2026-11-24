# Ternary LFM2-350M as an exact MLX 2-bit model: GPU inference (2026-11-24)

Tool: `ternary-export` (Sources/ternary-export/main.swift). Test checkpoints come from `ane-infer write-ckpt DIR [--colscale [--tied]]`
(TWN trits of the base model; `--colscale` draws random input-channel scales; `--tied` makes them foldable). No trained checkpoint
was available when this ran, so parity and speed are measured on the TWN-initialised model (quality is not the point here; fidelity and speed are).

## Method

`ternary-export export CKPT OUT [--embed-bits 4]` writes `OUT/config.json` (base config plus `"quantization": {"group_size": 64, "bits": 2}`;
with `--embed-bits 4` also `"model.embed_tokens": {"group_size": 64, "bits": 4}`) and `OUT/model.safetensors`. mlx-swift-lm loads it with its
standard path (`loadWeights` quantizes exactly the modules that have `.scales`, so embedding and tied head stay bf16 unless asked).
Layout per linear: `weight` uint32 [N, K/16] (2 bits, 16 per word, little-endian), `scales` and `biases` bf16 [N, K/64].
Affine 2-bit: w = scale*q + bias with q = t+1, scale = s_row, bias = -s_row in every group. Norms, conv taps, embedding: bf16.

Colscale folding (documented in the source header), all exact:

| colscale on | folded into | condition |
|---|---|---|
| q/k/v_proj (shared input) | `operator_norm.weight` | the three identical, else error |
| conv in_proj | `operator_norm.weight` | none |
| w1/w3 (shared input) | `ffn_norm.weight` | identical, else error |
| conv out_proj c[d] | scale of in_proj row D+d (the C chunk; chunks are B, C, x, y = C * conv(B*x), conv is depthwise) | none |
| attention out_proj c[h*64+d] | scale of v_proj row g*64+d, g = h/2 (attention is linear in V) | the two query heads of a kv head equal, else error |
| w2 | scale of w3 row j (h = silu(w1 x) * w3 x) | none |

Untied colscales fail with a clear message (checked: `ck-untied` aborts at layer 0 w1/w3).

## Exactness

For all 92 linears of the model, MLX's own `dequantized(weight, scales, biases, groupSize 64, bits 2)` is compared element by element with
`trits * bf16(folded row scale)`: **0 mismatching elements** (with and without colscale folding). The only storage loss is rounding the
fp32 row scale to bf16: max relative deviation 0.00389 (< 2^-8 = 0.0039). Packing is also cross-checked against MLX `quantized()`
(identical words on a probe matrix). Folded norms are also rounded to bf16.

## Parity (4 wikitext sequences x 256 tokens, 1024 positions)

Reference: the checkpoint evaluated as ternarise-style TernaryLinear in fp32, floats from the checkpoint. Candidate: exported model through the
standard load, bf16 activations. The noise floor is the same TernaryLinear with bf16 matmuls against fp32.

| checkpoint | rel-L2 | argmax agreement | KL(ref‖exported) nats/token | noise floor (bf16 TernaryLinear vs fp32): rel-L2 / agree / KL |
|---|---|---|---|---|
| write-ckpt (no colscale) | 0.0252 | 96.19% | 0.0040 | 0.0176 / 97.46% / 0.0018 |
| write-ckpt --colscale --tied (all folds exercised) | 0.0354 | 94.24% | 0.0054 | 0.0316 / 95.21% / 0.0043 |

So the exported model is at bf16 noise (one extra bf16 rounding of the scales and folded norms). Exported vs the bf16 TernaryLinear: rel-L2 0.028 / 0.045.
`--embed-bits 4`: KL(ref‖exported) 0.474 nats/token, argmax agreement 54%: the 4-bit embedding / tied head costs ~0.47 nats on this model
(for scale, a standard whole-model MLX 4-bit of the base is KL 0.21 against bf16 base). Not worth it for LFM2-350M: keep the embedding bf16.

## Benchmark (GPU, mlx-swift-lm, M2 Max)

Prefill: forward of T=256 tokens, B=1 and B=8, 5 timed forwards per run. Decode: greedy, batch 1, 16-token prompt, 256 new tokens, KV and conv caches,
argmax each step. Median of runs. "weights" is the safetensors size. Reference 4-bit: `ternary-export quant4` (MLX `quantize(model:)`, group 64, all Linear and the embedding).

**A long ternarise training run (3 processes) shares the GPU throughout, and its load varies, so absolute numbers are pessimistic and noisy.**
Three sets were taken; the GPU was never idle. Set A (22:32 UTC, 5 reps) caught a quiet moment (bf16 prefill 10.4k, close to the 10.8k measured idle);
sets B (22:34, 5 reps) and C (22:29, 3 reps) were under heavy contention.

| model | weights | prefill B=1 | prefill B=8 | decode B=1 |
|---|---|---|---|---|
| **Set A (quiet)** | | | | |
| bf16 base | 709 MB | 10389 | 13031 | 262 |
| ternary 2-bit (bf16 embedding) | 224 MB | 4594 | 10074 | 441 |
| ternary 2-bit + 4-bit embedding | 128 MB | 8438 | 9574 | 436 |
| MLX 4-bit base | 200 MB | 8461 | 9603 | 397 |
| **Set B (loaded)** | | | | |
| bf16 base | 709 MB | 4627 | 8849 | 43 |
| ternary 2-bit | 224 MB | 2807 / 4034 | 6329 / 6021 | 49 / 62 |
| ternary 2-bit + 4-bit embedding | 128 MB | 6524 | 6759 | 61 |
| MLX 4-bit base | 200 MB | 3560 | 8266 | 346 |
| **Set C (loaded, 3 reps)** | | | | |
| bf16 base | 709 MB | 5206 | 9397 | 61 |
| ternary 2-bit | 224 MB | 4492 | 7051 | 73 |
| ternary 2-bit + 4-bit embedding | 128 MB | 3079 | 4952 | 259 (runs 46-367) |
| MLX 4-bit base | 200 MB | 4024 | 5803 | 44 |

(Tokens/s. Two ternary rows in set B are two repeated runs.) Per-run values are printed by `ternary-export bench`.

Reading, quiet set A only: decode of the 2-bit model is 1.7x bf16 (441 vs 262 tok/s) and slightly above 4-bit (397); batch-8 prefill is 77% of bf16;
batch-1 prefill at T=256 is the weak spot: 4.6k tok/s, 44% of bf16 and about half of the 4-bit model's 8.5k, i.e. the 2-bit quantized matmul kernel at M=256
is slower than the 4-bit one here (the decode figure shows no such penalty). Both are still above the ANE (prefill 3.9k, decode 30). The set A prefill-B1 run
varied 4.0-5.9k even in set A, so treat the gap as indicative.

## Caveats

* GPU shared with training; sets differ by up to 5x (decode of bf16: 262 vs 43). Only the ordering within a quiet set means anything. A later quiet set was not possible: the training run was still going when this finished.
* Checkpoint is TWN-initialised, not trained; fidelity numbers are about the export, not model quality.
* The `kl` subcommand (`ternary-export kl MODEL REF`) reuses the parity printer; its labels say "exported ... vs fp32 ternary reference" but compare two standard-loaded models.
* Scales are bf16, so the exported model is not bit-identical to an fp32-scale evaluation; the difference is below bf16 activation noise (table above).
* Decode loop is eval-per-token (no async pipelining), so decode figures are slightly below mlx-swift-lm `generate`.

## Reproduce

```
ane-infer write-ckpt /tmp/te/ck-tied --colscale --tied     # then copy config.json (dereference the HF symlink) if absent
ternary-export export /tmp/te/ck-tied /tmp/te/x-tied [--embed-bits 4]
ternary-export parity /tmp/te/ck-tied /tmp/te/x-tied
ternary-export quant4 /tmp/te/q4
ternary-export bench /tmp/te/x-tied --reps 5
```
