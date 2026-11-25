# Ternary LFM2-350M, night 2: layerwise reconstruction and a long TTF run (2026-11-24 to 25)

Follows [`ternarise-2026-11-24.md`](ternarise-2026-11-24.md), where 2.05M tokens of TTF left the ternary model at KL 1.25 and
wikitext-2 perplexity 456 (base 32.37). Goal tonight: the best recovery of LFM2-350M with every decoder linear ternary, trained
and run on this Mac (M2 Max). Same harness as yesterday: held-out KL on 8 teacher-generated sequences of 256 tokens (seed 424242),
wikitext-2 test perplexity (64 x 512 tokens).

**Model.** Every decoder linear (conv in/out_proj, attention q/k/v/o, FFN w1/w2/w3; 92 linears) is trits x per-row scale, TWN init.
Embedding and tied head are bf16, norms fp32/bf16, conv taps float. No column scales.

## Result

| model | wikitext-2 ppl | KL vs base (wikitext) | weights |
|---|---|---|---|
| base bf16 | 32.37 | 0 | 709 MB |
| MLX 4-bit of the base | 38.03 | 0.20 | 200 MB |
| 3-bit, repaired (2026-11-23) | 49.82 | | |
| **ternary (2-bit export), this run** | **123.57** | 1.80 | 224 MB |
| ternary after yesterday's 2.05M run | 456 | | |
| TWN start | 32,355 | | |

The ternary model is 3.8x the base perplexity and 3.2x the 4-bit model's. It is not a drop-in replacement; it is a usable ternary model
where yesterday there was none, and a 3.7x perplexity improvement on yesterday's best (456 to 123.6).

## Final recipe

- Init: TWN, residual accumulator init, fp32 accumulators, TTF theta 32, tau 1000.
- **Layerwise reconstruction phase** (`--layerwise 3072000 --layerwise-lr 3e-4`): each student layer, fed the teacher's input,
  is trained to reproduce the teacher's layer output. 3.07M tokens, float Adam lr 3e-4, rho x0.5.
- **End-to-end KL** to 16.5M tokens, with the schedule below, WSD decay over the last 15%.

The run is a chain of resumes, assembled from probes rather than planned as one schedule:

```
layerwise 3.07M (lr 3e-4)
  -> e2e run 1: float lr 1e-4, rho x0.5            0 - 2.05M   (KL ~0.90)
  -> probe B:   float lr 3e-5, rho x0.1            2.05M - 4.10M
  -> probe D:   float lr 3e-5, rho x0.35           4.10M - 6.14M   (resumed from B's 4.10M checkpoint)
  -> final-D:   same, WSD decay from 14.0M         6.14M - 16.5M   (resumed from D's 6.14M, --tokens 16.5M)
```

Total 3.07M layerwise + 16.5M e2e tokens, about 5.5 h of training wall time. Throughput ~1.07k tok/s alone (~1.7k tok/s in total
with two concurrent runs).

## Curve (held-out KL on 8 teacher sequences / wikitext-2 perplexity)

| point | KL | ppl |
|---|---|---|
| TWN start | 6.547 | 32,355 |
| layerwise handover (3.07M layerwise tokens) | 1.099 | 329 |
| e2e 2.05M | ~0.90 | 210 |
| B, 4.10M | 0.768 | 161 |
| D, 6.14M | 0.716 | 140 |
| 8.19M | 0.709 | 141 |
| 12.3M | 0.683 | 132 |
| **final, 16.5M** | **0.652** | **123.5** |

Held-out KL every 512k tokens, final-D: 6.66M 0.728, 7.17M 0.714, 7.68M 0.711, 8.19M 0.709, 8.70M 0.700, 9.22M 0.692,
9.73M 0.696, 10.2M 0.691, 10.8M 0.686, 11.3M 0.684, 11.8M 0.686, 12.3M 0.683, 12.8M 0.678, 13.3M 0.680, 13.8M 0.669, 14.3M 0.670,
14.8M 0.669, 15.4M 0.663, 15.9M 0.654, 16.4M 0.652. Most of the last 0.03 comes from the WSD decay (from 14.0M). 8 sequences
give a KL noise of a few hundredths, so single steps in this table are not significant; the trend is.

## Screens

**Layerwise reconstruction is the big lever.** 512k tokens end to end, A2 trials (`runs-2026-11-24/night2-a2-*.jsonl`):

| trial | start KL | KL @512k | ppl @512k |
|---|---|---|---|
| baseline (no layerwise) | 6.547 | 1.898 | 1095 |
| layerwise 256k first | 3.005 | **1.138** | **305** |
| colscale (tied) | 6.086 | 1.931 | 1223 |
| colscale, untied | 5.539 | 2.030 | 1435 |
| colscale + layerwise 256k | 2.740 | 1.177 | 338 |

With 1M layerwise tokens at lr 1e-5 (`night2-screen-layerwise1m.jsonl`): handover 1.426 / ppl 576, then e2e 128k 1.015, 256k 0.930,
512k **0.903 / ppl 208**. Mean per-layer relative MSE fell 0.084 (96k) to 0.051 (1M), still falling. The final run's 3.07M-token
phase (lr 3e-4) handed over at 1.099 / 329 with mean relative MSE ~0.04. A layerwise token costs about the same as an e2e token
(256k took 330-340 s under contention).

**Colscale gave no gain** (rows above: 1.931 vs 1.898; with layerwise 1.177 vs 1.138), so the architecture stays plain trits x row scale.

**Float lr** (layerwise 256k then e2e 256k, the same lr in both; `night2-screen-lr*.jsonl`):

| lr | handover KL | e2e 128k | final KL | final ppl |
|---|---|---|---|---|
| 3e-5 | 2.627 | 1.247 | 1.183 | 438 |
| 1e-4 | 2.291 | 1.146 | **1.070** | 357 |
| 3e-4 | **2.027** | 1.456 | 1.282 | 505 |
| 1e-3 | 2.129 | 2.342 | 2.000 | 1968 |

The layerwise phase prefers 3e-4 and e2e 1e-4, hence `--layerwise-lr`. The 1e-5 layerwise screen above was also run, in the 1M test.

**Cosine baseline** (`night2-baseline-cosine8m-stopped.jsonl`): a cosine schedule over 8.2M tokens, stopped at 2.05M, reached KL 1.012 /
ppl 265 (1.41 at 1.02M); the 2.05M-schedule run of yesterday reached 1.248 / 456 at the same token count. Cooling early costs real
quality, so the final run uses a long plateau and WSD. (This baseline had no layerwise phase, so it is a schedule comparison only.)

## The plateau and the rho probes

The first final run (`night2-final-phase1-layerwise-and-e2e-to-3m.jsonl`) used e2e float lr 1e-4 and rho x0.5 for the whole plateau.
Held-out KL rose: 0.872 at 1.02M, 0.901 at 2.05M, 0.938 at 3.07M. Train KL tracked the flip rate: lowest (0.58) while rho warmed
up with few flips, ~0.70 once flips reached ~3,000 per update at peak rho. After a good layerwise init, peak-rho flip noise is
destructive; yesterday's cosine screens never sat at peak long enough to show it. The run was stopped and resumed from the 2.05M
checkpoint (KL 0.901, ppl 210) as probes:

| probe | settings | from | KL along the way (every 256k) | flips |
|---|---|---|---|---|
| A | rho x0.1, float lr 1e-4 | 2.05M | 0.901, 0.857, 0.838, 0.864, 0.854 (3.07M) | ~0 |
| B | rho x0.1, float lr 3e-5 | 2.05M | 0.901, 0.851, 0.824, 0.816, 0.796 (3.07M), 0.768 (4.10M), 0.742 (5.4M), 0.736 (6.14M), then ~0.736-0.740 (6.9M) | ~0 |
| C | rho x0.2, float lr 3e-5 | B, 3.07M | 0.796, 0.783, 0.782, 0.772, 0.767 (4.10M) | ~0 |
| D | rho x0.35, float lr 3e-5 | B, 4.10M | 0.768, 0.761, 0.750, 0.738, 0.731, 0.729, 0.722, 0.727, 0.716 (6.14M), 0.710 (6.40M) | ~65-70 per update |

Findings:
- At rho x0.1-0.2 the trits froze (about 1,000 flips in total, almost all in the first updates); the floats do the work.
  Float lr 3e-5 beat 1e-4 on the plateau (B vs A: 0.796 vs 0.854 at 3.07M).
- B and C tie (0.767-0.768 at 4.10M), both frozen. B then flattened at ~0.74.
- **rho x0.35 (D) beat frozen trits by 0.02-0.027 at equal tokens** (0.710 at 6.40M vs B's 0.737 at 6.40M). Modest trit movement (~65 flips per
  update) helps; heavy flipping (~3k per update at x0.5) hurts. The `flips_total` field in the JSONLs is cumulative per run and
  restarts at each resume.

## Deployable export (`ternary-export export`)

Exact MLX 2-bit affine (q = t+1, scale = s, bias = -s, group 64); embedding and head stay bf16.
- **0 dequantization mismatches** over the 92 linears (MLX's own dequantized() equals trits x bf16(scale)).
- Parity against the fp32 ternary reference: KL **0.00024**/token.
- Wikitext-2 perplexity **123.57** through the standard mlx-swift-lm load path (123.5 in the fp32 reference). KL vs base on wikitext 1.80.

**Idle-GPU benchmark** (prefill T=256, B1 / B8 tok/s; greedy decode B1 tok/s; weights):

| model | prefill B1 / B8 | decode B1 | weights |
|---|---|---|---|
| bf16 | 10964 / 13296 | 179 (runs 237 / 179 / 134, noisy) | 709 MB |
| ternary 2-bit | 9193 / 10452 | **500** | 224 MB (mostly the bf16 embedding) |
| MLX 4-bit | 8806 / 9917 | 533 | 200 MB |

Decode of the ternary model is 2.8x the bf16 run's median (2.1x its best), and 94% of 4-bit's speed at 12% more weight bytes. Prefill is
slower than bf16 (84% at B1; unpacking 2-bit weights costs compute). The bf16 decode number varied across repeats, so
the ratio is soft.

## ANE (`ane-infer --ckpt`, this checkpoint)

- Parity against the GPU ternary model: KL 0.00002/token, argmax agreement 99.6%.
- 0 overflow; max |y| 5.6 over all layers.
- Prefill 4121 tok/s end to end (embedding and head on the GPU). Decode ~33 tok/s: 16 single-token programs per token, dispatch-bound.
- The ANE stays slower than the GPU here (see [`ane-infer-2026-11-24.md`](ane-infer-2026-11-24.md)); its role is concurrency and efficiency.

## Caveats

- **Single seed** throughout; the screens and probes are one run each.
- **The held-out eval is 8 sequences** of teacher text, with KL noise of a few hundredths. Perplexity (wikitext, 32k tokens) is the independent check.
- **The chain was assembled from probes, not planned as one schedule:** rho and lr changed at resume points (2.05M, 4.10M, 6.14M) as
  evidence came in. A single planned schedule (layerwise, then rho x0.35 and lr 3e-5 from 2M) might do better or worse; it was not run.
- **Contention during screens:** other jobs shared the GPU, so screen times and throughput are unreliable (the 8.2M baseline ran at ~413 tok/s).
- **Distribution gap.** Wikitext KL vs base (1.75-1.87) is far above held-out teacher-text KL (~0.65-0.72). Training text is teacher
  samples from a random first token at T=1, which covers natural text poorly; this is the main remaining gap, and the held-out KL
  flatters the model.
- **bf16 decode is noisy** (134-237 tok/s across repeats), so decode speedups against bf16 are soft; the 2-bit vs 4-bit comparison is tighter.
- The chain's final settings (rho x0.35, lr 3e-5) were not swept further.

## What next

- Better data coverage: prompted or diverse generation (varied first tokens, prefixes from real text, temperature mixes), aimed at the wikitext gap.
- Compress the embedding/head (4-bit costs KL 0.47 on its own; it is most of the 224 MB).
- Fused ANE decode (fewer, larger programs per token).
- int8 accumulators (`--acc int8` works; untested at this length) and packed trits, for training-state memory.

## Files

Run JSONLs: [`runs-2026-11-25/`](runs-2026-11-25/) (`night2-final-phase1-layerwise-and-e2e-to-3m`, `night2-probeA/B/C/D`,
`night2-final-D`, `night2-screen-lr3e-5/1e-4/3e-4/1e-3`, `night2-screen-layerwise1m`, `night2-baseline-cosine8m-stopped`) and
[`runs-2026-11-24/night2-a2-*.jsonl`](runs-2026-11-24/) (baseline, layerwise, colscale, colscale-untied, both).
Checkpoint and export (git-ignored, local): `artifacts/lfm2-350m-ternary-ckpt`, `artifacts/lfm2-350m-ternary-mlx2bit`.
