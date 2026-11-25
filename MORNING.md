# Morning brief: 2026-11-25

Night 2 (22:00–08:00 GMT). The goal was the best possible recovery of LFM2-350M as a ternary, device-aligned
model, with training and inference co-optimised for this Mac. The detailed record is in [NOTEBOOK.md](NOTEBOOK.md)
(the "Night 2 plan" section and its Entries). The receipt is
[results/ternarise-night2-2026-11-25.md](results/ternarise-night2-2026-11-25.md). The previous brief is in
[results/morning-2026-11-24.md](results/morning-2026-11-24.md).

## Result

LFM2-350M with **every decoder linear ternary** (trits × per-row scale), recovered by distillation against its own
bf16 base with no dataset. Training used TTF, with no master weights.

| model | wikitext-2 perplexity | KL vs base (wikitext) | weights | GPU decode tok/s |
|---|---|---|---|---|
| bf16 base | 32.4 | 0 | 709 MB | 179 (noisy, 134–237) |
| MLX 4-bit of the base | 38.0 | 0.20 | 200 MB | 533 |
| ternary, untrained (TWN) | 32,355 | ~6.5 | 224 MB | 500 |
| ternary, yesterday's recovery (2.05M tokens) | 456 | | 224 MB | 500 |
| **ternary, tonight (3.07M layerwise + 16.5M e2e tokens)** | **123.5** | 1.80 | **224 MB** | **500** |

- **Recovered: 32,355 → 123.5 perplexity. Not recovered: 123.5 against a base of 32.4.** It's a big step from
  yesterday, but still a long way from the base, and well behind 4-bit (38.0).
- **Deployable and exact.** `ternary-export` writes a standard MLX 2-bit model with zero dequantization
  mismatches. Its perplexity through the normal load path is 123.57, matching training.
- **Fast on this Mac's GPU.** Decode is 2.8× faster than bf16, at a third of the size. 4-bit is just as fast,
  because the bf16 embedding dominates the ternary file.
- **The same model runs on the ANE:**
  - parity KL 0.00002 against the GPU, no overflow (max activation 5.6);
  - prefill 4.1k tok/s;
  - decode only ~33 tok/s, because each of 16 single-layer programs pays a dispatch cost.
- **On this M2 Max the GPU is the faster inference device.** The ANE's value here is concurrency and efficiency.

## What we learned tonight

1. **Layerwise reconstruction is the big lever.** First train each ternary layer to reproduce the base layer's
   output from the base's input, then train end to end.

   | layerwise tokens (then 512k end to end) | KL | perplexity |
   |---|---|---|
   | 0 | 1.898 | 1095 |
   | 256k | 1.138 | 305 |
   | 1M | 0.903 | 208 |
   | 3.07M (tonight's handover, before any end-to-end) | 1.099 | 329 |

   It needs one layer of memory and one shared teacher pass, which suits big models and the ANE.
2. **The two phases want different float rates.** Layerwise works best at 3e-4; end to end at 1e-4 early and
   3e-5 on the long plateau. Added `--layerwise-lr`.
3. **After a good init, trit flipping must be gentle.**
   - At peak ρ×0.5 (~3k flips per update), held-out KL rose 0.87 → 0.94 on the plateau.
   - At ρ×0.1–0.2 the trits froze, and only the floats learnt; that plateaued at ~0.74.
   - At ρ×0.35 (~65 flips per update) it beat frozen trits by ~0.02–0.03 and kept improving.
   - The flip rate should be a controlled target, not a side effect of ρ and θ.
4. **Don't cool early.** A cosine schedule over 8.2M tokens was at 1.012 / perplexity 265 when stopped at 2.05M.
   A cosine run over 2.05M ended at 1.248 / 456. WSD with resumable checkpoints made it possible to change course
   mid-run.
5. **Two-sided (row × column) scales: no gain.** Dropped. Kept as an option: tied per shared input, they fold into
   norm gains at zero inference cost.
6. **The main remaining gap is data coverage, not the ternary format.** On held-out teacher text the KL is ~0.65;
   on wikitext it's 1.75–1.87. The teacher's own samples, from a random first token, cover natural text poorly.

## What ran (and what went wrong)

| GMT | what |
|---|---|
| 22:00–23:45 | Four subagents. A1 added checkpoints, WSD, int8 accumulators and profiling. A2 added colscale and layerwise. A3 built ANE inference for the whole 350M model. B4 built the exact 2-bit GPU export and benchmark. |
| 23:15–00:10 | Screens: layerwise size, float lr. |
| 00:10–01:45 | Final run, part 1: layerwise 3.07M, then end to end. Held-out KL rose on the plateau, so I stopped it. |
| 01:50–03:45 | Resumed probes from checkpoints (A, B, C, D) to fix ρ and lr. D won. |
| 03:45–06:25 | D alone to 16.5M tokens, with WSD decay. |
| 06:25–07:30 | Export, benchmarks on an idle GPU, ANE check, write-up. |

- **The final model was assembled from a chain of resumed runs**, not one planned schedule.
- **Everything is single-seed, with an 8-sequence held-out eval.** Wikitext perplexity is the honest headline.
- **zsh word-splitting bit me again** while launching the probes; relaunched with literal arguments. ~5 min lost.
- **The ternary inference speed-up is only on the GPU.** On the ANE the trits are still fp16, so there's no
  bandwidth saving; Core ML-style palettisation is untested.

## Artifacts (local, git-ignored)

- `artifacts/lfm2-350m-ternary-ckpt/`: resumable checkpoint (trits, scales, floats, optimiser state).
- `artifacts/lfm2-350m-ternary-mlx2bit/`: the exported MLX 2-bit model. It loads with mlx-swift-lm like any
  quantized model.

## Suggested next steps

1. **Data coverage, the biggest lever left.** Have the teacher write from diverse prefixes: more varied starting
   points, its own prompted continuations, or a lower temperature for part of each pool. Measure wikitext KL, not
   just teacher-text KL.
2. **Control the flip rate.** Make the flip rate per update the target and adapt ρ to hit it, together with
   layerwise.
3. **Longer layerwise, possibly interleaved.** Return to layerwise MSE periodically, or use it as an auxiliary loss
   during end-to-end training.
4. **Compress the embedding and head.** They are now most of the bytes, and a naive 4-bit embedding cost KL 0.47.
   Train with it, or use a factorised or shared head.
5. **Make the ANE path pay:**
   - fuse layers into fewer programs for decode;
   - run the ANE student or writer concurrently with the GPU teacher;
   - try palettised 2-bit weights.
6. **Toward the north star:** int8 accumulators (validated on the small model), packed trits, then a 1–2B model.
