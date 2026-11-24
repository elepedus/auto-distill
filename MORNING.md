# Morning brief: 2026-11-24

Overnight session, 20:40–08:00 GMT. The detailed record is in [NOTEBOOK.md](NOTEBOOK.md) (Entries), and the main
receipt is [results/ane-vs-gpu-2026-11-24.md](results/ane-vs-gpu-2026-11-24.md).

## TL;DR

- **Ternary training on the ANE now matches the same model trained on the GPU with fp32 gradients, at equal
  tokens.**

  | budget | ANE, `--bound` | ANE, `--bound --grad-prescale` | GPU ternary TTF, fp32 |
  |---|---|---|---|
  | 2.05M tokens, 3 seeds | 2.218 ± 0.05 | 2.245 ± 0.04 | 2.286 ± 0.13 |
  | 8.2M tokens | | 1.794 (seed 1: pending) | 1.832 / 1.811 |

  Held-out KL against LFM2-350M, on a fixed eval set.
- **Yesterday's ANE numbers were wrong.** P=1 at 7.4–9.9 and PBT at 2.5–2.7 came from a bug, not from the
  limits of fp16 or ternary.
  - Layer 0's scale, gain and tap gradients (ds/dg/dk) overflowed, and the finite check never looked at them, so
    infs went into Adam.
  - The overflows the check did catch kept halving the loss scale, so the weight gradients underflowed.
  - On identical settings, the fixed trainer goes from 9.35 to 5.40 at 256k tokens.
- **Next, the forward overflowed: per-row scales grow 3–6×**, and the forward is a product chain of them.
  - `--bound` clamps the float parameters after each Adam step. Without it, 3 of 4 runs diverged; with it, 0 of 7
    did.
  - The bound is practical, not a worst-case proof: the worst-case bounds are below the init values.
- **Float fp32 leads ternary by ~0.2 nats at 8.2M tokens** (1.607 / 1.555 vs ~1.82). At 2.05M tokens there
  was no gap. This is the first sign that ternary capacity matters at d 256.
- **PBT-8 adds little at 2.05M tokens:** 2.172 vs 2.16–2.25 for P=1. Its value tonight was robustness: it
  replaces diverged members.

## What ran

| phase | result |
|---|---|
| 0, tooling | `--seed`, `--hyper`, `--loss-scale`/`--fixed-scale`, `--eval`, `--jsonl`, gradient diagnostics, `scripts/run-queue.sh`. Also found that P=1 was broken (a kernel address-space bug, fixed in 7fd7052). |
| 1, variance and loss scale | Cut short once the diagnostics showed the loss scale wasn't the real problem. A fixed scale of 2048 overflowed on 98% of updates. |
| new | `float-train`: an fp32 MLX baseline of the same student, same pipeline and same eval. `--ternary` runs the same TTF with fp32 autodiff gradients. |
| 2, numerics | Full finite check plus overflow attribution; `--grad-prescale` (per-tensor power-of-two factors in one runtime input); fixed eval set (seed 424242). |
| 2b, forward | `--bound` (parameter caps derived from the forward's product chain). |
| 3, comparisons | 3 seeds per arm at 2.05M tokens: GPU ternary, ANE bound, ANE bound+prescale. Prescale-only and neither diverged. PBT-8 with and without prescale. Long runs at 8.2M tokens: ANE, GPU ternary and GPU float. |

An ANE job and a GPU job ran concurrently most of the night, so **wall-clock times from tonight are not
throughput measurements**.

## Mistakes and caveats (plainly)

- **Phase 0 lost an hour.** zsh doesn't word-split `$VAR`, so the subagent's "64k-token" tests ran as full 28-min
  runs.
- **A wrong call, corrected 25 minutes later.** At 23:00 I declared the gap to be "fp16 numerics". The
  attribution run then showed it was the unchecked-overflow bug. Both entries are in the notebook.
- **Runs are not bit-reproducible.** The same seed and binary diverge after the first round, probably from GPU
  scatter-add atomics in the embedding gradient. Seed std is ~0.05–0.13 at 2.05M tokens.
- **The evaluation is small.** It is an 8-sequence (2048-token) held-out set. PBT selects on the same set, which
  gives it a slight optimistic bias.
- **Bounds are incomplete.** The bound comes from an alpha·D row-sum estimate (alpha 0.1), and attention's
  `e @ (V·2^8)` is still unbounded in the worst case. Overflow events still happen in backward outputs (28–93
  per 2.05M-token run), and each one skips its sequence.
- **Old receipts are superseded.** The receipts from yesterday (ane-ttf, ane-pbt8, ane-accum*) used the buggy
  trainer and a seed-dependent eval set. The README marks them as superseded.

## Decisions taken

- **D1: P=1 variance is not the priority.** The ANE path was broken numerically, so its variance said nothing.
  The machine time went to the float baseline and the numerics instead.
- **Ternary capacity vs precision.** The float gap was more than 1 nat, so I built `float-train --ternary` to
  separate the two. It showed precision wasn't the culprit; the bug was.
- **`--bound` is the recommended setting.** `--grad-prescale` stays optional: it cuts gradient zeros (0.4 → 0.05
  in the worst tensor) but buys no KL here.

## What's next (suggested)

1. **Make `--bound` the default.** Replace the alpha estimate with a derivation that holds, or with per-layer
   activation monitoring that tightens the caps. Bound attention's numerator.
2. **Capacity.** At 8.2M tokens float wins by about 0.2 nats. Try a wider ternary student (d 384 or 512) on the
   ANE. The population programs already handle width.
3. **Clean throughput measurement**, with one job at a time. Teacher text generation (~30–40 ms per sequence)
   and the GPU head and update dominate; the ANE trunk is a small fraction. Larger generation batches, or
   P>1 sharing one teacher pass, are the cheap wins.
4. **PBT at longer budgets.** The 8.2M-token runs used fixed default settings. PBT drifted to τ 3–4k, which is
   worth adopting as the default if it holds.
5. **Back to the main goal.** Use the now-sound ANE student pipeline for dataset-free transfer into larger
   ternary students, or return to the quantised-repair track (the 3-bit LFM2 repair, 56.6 → 49.8 perplexity).

## Receipts

- [results/ane-vs-gpu-2026-11-24.md](results/ane-vs-gpu-2026-11-24.md): all of tonight's runs, ablation, long
  runs, caveats. The JSONL files are in [results/runs-2026-11-24/](results/runs-2026-11-24/).
- [results/grad-numerics-2026-11-23.md](results/grad-numerics-2026-11-23.md): the overflow root cause and the
  prescale design.
- [results/forward-bounds-2026-11-24.md](results/forward-bounds-2026-11-24.md): what grows, the bound
  derivation, the 512k checks.
