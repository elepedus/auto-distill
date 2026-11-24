# Lab notebook

Running record of experiments, decisions and plans. Newest entries at the bottom of each section. Receipts live in
`results/`; this file says what they mean and what we decided.

## State at 2026-11-23 20:40 GMT

**Built and verified** (see README for numbers and receipts):
- `distill`: dataset-free distillation in Swift/MLX. 3-bit LFM2-350M repair: perplexity 56.57 → 49.82.
- `ternary-probe`: ternary × fp16 matmuls on the ANE are exact up to one final rounding, which goes
  **ties away from zero**. A power-of-two prescale rules out overflow.
- `lfm2-ane`: ternary LFM2 layers on the ANE. Forward within 0.05–0.07%; hand-written backward, cosine ≥ 0.9989
  against MLX autodiff.
- `population-probe`: P models in one ANE program, each member bit-identical to running alone. Per member,
  3.1×/2.2× faster for conv (fwd/bwd) and 1.4×/1.3× for attention.
- `ane-train`: tiny ternary LFM2 student (d 256, 4 layers, 3.4M trits), trunk forward and backward on the ANE,
  teacher bf16 LFM2-350M on the GPU. Zero-copy throughout. TTF for the trits, Adam for the floats. Population
  mode with population-based training (PBT).

**Key results so far** (held-out KL on 8 teacher sequences, 2.05M tokens):
- Single run, hand-picked settings: 7.43. Trits frozen: ~9.5. 2k tokens per update: 7.67; 8k per update: 9.56.
- PBT with 8 members: best 2.61 at 1.79M tokens and 2.67 at 2.05M (Adam rate ~1–2e-3, θ 10–16, τ 1.2–2.5k,
  ρ× 0.6–1.0).

**Open problems:**
1. **Not reproducible.** PBT member 0 had the single run's exact settings and data, yet reached 8.00 vs ~9.9 at
   256k tokens. Noise flips at ρ ≈ 1 probably make trajectories chaotic, so single-run comparisons are unreliable.
2. **Unstable.** Training KL swings ±0.5 late in a run, and the leading PBT lineage collapsed from 2.61 to 7.8 in
   the last round.
3. **Loss scale.** The dynamic fp16 loss scale wanders between 8 and 2048 across runs. Suspected of both effects,
   through gradient underflow at low scale and something at high scale.
4. **No float baseline.** We don't know how much ternary-on-ANE costs against a float model of the same shape
   trained on the GPU.

## Overnight plan, 20:45 → 08:00 GMT

The lead (Opus) plans, decides and reviews. Sonnet subagents do the implementation, analysis and write-ups.
Only one heavy job (GPU/ANE) runs at a time, and the lead starts it.

| phase | GMT | work | machine |
|---|---|---|---|
| 0 | 20:45–21:30 | Tooling: `--seed`, fixed `--hyper`, `--loss-scale`/`--fixed-scale`, JSONL eval output, gradient underflow and overflow diagnostics, a run queue script | idle |
| 1 | 21:30–23:00 | Variance: best PBT settings × 3 seeds, defaults × 2 more seeds. Loss scale: fixed 16 / 256 / 2048 vs dynamic, best settings. All P=1, 2.05M tokens | ~95 min queue |
| D1 | 23:00 | Decide: how big is the variance, and which loss-scale policy and stabiliser to adopt | |
| 2 | 23:00–02:00 | Stabiliser (per D1) built in a worktree during Phase 1; PBT v2 with the stabiliser, 32 eval sequences, P=8, 2.05M tokens. Meanwhile a float GPU baseline trainer is built in a worktree | ~30–60 min runs |
| 3 | 02:00–06:30 | Float baseline vs ternary ANE at equal tokens (best settings, 3 seeds each if time). Long run: best settings or PBT v2 to 8.2M tokens | ~3–4 h queue |
| 4 | 06:30–08:00 | Analysis, README results, `MORNING.md` brief, commit and push | idle |

About 1.5 hours is deliberately unallocated.

**Decision rules:**
- **Replicates.** If the std of final KL across seeds is above 0.5: stability comes first, and every later
  comparison needs at least 3 seeds.
- **Loss scale.** If a fixed scale beats dynamic by more than the seed std: adopt it (or per-member scales), then
  rerun PBT.
- **Float baseline.** If float on the GPU beats ternary on the ANE by more than 1 nat at equal tokens: the morning
  question becomes precision (fp16 gradients) vs ternary capacity. Plan a GPU ternary-STE run to separate them.
- **Failures.** A crash or nonfinite result gets 30 minutes of investigation, then it's skipped and logged here.

**Run timings (M2 Max):**
- P=1, 2.05M tokens: about 12 min.
- P=8, 2.05M tokens: about 28 min.
- P=8, 8.2M tokens: about 1 h 50 min.

## Subagent protocol

- **Model and type.** Sonnet, `general-purpose`. Briefs are self-contained: goal, files, constraints, acceptance
  test, and the return format.
- **Code tasks.** Use `isolation: worktree` whenever a run is in progress, so a rebuild cannot touch the running
  binary. The subagent commits on its branch; the lead merges and rebuilds when the machine is idle.
- **Runs.** The lead starts them (in the background, one at a time) through the queue script. When a run finishes,
  an analysis subagent parses the JSONL and logs, writes `results/<name>.md` with a table and a receipt, appends a
  dated entry below, and commits and pushes.
- **Return format.** At most 15 lines: what changed or was found, commit hash, key numbers, anomalies. Details go
  in files, not replies.
- **Conventions.**
  - Native Swift only in the repo; throwaway parsing scripts go in the scratchpad.
  - Commit as you go and push to `origin main`.
  - Commit messages end with
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01E4WDxrcvTRwYmyP1TLLskz`.
  - README stays factual, with receipts linked.

## Entries

- 2026-11-23 20:40 — Plan written. Next: Phase 0 tooling.
- 2026-11-23 21:00. **P=1 was broken at HEAD.** MLX passes the 4-float `params` in `constant` memory, and the
  TTF kernel declared it `device` (7fd7052 fixes it). After the fix, P=1 defaults over 2.05M tokens reached
  held-out KL **7.517**, against 7.43 for the earlier single run. It had 39 overflow events, one roughly every
  200 updates: the dynamic scale doubles, overflows and halves, sliding from 1024 down to 4–8.
- 2026-11-23 22:30. **Phase 0 slipped by an hour.** zsh doesn't word-split `$A`, so the subagent's "64k" tests
  ran as default P=8 runs over 2.05M tokens. Those accidental runs answered two questions:
  - **Not bit-reproducible.** The same binary and seed match at the 0-token eval and diverge by 256k tokens.
    Likely cause: GPU atomics (scatter-add in the embedding gradient) combined with chaotic trit flips.
  - **But PBT-8 quality does reproduce.** Final best was 2.483, 2.455 and 2.668 (the earlier run) across three
    runs. Single runs (7.4–9.9) are the chaotic ones.
  - **Gradient diagnostics: the outputs underflow; they don't overflow.** Across the gp_ outputs, gmax is at most
    ~0.13, while some layers sit at gmax ~4e-6 with 93% exact zeros, at scale 32–64. So the "overflow" events
    must come from an internal ANE intermediate, not the gradient outputs, and the scale backs off into
    underflow. Next: find which tensor goes nonfinite, and give each tensor its own scaling.
- 2026-11-23 22:45. **Phase 1, first results** (queue in the scratchpad, `runs/phase1`).
  - `best-s0` (`--hyper 1e-3,12.8,1953,0.8`, P=1) hit "every member diverged" (nonfinite forward) within 40 s.
    Its log was lost to stdout buffering; 9867c25 line-buffers stdout.
  - `best-fix2048` (fixed loss scale 2048) overflowed on 7271 of ~7400 updates and never learned (KL 9.7
    throughout). So the overflow scales with the loss scale and is systematic. Suspect dx growing through the
    backward pass (RMSNorm backward on a tiny residual stream).
  - **Float baseline built** (`float-train`, 6d80af5): fp32 MLX, same architecture and pipeline, 20.2M
    parameters including the tied embedding. At 256k tokens it reaches **4.36** (lr 3e-3), 5.29 (1e-3) and
    5.29 (3e-4). Ternary-ANE P=1 is at ~9.5 at that point.
  - **Decision.** The float gap is far above 1 nat, so build `float-train --ternary` (fp32 GPU gradients with the
    same TTF) to separate fp16 numerics from ternary capacity. In parallel, a numerics agent is finding the
    overflow source and adding per-tensor power-of-two gradient prescales.
- 2026-11-23 23:00. **KEY: the ANE gap is numerics, not ternary or TTF.** `float-train --ternary` (70d0306) runs
  the same ternary model and the same TTF kernel and schedules, with fp32 autodiff gradients on the GPU. At 256k
  tokens it reaches **~5.0** with the default settings (9.17 → 6.64 → 5.95 → 5.42 → 4.99) and 5.13 with the best
  settings. ane-train at the same point is ~9.5, and float fp32 is 5.29 at lr 1e-3 and 4.36 at lr 3e-3. So
  fp16 ANE backward numerics cost ~4.5 nats at 256k tokens. The fix is the critical path.
  - **Phase 1 cut short.** best-s1 runs to completion; best-fix256 and best-s2 are dropped, because the variance
    of a numerically broken path isn't worth the machine time.
  - **Machine now:** GPU reference curves at 2.05M tokens (ternary-TTF with defaults and with best, and fp32 at
    3e-3 with and without the schedule).
- 2026-11-23 23:25. **Correction to the 23:00 entry.** The ANE gap was mainly an overflow-handling bug, not fp16
  numerics as such.
  - The numerics agent (2561608, `results/grad-numerics-2026-11-23.md`) found that layer 0's ds_* (and the
    dg/dk) outputs overflow. They are token-sum matmuls and the gradient grows toward layer 0. The old finite
    check looked only at gp_* and dx0, so those infs went straight into Adam. Each overflow it did catch halved
    the scale, which then underflowed gp_*.
  - Same settings (`--hyper 1e-3,12.8,1953,0.8 --seed 1`, P=1), 256k tokens:

    | binary | train KL | held-out KL | loss scale |
    |---|---|---|---|
    | old (best-s1) | 9.27 | 9.35 | 8–128 |
    | new, checking every output | 6.55 | 5.40 | 512–2048, 3 overflows |

  - GPU fp32 TTF (tern-def-s0) is 4.94 at 256k and **2.232** at 2.05M tokens. Fixed eval set throughout.
  - `--grad-prescale` adds per-tensor power-of-two factors, inside one runtime input `f` (separate inputs hit
    the program's input limit). gzero falls from ~0.4 to ≤ 0.05 in the worst tensor. At 512k: 4.235 vs 4.408
    baseline, one seed.
  - **Now:** ANE P=1 base vs prescale × seeds 0–2 at 2.05M with the default settings (to match
    GPU tern-def); PBT-8 with prescale and without. GPU queue: fp32 references and tern-def seeds 1–2.
    One ANE job and one GPU job run concurrently, so wall-clock numbers from tonight are not throughput
    measurements.
- 2026-11-23 23:55. **ANE matches GPU fp32 at equal tokens** (single seeds, fixed 8-sequence eval set, 2.05M
  tokens, eval every 256k):

  | run | trunk | grads | curve 0 → 2.05M | final |
  |---|---|---|---|---|
  | ane-def-pre-s0 | ANE ternary, `--grad-prescale` | fp16 + per-tensor 2^k | 9.17 5.64 3.78 3.17 2.81 2.43 2.25 2.19 | **2.150** |
  | tern-def-s0 | GPU ternary TTF | fp32 | 9.17 4.94 3.80 2.78 2.73 2.48 2.35 2.25 | 2.232 |
  | tern-best-s0 | GPU ternary TTF, best settings | fp32 | 9.17 5.60 3.82 2.70 2.72 2.49 2.42 2.31 | 2.254 |
  | fp32-lr3e-3-s0 | GPU float, constant lr | fp32 | 9.37 4.42 3.65 2.91 2.83 2.89 2.78 2.43 | 2.453 |
  | ane-def-base-s0 | ANE ternary, no prescale | fp16 | 9.17 5.95, then diverged at ~300k | — |

  - The baseline diverged from a forward overflow (L3.a_mix nonfinite; gains, scales and taps grow).
  - The prescale run had 91 overflow events, all skipped, and no divergence. It may have been lucky.
  - A stabiliser agent is deriving parameter bounds (`--bound`) so the forward can't overflow by construction.
- 2026-11-24 00:10. **`--bound`** (6d0b2c3, `results/forward-bounds-2026-11-24.md`) clamps the float
  parameters after each Adam step.
  - What grows is the per-row scales (0.077 → 0.2–0.45). The forward is a product chain, quartic in the scales
    in the conv mixer and cubic in SwiGLU, and the residual reached 3e4.
  - Strict worst-case bounds are infeasible: they'd need s ≤ 0.0155, below init. So the caps are practical,
    using a row-sum factor alpha·D with alpha 0.1, which is not a proof.
  - P=1 at 512k, seeds 0–2: no divergence; KL 4.36, 4.50 and 4.24 (unbounded: 5.13). The cap on o's scale
    binds everywhere.
  - Attention's `e @ (V·2^8)` is still unbounded in the worst case.
  - PBT-8 with prescale: best 3.36 at 768k, 4 divergences replaced, 222 overflow events.
  - **Queue (ane3):** prescale seeds 1–2, bound+prescale seeds 0–2, base seeds 1–2, then a long run of 8.2M
    tokens with whichever wins. GPU: tern-def seeds 1–2, then tern-def-long (8.2M).
- 2026-11-24 00:55. **Seeds.**
  - **GPU tern-def, seeds 0–2:** 2.232 / 2.439 / 2.187. Mean 2.29, std ~0.13, so the D1 rule (std > 0.5)
    isn't triggered for the GPU path.
  - **ANE prescale without bound:** seeds 1 and 2 both diverged at ~0.8M tokens ("every member diverged"; seed 0
    survived to 2.150). **The forward bound is necessary at P=1.** Bound+prescale seeds 0–2 are running now.
  - **PBT-8 with prescale:** best 2.243 at 2.05M. Replacing diverged members keeps it alive, but it does no
    better than a lucky P=1 run.
- 2026-11-24 01:55. **Headline holds across seeds.** Held-out KL at 2.05M tokens, P=1, default settings:

  | path | s0 | s1 | s2 | mean ± std |
  |---|---|---|---|---|
  | ANE ternary, `--grad-prescale --bound` | 2.203 | 2.278 | 2.253 | **2.245 ± 0.04** (0/3 diverged) |
  | GPU ternary TTF, fp32 gradients | 2.232 | 2.439 | 2.187 | 2.286 ± 0.13 |
  | ANE ternary, `--grad-prescale` only | 2.150 | diverged ~0.8M | diverged ~0.8M | |
  | ANE ternary, neither (post-fix) | diverged ~0.3M | | | |

  - **GPU tern-def-long (8.2M tokens, cosine over the whole run):** 3.05 at 1M, 2.82 at 2M, 2.33 at 3M, 2.16 at
    4M, 2.06 at 5M, 1.92 at 6M, 1.84 at 7M, **1.832** at 8.2M. The ANE long run (bound+prescale, 8.2M) has
    started.
- 2026-11-24 03:25. **Long runs** (8.2M tokens, seed 0, cosine over the whole run, eval every 1.024M):

  | run | 1M | 2M | 3M | 4M | 5M | 6M | 7M | 8.2M |
  |---|---|---|---|---|---|---|---|---|
  | ANE ternary, bound+prescale | 3.55 | 2.37 | 2.20 | 2.17 | 2.05 | 1.98 | 1.79 | **1.794** (225 overflow events, no divergence) |
  | GPU ternary TTF, fp32 | 3.05 | 2.82 | 2.33 | 2.16 | 2.06 | 1.92 | 1.84 | 1.832 |
  | GPU float fp32, schedule | 3.06 | 2.45 | 2.18 | 2.03 | 1.88 | 1.70 | 1.66 | 1.607 |

  - The ANE still matches GPU ternary at 4× the tokens.
  - Float now leads ternary by ~0.2 nats. That's the first sign of ternary capacity mattering, at d 256 with 3.4M
    trits.
  - Running: seed-1 replicates of both GPU long runs; ANE bound-only seeds and PBT-8 with bound+prescale.
- 2026-11-24 04:50. **Ablation complete** (2.05M tokens, P=1, default settings, seeds 0/1/2):
  - **Bound only:** 2.160 / 2.243 / 2.250, mean 2.218 ± 0.05.
  - **Bound+prescale:** 2.203 / 2.278 / 2.253, mean 2.245 ± 0.04.
  - **Neither / prescale only:** 4 of 4 diverged, except pre-s0.
  - So the stabiliser that matters is the bound, on top of the full finite check. Prescale cuts gradient zeros
    (0.4 → 0.05) but buys no KL here. Keep it off by default.
  - **PBT-8, bound+prescale:** best 2.172; all 8 members within 2.17–2.21; 0 divergences. PBT drifted to τ ~3–4k
    and θ 12–19, lr 1.5–2.8e-3. The gain over P=1 defaults is only ~0.05–0.1 at this budget.
  - **GPU ternary long, seed 1:** 1.811 (seed 0: 1.832).
  - Remaining: GPU fp32 long seed 1 (~05:30) and ANE long seed 1, bound+prescale (~06:30).
- 2026-11-24 05:45. **Long-run seed 1 done.**

  | run | seed 0 | seed 1 | mean |
  |---|---|---|---|
  | ANE ternary, bound+prescale | 1.794 | 1.716 | 1.755 |
  | GPU ternary TTF, fp32 | 1.832 | 1.811 | 1.822 |
  | GPU float fp32 | 1.607 | 1.555 | 1.581 |

  - ANE and GPU ternary tie within seed spread. Float leads ternary by ~0.17–0.24.
  - ANE long seed 1 mostly ran alone. Per sequence: teacher scoring 27.8 ms, Adam/TTF update 12.4, head 8.6,
    accumulate 8.1, ANE backward 6.7, ANE forward 3.6, plus teacher generation 33.8. That is 3807 tok/s
    excluding generation, and the ANE trunk is ~15% of the non-generation time.
  - A short clean-throughput queue is running (one job at a time).
- 2026-11-24 05:55. **Clean throughput** (`results/throughput-2026-11-24.md`), excluding text generation:
  - ANE P=1: 4239 tok/s. ANE P=8: 11,455 member-tok/s. GPU ternary: 5500. GPU fp32: 5740.
  - The ANE trunk is 9.9 of 60 ms per sequence at P=1; the GPU-side teacher scoring, update, head and
    accumulation dominate.
  - Phase 4 done: MORNING.md written.
- 2026-11-24 06:30. **PBT settings at P=1** (`--bound`, `--hyper 2.27e-3,15.4,2947,0.83`, seeds 0–2):
  - Results 2.047 / 2.156 / 2.253 (mean 2.152), against defaults 2.160 / 2.243 / 2.250 (mean 2.218).
  - Paired differences: −0.11, −0.09, +0.00. Suggestive; selected on the same eval set.
  - End of the overnight session.
- 2026-11-24 07:00. **Bound on by default** (8f94889; `--no-bound` to disable).
- **Next, as discussed with the user:**
  1. **Fuse teacher scoring into generation.** The scoring pass (24.6 of 60 ms per sequence) is redundant with
     the decode logits.
  2. **`float-train --ternary-ste`.** fp32 master weights with a straight-through estimator: is the 0.2-nat
     float lead capacity, or the TTF optimiser?
  3. **Teacher-led vs teacher-verified (on-policy) data**, built on step 1's pipeline:
     - **Signal:** per-token overlap α = Σ min(p, q) = 1 − TV, the speculative acceptance rate. It's free from
       logits we already have.
     - **Knob:** tempered verification, accept with probability min(1, (p/q)^β). β = 1 is exact teacher text;
       β = 0 is pure on-policy.
     - **Arms (float-train --ternary first):** teacher-led, a fixed switch, a λ ramp, and an α-triggered switch.
     - **Evals:** held-out KL on teacher text, plus the teacher's KL/NLL on the student's own samples.
- 2026-11-24 11:45. **The float lead was the optimiser, not ternary capacity.**
  `float-train --ternary-ste` (fp32 master weights, per-row absmean STE, lr 1e-3; receipt
  `results/ternary-ste-2026-11-24.md`):

  | run | 8.2M tokens, s0 / s1 | mean | 2.05M tokens, s0 / s1 |
  |---|---|---|---|
  | STE ternary | 1.616 / 1.505 | 1.56 | 2.275 / 2.538 |
  | float fp32 | 1.607 / 1.555 | 1.58 | 2.390 (s0 only) |
  | TTF ternary | 1.832 / 1.811 | 1.82 | 2.232 / 2.439 / 2.187 |

  - TTF's disadvantage appears only after ~2M tokens. n = 2; STE vs float is unresolved, but STE beats TTF on
    both seeds.
  - Implication for the ANE: master-weight-free TTF costs ~0.25 nats at long budgets. Options: switch from TTF
    to master weights later in training, use low-precision master weights, or retune θ/τ for long runs.
- 2026-11-24 12:00. **Fused teacher scoring** (e1af7bb plus a fix). Training targets are now the decode-step
  log-probs, so the separate scoring pass (24.6 ms/seq) is gone.
  - **Fix:** the first version stacked one [pool, T, V] array, which passes 2^31 elements at pool 256 and gave
    garbage targets (pool 128 sat exactly on the edge). Targets now stay as per-step [pool, V] bf16 arrays, and
    one sequence's [T, V] is gathered when it's used. Default pool 256.
  - **Equivalence** against the scoring pass (agent, pool 128): mean KL ~1e-4, max |Δ log p| 0.3–0.7 in tail
    tokens (bf16 and KV-cache differences).
  - **Check** (P=1, 256k tokens, pool 256): ane-train 9.169 → 6.019 → 5.081; float-train --ternary 9.170 →
    5.919 → 5.053.
  - **Speed:** ane-train 6268 tok/s excluding generation, generation 34.5 ms/seq, ~3.4k tok/s end to end (was
    ~2.8k). float-train --ternary 9444 tok/s, ~4.2k end to end. Teacher decoding now dominates, which on-policy
    data would remove.
- 2026-11-24 14:30. **Teacher-led vs on-policy** (`results/onpolicy-2026-11-24.md`; float-train --ternary,
  2.05M tokens).
  - **α (train) rises:** 0.09 at 256k, 0.27 at 512k, 0.37, 0.48 at 1M, then plateaus at ~0.42–0.54.
  - **Held-out KL** on teacher text, seed 0:

    | run | held-out KL |
    |---|---|
    | A teacher-led | 2.190 |
    | B all on-policy | 3.117 |
    | C ramp 0.5–1.5M | 2.592 |
    | D α-triggered (fired early) | 2.708 |
    | E half on-policy after 1M | 2.278 |

  - **Paired seed 1:** A 2.353 vs E 2.322. So E ties A on teacher-text KL (n = 2).
  - **Exposure bias is real.** The teacher-led student's KL on its own text is 2.5–3.4, against ~2.2 on teacher
    text. E halves that on-policy KL (1.2–1.5).
  - **But samples don't get more teacher-plausible.** Teacher NLL is worse for E (~4.6 vs ~2.2). It's measured
    on 8 samples (range 1.1–9.6), so it's very noisy.
  - **Reading:** forward KL on student text is mass-covering. It teaches the student to match the teacher in its
    own states, but doesn't reward plausible sampling. The on-policy literature uses reverse KL or JSD there
    for that reason.
  - **No speed gain.** A student sequence costs ~29 ms (8 ms sampling plus ~21 ms scoring), against ~34 ms of
    teacher decoding. The teacher's forward FLOPs per token (~0.7 GFLOP) are the floor for any
    teacher-scored scheme. The real lever is sharing the teacher across a population.
  - **Next:**
    - reverse KL / JSD on the on-policy part;
    - a 64-sample on-policy eval;
    - budgets of 8.2M tokens, where exposure bias should matter more;
    - port to the ANE only after a clear winner.
- 2026-11-24 15:00. **North star, restated by the user.** Mould great pretrained models (multi-B parameters, possibly
  MoE or omni) into hardware-suited ternary or ANE-shaped forms via distillation, with maximum efficiency. LFM2-350M is
  only a testbed.
  - **Consequences:**
    - **Bytes per parameter is a first-class metric.** fp32 master weights plus Adam is ~14 B/param, which doesn't
      fit at 8B beside a teacher. TTF today is 6 B/param; a lean TTF could be ~1.25. So TTF stays the path, and its
      ~0.25-nat long-budget gap is a problem to close.
    - **Teacher FLOPs per token is the other floor.** Levers: layerwise local distillation first, then end-to-end
      KL; a population sharing one teacher pass; deliberate reuse of targets.
    - **Headline metrics:** tokens and wall-clock to recover X% of the gap.
  - **Started:** `ternarise`.
    - Ternarise LFM2-350M (absmean vs TWN, optionally keeping the first and last layers in fp).
    - Recover with TTF from a zero accumulator vs one initialised from the rounding residual.
    - fp32-master STE as an upper-bound reference only.
- 2026-11-24 15:20. **Aim restated by the user: training as accessible as inference, with a constant overhead; then
  make inference fast by fitting the architecture to the device.**
  - **Flywheel.** Faster inference gives a faster teacher and student, which gives faster training. Teacher decoding is
    34.5 of ~75 ms per sequence.
  - **Pieces we have:**
    - per-layer fused gradient → update on the ANE, so no full-model gradient;
    - population sharing one teacher pass.
  - **Pieces to build:**
    - lean or stateless TTF (agent queued);
    - layer-by-layer training to bound activations;
    - a converted model as a fast ANE writer while the exact teacher scores on the GPU concurrently, so a step costs
      max(GPU, ANE), not the sum;
    - 2-bit palettised weights on the ANE for decode bandwidth (needs a compile-to-disk path; untested).
  - **Queue on the machine lock:** on-policy objective (running) → ternarise → lean TTF.
- 2026-11-24 17:40. **On-policy objective** (`results/onpolicy2-2026-11-24.md`; float-train --ternary; seeds 0/1).
  E = half student-written after the midpoint; reverse KL (rkl) or JSD on the student-written text.
  - **2.05M tokens:**

    | run | held-out KL | teacher NLL of samples | rep4 |
    |---|---|---|---|
    | A teacher-led | 2.280 / 2.475 | 1.40 / 1.44 | 0.88 |
    | E, forward KL | 2.301 / 2.471 | 4.22 / 6.86 | 0.34–0.56 |
    | E, rkl | 2.301 / 2.219 | 0.92 / 1.14 | 0.92 |
    | E, JSD | 2.435 / 2.424 | 1.51 / 1.59 | 0.85 |

    Teacher text has rep4 0.556. So the low teacher NLL of A, rkl and JSD samples is mostly repetition (looping text
    is easy to predict). Forward-KL samples are the only diverse ones.
  - **8.2M tokens:**

    | run | held-out KL | on-policy KL | teacher NLL | rep4 |
    |---|---|---|---|---|
    | A | 1.853 / 1.812 | 1.85 / 1.35 | 2.91 / 2.62 | 0.71–0.75 |
    | E rkl | 1.945 / 1.876 | 0.88 / 0.82 | 2.49 / 2.70 | 0.71–0.75 |

    rkl halves the on-policy KL but costs ~0.08 held-out KL (same sign on both seeds). NLL is indistinguishable.
  - **Conclusion:** no on-policy variant clearly wins at this scale. We need a diversity-aware sample-quality
    metric before judging; teacher NLL alone is gamed by repetition. Park on-policy; keep it as a tool for later
    (exposure bias does get halved).
- 2026-11-24 18:30. **Lean TTF** (`results/lean-ttf-2026-11-24.md`; `--acc fp32|fp16|int8|stateless` in float-train
  --ternary; new `ttfLeanKernel`).
  - **2.05M tokens, seeds 0/1:** fp32 2.218 / 2.443; fp16 2.189 / 2.435; int8 (stochastic rounding) 2.169 / 2.467;
    stateless 5.456 / 5.682.
  - **8.2M tokens, seed 0:** fp32 1.857; int8 1.847.
  - **The accumulator drops from 4 B to 1 B per parameter at no measurable cost.** Zero state fails: about 15× more
    flips per step, because nothing averages out the gradient noise.
  - **Implication:** TTF training state = packed trits (~0.25 B) + int8 accumulator (1 B) ≈ 1.25 B/param, against
    ~0.25 B/param for inference weights. A constant 1 B/param overhead, not a multiple.
  - **Next:**
    - port int8 accumulators to ane-train;
    - try 4-bit state;
    - pack trits off-ANE and expand per layer.
- 2026-11-24 20:00. **Ternarising LFM2-350M** (`ternarise`, `results/ternarise-2026-11-24.md`). Self-distillation
  against the bf16 base, 2.05M tokens, seed 0.
  - **Starting points:** absmean KL 8.73 (perplexity 1.1e9); TWN KL 6.55 (perplexity 32k); TWN with layers 0 and 15
    kept in fp, KL 5.58 (perplexity 17k). Base perplexity is 32.37.
  - **Held-out KL at 0 / 256k / 1.02M / 2.05M tokens, wikitext perplexity at the end:**

    | run | held-out KL | final perplexity |
    |---|---|---|
    | TTF, accumulator from zero | 6.55 / 2.41 / 1.34 / 1.274 | 515 |
    | TTF, accumulator from the rounding residual | 6.55 / 2.04 / 1.30 / **1.248** | **456** |
    | STE reference (absmean init, untuned) | 8.71 / 2.45 / 1.64 / 1.440 | 983 |

    TTF used θ 32, τ 1000, ρ× 0.5, Adam 1e-5 on the floats.
  - **More than half the KL gap closes within 256k tokens, then the curve flattens as the schedule cools.** Flips
    stop after ~1.2M tokens. Not recovered: perplexity 456 vs 32. The 3-bit repair reached KL 0.80 in 600k tokens.
  - **TTF is at least as good as the master-weight reference** at half the state, but that comparison is weak.
  - **The residual accumulator init is an early-speed win** (2.04 vs 2.41 at 256k) and only ~0.03 at the end.
  - **Throughput:** TTF 1075 tok/s on the GPU at 350M. Peak memory 23.5 GB, mostly the 256-sequence target pool.
  - **Wikitext text** for perplexity is at /tmp/wikitext2-test-join.txt. The "\n\n"-joined variant gives teacher
    perplexity 33.13; the join variant matches the known 32.37.

## Night 2 plan, 2026-11-24 22:00 → 08:00 GMT

**Goal (user):** the best possible recovery of LFM2-350M in a ternarised, device-aligned architecture, with training
and inference co-optimised for this Mac.

| phase | GMT | work |
|---|---|---|
| A | 22:00–23:15 | Three subagents in worktrees. A1: ternarise infrastructure and speed (int8 accumulators, WSD schedule, checkpoint save/resume, throughput). A2: recipe features (two-sided row×column scales, layerwise reconstruction init). A3: ANE inference of the ternary 350M (prefill benchmark and parity, decode stretch). Meanwhile: an 8.2M-token baseline with today's recipe (does the curve saturate?). |
| B | 23:15–01:30 | 1M-token screens: float lr, two-sided scales, layerwise init (cost counted), WSD vs cosine, int8 accumulator, keep-fp first/last (mixed-precision track). |
| D1 | 01:30 | Choose the recipe. |
| C | 01:30–06:45 | One long final run, checkpointed every ~2M tokens, WSD schedule so it can end on time. |
| D | 06:45–08:00 | Final perplexity and KL; ANE inference benchmark of the final model; write-up; MORNING.md; push. |

**Checkpoint format** (A1 and A3 agree): a directory holding the base `config.json` plus `model.safetensors`.
For each ternary linear `<path>.trits` (int8 [N,K]), `<path>.scale` (fp32 [N]) and optionally `<path>.colscale`
(fp32 [K]); every other parameter under its mlx-swift-lm name.
- 2026-11-24 22:25. **A3: ANE inference of the ternary 350M** (`ane-infer`, `results/ane-infer-2026-11-24.md`).
  - **Parity with the GPU ternary model:** prefill KL 0.0002, argmax 98.9%; decode argmax 100%; no overflow.
  - **The ANE is slower than the GPU on the M2 Max:**

    | path | prefill tok/s | decode tok/s |
    |---|---|---|
    | ANE | 3.9k (3.2–3.7 ms per layer) | ~30 |
    | GPU bf16 | 10.8k | 157–223 |

    Decode looks dominated by per-program dispatch, 16 programs per token; fusing layers is the fix.
  - **Decision: co-optimised inference on this Mac = the GPU with packed weights.** Decode is bandwidth-bound, and
    ternary is exactly MLX 2-bit affine (q = t+1, scale = s, bias = −s), so ~8× fewer weight bytes than bf16.
    Agent B4 is building an exact export and a benchmark.
  - **Colscales** are tied per shared input (q/k/v; w1/w3), so they fold into the norm gains. out_proj/o_proj/w2
    colscales fold into the C/v/w3 rows. Zero inference overhead.
  - **The ANE's role on this machine:** concurrency (student or writer beside the GPU teacher) and efficiency, not
    peak speed.
- 2026-11-24 22:40. **B4: exact 2-bit GPU export** (`ternary-export`, `results/gpu-ternary-infer-2026-11-24.md`).
  - **Exactness:** MLX's dequantized() of all 92 linears equals trits·bf16(scale), with 0 mismatches.
  - **Colscale folding is exact.** Untied colscales abort.
  - **Parity:** KL 0.004–0.005 per token against fp32 TernaryLinear (bf16 floor 0.002–0.004).
  - **Quiet benchmark:**

    | model | prefill B1 / B8 tok/s | decode tok/s | size |
    |---|---|---|---|
    | bf16 | 10.4k / 13.0k | 262 | 709 MB |
    | ternary 2-bit | 4.6k / 10.1k | **441** (1.7×) | 224 MB |
    | MLX 4-bit | 8.5k / 9.6k | 397 | 200 MB |

    A 4-bit embedding costs KL 0.47, so the embedding stays bf16; it is most of the 224 MB.
  - **Plan adjustment.** The 8.2M baseline runs at only ~413 tok/s while agents share the GPU, so it can't finish
    before Phase C. At D1 it gets stopped wherever it is, and the final run gets the GPU alone.
- 2026-11-24 23:00. **A1 merged.**
  - **Checkpoints:** `--save DIR --save-every N --resume DIR`. A resume is bit-continuous on the eval.
  - **Schedule:** `--schedule wsd --decay-frac F`.
  - **Accumulators:** `--acc int8`, with the residual init quantised. At 128k tokens: KL 2.81 vs 2.77 for fp32.
  - **Profiling:** `--profile`. Per update: generation ~40% of wall time, then forward+backward, then TTF+Adam. No
    throughput change was committed: bf16 trit copies gave no gain. Contended numbers are unreliable.
- 2026-11-24 23:20. **Phase B screens** (A2, 512k tokens end to end, TWN, residual accumulator, θ32 τ1000 ρ0.5):

  | trial | start KL | final KL | perplexity |
  |---|---|---|---|
  | baseline | 6.55 | 1.898 | 1095 |
  | colscale (tied) | 6.09 | 1.931 | 1223 |
  | **layerwise 256k first** | **3.005** after the layerwise phase | **1.138** | **305** |

  - The layerwise phase costs about the same per token as end-to-end training (339 s for 256k under contention).
  - **The 8.2M cosine baseline was stopped at 2.05M:** KL 1.012, perplexity 265. The 2.05M-schedule run reached
    1.248 / 456 at the same tokens. Cooling early hurts, so use WSD with a long plateau.
  - **Screening `--layerwise 1024000`** (then 512k end to end) to size the layerwise phase of the final run.
