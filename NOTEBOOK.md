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
