# fp16 gradient numerics in the ANE backward (2026-11-23)

## Root cause of the overflows
Repro: `--population 1 --tokens 64000 --round 32000 --loss-scale 4096` (overflows at updates 49, 106, 158 with
the diagnostics added in this branch). Every event was the same: **`ds_*` of layer 0 (`ds_b`, `ds_c`, `ds_x`)
went nonfinite**, while the `gp_*` outputs of the same layer had max |value| 0.03..2.8.

- `ds_n = tokenSum @ (gy * pre)` sums gy·pre over all T=256 tokens in one matmul and emits fp16. `pre` is the raw
  ternary matmul output (about 1/scale = 16x larger than y), so ds is ~T·16 times bigger than the typical per-element
  gradient. `gp = gy*s` is the opposite: s~0.06 makes it tiny.
- gy grows towards the bottom of the network: max |dx| was 0.6 (L3), 7 (L1), 560 (L0) at scale 4096. At layer 0
  `ds_*` reached 7e3..1.5e4, `dk*` 2e3..8e3, `dg_op` 1.3e3; a spike at 4096 crosses 32752.
- The old check looked only at gp_* and dx, which stay finite, so the culprit was invisible. Halving the scale then
  pushes gp_* (max 4e-3..0.1 at scale 512) into underflow (gzero up to 0.5 per tensor).
- This also explains the `--fixed-scale 2048` run that overflowed on ~98% of updates: the layer-0 ds/dk/dg token
  sums sit right at the fp16 limit.
- ds/dk/dg (token sums) and gp (per element, times s) differ by 3-5 orders of magnitude, so no single global scale
  fits both. Per-tensor scaling is the fix.

## `--grad-prescale`
- Graph.swift `layerProgram(..., prescale:)`: one extra runtime input `f` [1,P,1,W] holding all factors (the ANE
  refuses programs with more than 32 inputs: 6 extra inputs on the conv layer already failed with "Request cancelled").
  Slices of it multiply gy (before the matmul) for ds_*, dk*, dg_*, and `gy*(s*f)` for gp_*. The dx path is
  untouched. The default build has no `f` input.
- Trainer: per (layer, output, member) exponent k in [-14, 14] (2^k must itself be an fp16 normal number, so the
  suggested +-24 is impossible); gradients are divided by lossScale*2^k in fp32 before TTF/Adam.
  Adaptation at steps 1, 2, 4, ... and every 50: k so the window max lands near 2^9 (2^11 after step 50).
  On overflow the first (top) layer with a nonfinite output is the culprit: its overflowing controlled outputs get
  k-4 (all members); the loss scale is halved only if an uncontrolled output (dx, a_*, y) is bad.
- Loss scale (protects dx only): adapted from max|dx0| every 50 updates to land near 2^11 (at most x4 per step),
  floor 256, cap 65536; k is shifted the opposite way when it changes, so gradient outputs do not move.
  dx0 is the largest value of the chain, so it is the one the loss scale must keep in range.
- Check: `lfm2-ane README.md [--grad-prescale]` (the prescale test sets all f = 8 and divides it out): layers
  0, 2, 14, 15 min cosine 0.9990 default, 0.9990 prescaled (all >= 0.998, nonfinite 0).

## Results (P=1, `--hyper 1e-3,12.8,1953,0.8`, seed 0, fixed eval set, 512k tokens, default loss scale 1024)
| tokens | baseline held-out KL | --grad-prescale |
|---|---|---|
| 0 | 9.169 | 9.169 |
| 128k | 6.046 | 6.304 |
| 256k | 5.198 | 5.189 |
| 384k | 4.506 | 4.468 |
| 512k | 4.408 | 4.235 |

Overflow events (2000 updates): baseline 6; prescale 20, all single `gp_*` tensors (mostly L0/L1 w1/w2/w3) spiking
more than 16x above their window max; each costs one skipped sequence and k-4 on that tensor. A 64k-token run
with the early-step adaptation had 0 overflows. Per-tensor zero fraction: baseline worst tensor 0.37..0.51 at the
prints where the scale had sunk, prescale mostly 0.01..0.05 with occasional 0.4 for tensors at the +14 clamp.
Aggregate gzero: baseline up to 0.08, prescale <= 0.02. One seed per arm: the 0.17 difference at 512k is within
the chaos seen in earlier single runs.

## Notes
- Eval set is generated under seed 424242, then reseeded with --seed (init, training data and hypers follow --seed).
- Not investigated: the "every member diverged" at `--hyper 1e-3,12.8,1953,0.8 --seed 0` with the pre-fix binary.
- Logs: grad-numerics-base512.jsonl, grad-numerics-prescale512.jsonl.
