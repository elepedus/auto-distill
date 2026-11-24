# Ternary with fp32 master weights and STE: is the float-vs-TTF gap capacity or optimiser? (2026-11-24)

**Question.** At 8.2M tokens float fp32 beats ternary-TTF by about 0.2 nats of held-out KL (1.607/1.555 vs 1.832/1.811).
Is that ternary *capacity*, or TTF (a threshold-flip optimiser with no master weights) being less sample-efficient?

**Method.** `float-train --ternary-ste`: every linear that `--ternary` makes ternary instead keeps a latent fp32 weight W
(float-mode init, std 1/sqrt(k)). The forward uses W_q = s * round(clip(W/s, -1, 1)), s = per-row absmean of W (+1e-5),
derived from W, not a parameter (BitNet b1.58 style, per row). Gradient goes straight to W: W + stopGradient(W_q - W).
Adam on everything, ane-train warm-up + cosine schedule, `--lr` the peak. Embedding, norms and taps as in the other modes.
Same teacher data, same held-out KL eval. Zero-trit fraction of W_q stays at 0.31 throughout (logged per eval).

**LR scan** (256k tokens, `--round 128000`, seed 0, final held-out KL): 3e-4 5.740, **1e-3 4.944**, 3e-3 5.160, 1e-2 5.810.
1e-3 is best and bracketed, so all main runs use `--lr 1e-3`. (The 256k scan favours fast early progress and cannot
be read as a statement about the best lr at 8.2M.)

**Results** (held-out KL, same eval set).

| tokens | STE s0 | STE s1 | TTF s0 / s1 | float s0 / s1 |
|---|---|---|---|---|
| 1.02M | 3.027 | 3.716 | 3.046 / 3.425 | 3.055 / 3.450 |
| 2.05M | 2.440 | 2.573 | 2.821 / 2.634 | 2.454 / 2.653 |
| 3.07M | 2.075 | 2.036 | 2.326 / 2.307 | 2.182 / 2.166 |
| 4.10M | 1.962 | 1.958 | 2.159 / 2.140 | 2.027 / 2.095 |
| 5.12M | 1.862 | 1.764 | 2.057 / 2.097 | 1.875 / 1.915 |
| 6.14M | 1.653 | 1.579 | 1.915 / 1.883 | 1.702 / 1.682 |
| 7.17M | 1.674 | 1.580 | 1.836 / 1.882 | 1.658 / 1.647 |
| **8.19M** | **1.616** | **1.505** | 1.832 / 1.811 | 1.607 / 1.555 |

(The 2.05M row is the 8.2M run's schedule, not a 2.05M run.) Separate 2.05M-token runs (schedule ends at 2.05M),
final KL: STE 2.275 / 2.538 (seeds 0/1) against TTF 2.232 / 2.439 / 2.187 (seeds 0/1/2) and float with schedule 2.390 (seed 0).
Files: `ste-s{0,1}`, `ste-long-s{0,1}`, `ste-scan-*` in `results/runs-2026-11-24/`.

**Answer.** At 8.2M the gap is the optimiser, not capacity. STE-ternary reaches 1.616 / 1.505 (mean 1.56), matching
float (1.607 / 1.555, mean 1.58) and beating TTF (1.832 / 1.811) by about 0.26. A ternary network of this shape can
reach float quality at this budget; TTF just gets there more slowly. The curves agree: from 3M tokens STE tracks or beats float
and sits 0.2 below TTF.

**Uncertainty.** Two seeds per method. Seed-to-seed spread of STE at 8.2M is 0.11, similar to the 0.05-0.13 std at
2.05M, so STE vs float (difference 0.02) is not resolved: STE may be slightly better, equal, or slightly worse. The
STE-vs-TTF gap (0.26 mean, both STE seeds below both TTF seeds) is larger than that spread but rests on n=2.
At 2.05M with a full schedule the picture is different: STE (2.275/2.538) is no better than TTF (2.19-2.44), so TTF is
not less sample-efficient there. The TTF disadvantage appears after about 2M tokens. The lr was tuned only at 256k and only for
STE; float and TTF use earlier settings, which could bias either way. Whether a ternary model can *exceed* float, or hold
its advantage past 8.2M, is not tested.
