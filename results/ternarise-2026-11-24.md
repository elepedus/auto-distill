# Ternarising LFM2-350M and recovering it with TTF (2026-11-24)

Tool: `ternarise` (Sources/ternarise/main.swift). Teacher: LiquidAI/LFM2-350M in bf16. Student: the same model with every
decoder-layer Linear (conv in_proj/out_proj, attention q/k/v/o, FFN w1/w2/w3; 287.3M of 354.7M parameters, 81%) replaced by
trits x per-row scale. Embedding/tied head, norms and conv taps stay float (fp32 in the student; matmuls run in bf16).
Held-out KL: 8 teacher-generated sequences of 256 tokens, MLXRandom.seed(424242), full-vocabulary KL(teacher || student).
Perplexity: wikitext-2 test (the `distill eval` windows, 64 x 512 tokens; text rebuilt as `"".join(rows)`, which reproduces the
known 32.37 for the bf16 teacher; the fp32-stream student with no layer ternarised gives 32.34 and KL 0.0008, so the harness is clean).
Runs: `results/runs-2026-11-24/ternarise-*.jsonl`.

## Starting points (no training)

| init | held-out KL | alpha (1-TV) | wikitext-2 ppl | zero trits |
|---|---|---|---|---|
| bf16 teacher | 0 | 1 | 32.37 | |
| absmean (s = mean abs W, t = round(clip(W/s, -1, 1))) | 8.73 | 0.49 | 1.1e9 | 31.5% |
| TWN (delta = 0.7 mean abs W) | **6.55** | 0.045 | 32 355 | 42.9% |
| TWN, layers 0 and 15 kept bf16 | 5.58 | 0.088 | 17 012 | |

Both inits destroy the model; TWN is the better start by KL and by six orders of magnitude of perplexity, so it was used for
the TTF runs. Keeping the first and last layer in bf16 helps only ~1 nat. (Absmean has the higher KL yet higher alpha than TWN; its perplexity is worst by far, so KL on 8 sequences and ppl rank the inits differently in alpha but agree TWN is the better start.)

## TTF scan (residual acc-init, TWN init, 256k tokens, seed 0, lr 1e-5)

| theta, tau, rho | held-out KL @256k | ppl @256k | trits flipped in total |
|---|---|---|---|
| 16, 1000, 0.25 | 2.497 | 2970 | 13.1M |
| **32, 1000, 0.5** | **2.468** | 2407 | 12.7M |
| 16, 1000, 0.1 | 3.315 | 4020 | 3.0M |

Low rho under-trains; theta/rho in {16/0.25, 32/0.5} are equivalent to within noise (one seed each). The 2.05M runs use
theta 32, tau 1000, rho 0.5, Adam lr 1e-5 for scales/norms/embedding/conv taps. Rho and Adam lr follow float-train's
warm-up (5% of updates) and cosine schedules.

## Recovery, 2.05M tokens (8000 updates of 256 tokens, seed 0)

Held-out KL (lower is better), wikitext-2 perplexity in the last two columns' rows where measured:

| tokens | (a) TTF acc zero | (b) TTF acc residual | (c) STE reference |
|---|---|---|---|
| 0 | 6.547 | 6.547 | 8.710 (absmean) |
| 256k | 2.411 | 2.044 | 2.445 |
| 512k | 1.752 | 1.545 | 2.042 |
| 768k | 1.441 | 1.439 | 1.865 |
| 1.024M | 1.340 (ppl 583) | 1.304 (ppl 502) | 1.638 (ppl 1657) |
| 1.28M | 1.305 | 1.270 | 1.558 |
| 1.536M | 1.286 | 1.254 | 1.496 |
| 1.792M | 1.278 | 1.250 | 1.446 |
| 2.048M | **1.274** | **1.248** | 1.440 |
| ppl at 0 | 32 355 | 32 355 | 1.1e9 |
| ppl at end | **515** | **456** | 983 |

The STE reference starts from the absmean init (per-row absmean STE re-derives the scale from the master weight each step), not
from TWN, so its curve starts higher. Alpha (1-TV) at the end: 0.673, 0.658 (b), 0.656 (c). Every run sits at KL ~1.25-1.45
and ppl 450-1000 against the teacher's 32: this is a long way from recovered.

Tokens to close a fraction of the starting KL gap (first two consecutive 25-update training-KL windows below the threshold, so
resolution is ~6k tokens and the numbers are noisy; the held-out evals confirm all three runs had closed >50% by 256k):

| | 25% | 50% | 75% |
|---|---|---|---|
| (a) TTF zero | ~115k | ~147k | ~429k |
| (b) TTF residual | ~13k | ~51k | ~403k |
| (c) STE | ~13k | ~38k | ~224k |

Trit flips (TTF): (a) 2.81M flipped in total (1.0% of trits), all in the first ~1.2M tokens, up to 2100 per update, then none once
rho decays. (b) 3.1M flip in the very first update (1.1% of trits: the residual init puts them right at the threshold), 11.7M in
total (4.1%), nothing after ~1.3M tokens. So most of the structure recovery is the scale/norm/embedding path plus a few percent
of trit flips; TTF freezes itself as the schedule cools.

## Throughput (M2 Max, one sequence of 256 tokens per update, run alone)

| mode | tok/s excluding pool generation | including generation | peak MLX memory |
|---|---|---|---|
| TTF | 1280 | 1075 | 23.5 GB |
| STE | 905 | 792 | 25.2 GB |

Pool generation (teacher decode of 256 sequences with fused targets): 36 ms per sequence. The peak memory is dominated by the
256-sequence target pool (256 x 256 x 65536 bf16 = 8.6 GB) and fp32 activations/logits, not by the optimiser state.

## Bytes per parameter (optimiser plus weight state, computed from the implementation)

| mode | per ternary parameter | breakdown | whole model (incl. fp32 embedding, norms, scales with Adam) |
|---|---|---|---|
| TTF as built | **6 B** | fp16 trits (2) + fp32 accumulator (4); factored row/col moments ~0 | 7.14 B |
| TTF packed (not built) | ~2.25 B | 2-bit trit + fp16 accumulator | |
| STE reference | 12 B | fp32 master (4) + Adam m, v (8) | 12.0 B |
| fp32 master + Adam, plain | 12 B (+4 grad) | | |

Both modes also hold ~4 B per parameter transiently for the fp32 gradient of a matrix (and, as built, a fp32 copy of the trits
handed to autodiff: 4 B more). TTF's per-parameter state is half of STE's as built and ~5x smaller if the trits and
accumulator were packed; at multi-billion scale the state would be governed by the accumulator. The 1.4 GB + 1.4 GB transient copies per full model are the next thing to remove (per-layer
gradients inside the backward pass).

## Conclusion (with uncertainty)

- Post-training ternarisation of LFM2-350M is catastrophic (ppl 3e4 to 1e9) and 2M tokens of distillation recovers only part
  of it: best ppl 456 (KL 1.25) against 32.4. The north-star result (a usable ternary conversion) is not reached at this budget.
  The curves are still bending down slowly at the end, but the cosine schedule has also cooled to 10% of peak, so this is the
  end of this recipe rather than evidence about the limit of the approach.
- TTF without master weights matches and, here, beats the fp32-master STE reference (KL 1.25-1.27 vs 1.44; ppl 456-515 vs 983)
  at half the state per parameter (6 vs 12 B as built). The comparison is not clean: STE starts from absmean (worse than
  TWN), its lr was never tuned (1e-5 on the master weights moves them 0.08 at most), and one seed each. Safe reading: TTF is
  not worse than a lightly tuned STE; I would not claim it is better.
- Residual accumulator init helps early (KL 2.04 vs 2.41 at 256k; ppl 456 vs 515 at the end) and the gain shrinks to ~0.03 nats
  by the end; with single seeds and an eval of 8 sequences (KL noise ~0.02-0.05), it is a small, probably real, early-speed win
  and an unproven final-quality win.
- Caveats: one seed per cell; the 8-sequence held-out set is teacher text, so perplexity on wikitext (only 32k tokens) is the
  independent check and agrees in ordering; hyperparameters were scanned only at 256k tokens and over three settings; the
  teacher-generated data is only the teacher's own samples (no real text).
