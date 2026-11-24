# Forward-activation bounds for the fp16 ANE trainer (2026-11-24)

## Diagnosis
`ane-train --population 1 --tokens 2048000 --seed 0` diverged at ~300k tokens: before the nonfinite forward the overflow
scans showed `L3.a_mix` (= c*conv, input of the o projection) going nonfinite and `L3.y` = 4.4e4. A new per-eval line
(`params max|.| m0:` and JSONL `{"type":"params",...}`) prints max |value| of every float parameter of member 0 and the
max of each layer's residual output `y`. Repro (`--population 1 --tokens 512000 --round 64000 --seed 0`, no bound;
it did not diverge this time, nondeterminism, but shows the growth: `diag-s0.jsonl`):

| tokens | gains g_* | taps k0,k1 | tap k2 | s (b,c,x,q,k,v) | s_o | s_w1,w3 | s_w2 | residual y, L0..L3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 0 | 1.00 | 0.077 | 0.077 | 0.077 | 0.044 | 3 .. 7 |
| 64k | 1.13 | 0.09 | 1.08 | 0.21 | 0.17 | 0.23 | 0.21 | 89 .. 680 |
| 192k | 1.22 | 0.22 | 1.20 | 0.27 | 0.40 | 0.24 | 0.28 | 1.3e3, 2.6e4, 2.6e4, 2.6e4 |
| 320k | 1.34 | 0.37 | 1.21 | 0.30 | 0.46 | 0.25 | 0.30 | 1.2e3, 3.1e4, 3.1e4, 3.1e4 |
| 512k | 1.34 | 0.37 | 1.21 | 0.30 | 0.44 | 0.25 | 0.28 | 1.6e3, 4.0e3 .. 4.7e3 |

The RMSNorm gains and taps barely move (<= 1.35). **The per-row scales of the ternary linears grow 3-6x** (they get a
large Adam rate, lr 2e-3 on values of 0.08), and the forward is a product chain of them: conv mixer
`r = s_o * T(c * conv(b * x))` is quartic in the scales, the SwiGLU `m = s_w2 * T(silu(s_w1 T h) * s_w3 T h)` cubic.
Layer 1's output jumps to 2.6e4 (a single token with a massive activation) and the residual stays there for layers
2, 3; one more step in any scale and `a_mix` or `y` exceeds 65504. Also the fp32 GPU run is fine because fp32 has no such limit.

## Bounds
Notation: D = 256, x_hat = RMSNorm output, |x_hat_j| <= sqrt(D) = 16 and ||x_hat||_2 = sqrt(D). Gain cap g (elementwise),
so for a = g*x_hat and a ternary row t with scale s:

    |s * sum_j t_j a_j| <= s * ||a||_1 <= s * g * ||x_hat||_1 <= s * g * D          (Cauchy-Schwarz: ||x_hat||_1 <= D)

so a linear output is bounded by `Y = s*g*R` with `R = D` (strict). Per op:
- in_proj (b, c, x, q, k, v, w1, w3): |y| <= s*g*R = Y.
- bx = b*x: <= Y^2; conv (3 taps, |k_j| <= kc_j): <= K*Y^2, K = sum kc_j; a_mix = c*conv: <= K*Y^3.
- o projection: r = s_o * T(a_mix): |r| <= s_o*R*K*Y^3 (note a_mix is not normalised, so R enters again).
- FFN: silu(u) <= |u|, so |silu(w1)*w3| <= Y^2 and m = s_w2*T(.) <= s_w2*R*Y^2.
- Attention: q,k are RMS-normalised per head (|x_hat| <= 8): |q.k|/8 <= 8*g_q*g_k*8... = 64*g_q*g_k/... <= 512 g_q g_k = 1152 at g = 1.5, so
  logits and exp(s - max) are safe; v <= Y; attention out <= Y (convex combination); r = s_o*R*Y.
  **Not covered:** `e @ (V*2^8)` before the division sums up to T terms: worst case T*T*256*|v|/T... = 2^16*|v|, which overflows
  for uniform attention unless |v| < 0.5. This needs a graph change (drop or lower the 2^8) and was left alone.
- Residual: y = x + r + m accumulates over 4 layers, so r and m each get a budget B = 2048: |y| <= 8 B + |embedding| ~ 16.4k.
- Gains and the taps are clamped elementwise (`g <= 1.5`, `k0, k1 <= 0.5`, `k2 <= 1.5`: K = 2.5), which is close to what
  training reaches (1.35, 0.37, 1.21).

**Strict (R = D) is infeasible.** With all scales equal, `s^4 R^4 g^3 K <= B` gives s <= 0.0155, 0.2x the
initial 0.0765, so even the initial model violates it: the worst case needs every |x_hat_j| = 1 aligned with the trits
in every row, which random-sign trits never do. **Practical choice:** replace R = D by `R = alpha*D`, alpha = 0.1
(`--bound-alpha`): the typical max row sum of a trained layer is ~0.1 of the l1 bound (a_xn max 4, a_hn max 19 against
the 16 g worst case in the failing log). Then (choose s_o = 0.12): Y = cbrt(B/(R*K*s_o)) = 6.44,
s_in <= Y/(R*g) = 0.168 (2.2x init), s_o <= 0.12, s_w2 <= B/(R*Y^2) = 1.9 (not binding). Observed worst residual at evals
5.4k; fits the ~16k target with margin. The projection is a clamp applied in `Param.update` after Adam, elementwise, so per member.

## Results
(P=1, 512k tokens, round 64k, KL held-out per eval at 0, 64k, ... 512k; baseline P=1 256k ~5.4-5.9 and GPU fp32 ternary 512k 3.80)

| run | diverged | backward overflow events | held-out KL at 0, 64k, 128k, ... 512k | max residual y at an eval |
|---|---|---|---|---|
| no bound, seed 0 (diagnosis) | no (the earlier 2M run diverged at 300k) | 5 | 9.169 6.664 6.178 5.569 5.975 5.391 5.334 5.269 5.127 | 3.1e4 (L1..L3) |
| --bound seed 0 | no | 5 | 9.169 6.618 6.832 5.544 5.374 4.824 4.613 4.472 4.356 | 5.4e3 |
| --bound seed 1 | no | 6 | 9.174 6.720 6.299 5.908 5.349 4.856 4.617 4.504 4.500 | 2.3e3 |
| --bound seed 2 | no | 5 | 9.068 6.341 6.115 6.159 6.099 5.247 4.481 4.364 4.235 | 3.4e3 |
| --bound --grad-prescale seed 0 | no | 19 | 9.169 6.538 6.659 5.286 5.251 4.909 4.598 4.568 4.536 | 2.3e3 |

The overflow events are all backward outputs (ds_*/dk*/dg_*, see grad-numerics-2026-11-23.md), which this change does not
target; forward activations stayed far below 16k at every eval. The bound did not hurt: 512k KL 4.24-4.50 against 5.13
for the unbounded seed-0 run, whose residual was at 3e4 (within 2x of overflow). s_o sits on its cap (0.12) in every layer
for the whole run, so alpha and s_o are the knobs if more capacity is wanted. Single seeds, nondeterministic: differences
under ~0.3 are noise. Not checked: 2M tokens (the original failure) and the attention numerator caveat above.
Logs: forward-bounds-*.jsonl.
