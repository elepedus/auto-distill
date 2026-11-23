# auto-distill

Copy capabilities from one model to another without a dataset. The student writes its own text, the
teacher scores every token, and the student learns the teacher's full next-token distribution (KL).
Each batch is generated, used once and thrown away, so nothing is stored.

## `distill.py` (MLX)

```sh
uv run distill.py TEACHER STUDENT OUT [--steps 300] [--batch 8] [--lr 1e-5]
```

- **Quantised student:** only the quantisation scales and biases are trained, so the output is an ordinary
  MLX quantised model, with no adapter and no extra inference cost.
- **Full-precision student:** every weight is trained. Teacher and student must share a tokenizer.

First result, LiquidAI/LFM2-350M at 3 bits, 300 steps (about 600k self-generated tokens, about 12 minutes
on an M2 Max). Perplexity is on wikitext-2 test, which training never sees:

| model | perplexity | KL vs teacher |
|---|---|---|
| bf16 teacher | 32.37 | 0 |
| 3-bit | 56.56 | 0.866 |
| 3-bit, repaired | 48.10 | 0.731 |

A learning rate of 1e-4 destroys 3-bit scales within three steps; 1e-5 works.

## `ane/`: ternary on the Apple Neural Engine

`ternary-probe` builds one ANE program that multiplies fp16 activations by ternary weights ({-1, 0, 1}, fed
at runtime), in the forward (`x·W`) and backward (`g·Wᵀ`) directions. It compares every output bit for bit
with the exact sum rounded once to fp16.

On an M2 Max (macOS 27.2.1), every finite output matches the exact sum rounded once **with ties away
from zero** (not IEEE ties-to-even). That holds at K = 1024 and 4608, for 8-bit activation codes and for
general fp16 activations, so a simulator can reproduce these matmuls bit for bit. With 8-bit codes, a
power-of-two prescale chosen from 127·K makes overflow impossible; without it, the worst case is all `inf`.
Receipt: [`ane/results/ternary-probe-2026-11-23.txt`](ane/results/ternary-probe-2026-11-23.txt).

```sh
cd ane && swift build -c release && .build/release/ternary-probe
```

Building needs a local checkout of [swift-ane](ane/Package.swift), which is not published yet.
