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
