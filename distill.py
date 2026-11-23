# /// script
# dependencies = ["mlx-lm"]
# ///
"""Dataset-free distillation. The student writes text, the teacher scores every
token, the student learns the teacher's full next-token distribution. Nothing is
stored: each batch is generated, used once and thrown away.

  uv run distill.py TEACHER STUDENT OUT [--steps 300] [--batch 8]

Quantised student: only the quant scales and biases are trained, so OUT is still
an ordinary quantised model with no adapter and no extra inference cost.
Full-precision student: every weight is trained.
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import save

p = argparse.ArgumentParser()
p.add_argument("teacher")
p.add_argument("student")
p.add_argument("out")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch", type=int, default=8)  # ~1 GB per sequence on a 65k vocab: the student's backward through its output layer
p.add_argument("--seq", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-5)  # 1e-4 wrecks 3-bit scales within 3 steps
a = p.parse_args()

# ponytail: teacher and student must share a tokenizer; cross-tokenizer needs token alignment (e.g. ULD loss)
teacher, tok = load(a.teacher)
student, _, config = load(a.student, return_config=True)

student.freeze()
if any(isinstance(m, nn.QuantizedLinear) for _, m in student.named_modules()):
    student.unfreeze(keys=["scales", "biases"])
else:
    student.unfreeze()  # ponytail: full fine-tune, switch to LoRA when the student is too big for fp32 Adam state
orig = student.trainable_parameters()
student.update(tree_map(lambda x: x.astype(mx.float32), orig))  # bf16 swallows small updates


def sample(model):
    # random first token, then free-running sampling (as in LLM-QAT): the model supplies its own data
    x = mx.random.randint(0, tok.vocab_size, (a.batch, 1))
    cache, out = make_prompt_cache(model), [x]
    for _ in range(a.seq - 1):
        x = mx.random.categorical(model(x, cache=cache)[:, -1])[:, None]
        mx.eval(x)
        out.append(x)
    return mx.concatenate(out, axis=1)


def kl(model, x, t_logp):
    s_logp = nn.log_softmax(model(x).astype(mx.float32), axis=-1)
    return (mx.exp(t_logp) * (t_logp - s_logp)).sum(-1).mean()


opt = optim.Adam(learning_rate=a.lr)
step = nn.value_and_grad(student, kl)
for i in range(a.steps):
    t0 = time.perf_counter()
    x = sample(student)  # on-policy: teacher only prefills, never decodes
    t_logp = nn.log_softmax(teacher(x).astype(mx.float32), axis=-1)
    loss, grads = step(student, x, t_logp)
    opt.update(student, grads)
    mx.eval(student.parameters(), opt.state, loss)
    dt = time.perf_counter() - t0
    print(f"step {i} kl {loss.item():.4f} {dt:.2f}s {a.batch * a.seq / dt:,.0f} tok/s", flush=True)
print(f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")

student.update(tree_map(lambda new, old: new.astype(old.dtype), student.trainable_parameters(), orig))
save(a.out, a.student, student, tok, config)
