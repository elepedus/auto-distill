# ANE single-token decode of the ternary LFM2-350M: what limits it, and 33 -> 208 tok/s (2026-11-25, M2 Max, idle machine)

Checkpoint `artifacts/lfm2-350m-ternary-ckpt` (no colscale). Everything below is measured with `ane-infer --decode-only` (greedy from BOS,
128 tokens, GPU head + argmax) or `ane-decode-probe`. Parity in every fused/constant variant is identical to the old decode:
teacher-forced 128 positions vs the GPU ternary model, logits rel-L2 0.0010, worst position 0.0057, argmax agreement 99.22%, KL 0.00002.

## Answer

ANE decode was limited by how weights reach the engine, not by dispatch or compute. Runtime-input fp16 weights stream at ~27 GB/s, and 576 MB per
token is 21 ms. Weights as BLOBFILE constants consumed by a 1x1 conv with a 2-bit palette (constexpr_lut_to_dense) are handled about 8x faster
per weight. With all 16 layers in one program: **208 tok/s end to end (4.8 ms/token; 263 tok/s for the ANE part alone), at identical parity**, versus 33 tok/s before,
and versus 306 tok/s for the bf16 GPU decode and 243 for the GPU ternary decode measured in the same run.
Single-stream >= 400 tok/s is not reachable with this design (floor ~3.6 ms of weight decode + ~1.2 ms host/GPU head, so ~210-260). Throughput >= 400 is reachable by batching
tokens (speculative/draft verification or several sequences), see the last section.

## 1. Dispatch floor and weight-read bandwidth (`ane-decode-probe floor`)

Trivial T=1 program `y = sum_i x[1,1,1,1024] @ w_i[1,1,N,1024]^T`, w_i fp16 runtime inputs, median of 200 `evaluatePrepared` (ms of `elapsed`).

| weights (fp16 runtime inputs) | median ms | GB/s |
|---|---|---|
| ~0 (N=16) | 0.126 | - |
| 1 MB | 0.121 | 8.7 |
| 4.2 MB | 0.292 | 14.4 |
| 16.8 MB (4 x 4.2) | 0.806 | 20.8 |
| 37.7 MB | 1.463 | 25.8 |
| 67.1 MB | 2.551 | 26.3 |
| 134 MB (8 x 16.8) | 4.748 | 28.3 |

Dispatch floor 0.12 ms; asymptotic runtime-input weight read ~27-28 GB/s (about 0.036 ms/MB). A layer's 36 MB therefore costs ~1.3 ms plus 0.12. Input count: 31 inputs ran, 32 and 33 also ran
here (timings noisier, ~16 GB/s); the earlier "Request cancelled" limit did not reproduce at 33 small inputs, so the fused programs simply use 8-11 inputs (packed).

## 2. Where the 1.75 ms/layer went (old per-layer decode, 30.6 ms/token, 33 tok/s)

- Dispatch: 16 x 0.12 = 1.9 ms (6%).
- Weight streaming: 576 MB / 27.5 GB/s = 21 ms (70%).
- The remainder of the 28.9 ms the 16 layers take (~6 ms) is compute and layout work of the T=1 graphs on runtime-input operands (not separated further).
- Host side: end-to-end 30.6 ms minus layers 28.9 = 1.7 ms for embed memcpy, 2 KB conv-state and K/V-row copies, GPU head, argmax and sync. The copies are memcpys of 2 KB per conv layer and
  16 x 128 B rows per K/V; they are negligible. No readback between programs (zero-copy chain).

## 3. Fusion (k layers per program, `ane-infer --fuse k --wmode ...`, `Sources/LFM2ANE/DecodeFused.swift`)

Weights packed into 3 runtime inputs per program (wA [1,k,13312,1024] = q|b,k|c,v|x,o,w1,w3 stacked by rows; wB = w2 [1,k,1024,4608]; sm = all scales, gains, conv taps as rows of [1,k,12,4608]);
slices are along channel (layer) and height with width offset 0 only (no >4094 width offsets); conv state and K/V caches are channel-packed buffers and outputs `ns`, `kn`, `vn` are concatenated.
"ms/token" is end to end (head on the GPU).

| weights | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| fp16 runtime inputs | 33 (old per-layer programs) | 40.0 | 41.9 | 43.1 | 43.5 |
| fp16 constants, matmul | 95.3 | - | 111.1 | - | - |
| fp16 constants, conv | - | - | 80.6 | - | 85.3 |
| 2-bit palette constants, matmul | 91.1 | - | 107.6 | - | - |
| **2-bit palette constants, conv** | 131.1 | 154.1 | 176.6 | 185.3 | **200.5 / 208.6** |

(tok/s. The last cell is two runs; 208.6 = ANE 3.8 ms + 1.0 ms host.) Fusion with runtime-input weights is capped at 43 tok/s: it only removes dispatch (~2 ms of 30) and 23 ms is still weight streaming.
Fusion matters once weights are constants: 131 -> 208 tok/s from k=1 to k=16.

## 4. BLOBFILE constants and palettisation

- Cause of "verifyBundleAtPath: invalid model": **more than one weight file**. One BLOBFILE constant works; two or more separate `weights/<name>.bin` files fail at load, with or without the
  coremltools buildInfo header, for any size (N=16 fails too), and in conv or matmul form. A **single file holding all blobs**, each chunk at its own 64-byte-aligned offset with its own 64-byte
  chunk header (file header: u32 chunk count at byte 0, u32 2 at byte 4; chunk: 0xdeadbeef, element type, byte size, absolute data offset), and `BLOBFILE(path=".../weights/weight.bin", offset=<chunk header offset>)`
  per constant, loads, for 288 constants in one program. This is a packaging issue on the caller side (swift-ane's `MILBlob.encode` writes one-chunk files, so the `MIL` builder and Studies emit one file per tensor and only one is ever loadable here).
  Workaround in this repo (`MILText.blob`, Graph.swift); suggested swift-ane change: a `MILBlob.encode(chunks:)` that writes several chunks into one file and returns their offsets, and `MIL.constant` appending to one shared `weights/weight.bin`.
- Probe (`ane-decode-probe blobs`, 16 x [2048,1024] = 67 MB fp16-equivalent, x of width 1, ms):

| weight form | conv | matmul (constexpr operand) |
|---|---|---|
| fp16 constant | 1.237 | 0.941 |
| int8 + per-row scale (constexpr_blockwise_shift_scale) | 0.703 | compile fails |
| 2-bit palette, one shared 4-entry LUT | **0.425** | 0.927 |
| 2-bit palette, per-row LUT | 1.561, and compile time grows to 7.7 s | - |
| fp16 runtime inputs (section 1) | 2.551 | 2.551 |

  Palettised weights only take the fast path through **conv** (matmul dequantises to fp16 first and loses all of the gain). The ternary encoding is indices {0,1,2} = trit + 1 through one LUT {-1,0,1,0};
  the per-row scale is applied after, as before, so the arithmetic (and parity) is unchanged. A per-row LUT is the wrong place for the scale.
- The fused conv decode computes `x [1,1,1,K] -> reshape [1,K,1,1] -> conv(w [N,K,1,1]) -> reshape [1,1,1,N]`. In the real network palette conv gives 4.5 ms for the 16 layers at k=4 and 3.8 ms at k=16,
  consistent with the probe (576 MB fp16-equivalent at 0.425 ms per 67 MB = 3.65 ms): the layer is now bound by the engine's weight-decode rate (~80 G weights/s), not by DRAM. The packed 2-bit data is 36 MB per token in total.

## 5. Why not 400 tok/s, and what would be

- Single token stream: ANE floor 3.65 ms (weights) + ~0.2 ms (rest) + 0.12 ms dispatch, plus 1.0-1.2 ms host and GPU head. Best measured 208.6 tok/s. The gap to GPU decode (306) is the weight-decode rate; the ANE is not quicker than the GPU at batch 1 here.
- Probe of the same 67 MB palette conv with x of width W (tokens per call): W = 1, 2, 4, 8, 16, 32, 64 -> 0.414, 0.431, 0.448, 0.440, 0.457, 0.509, 0.466 ms. **Cost per call is independent of the number of tokens up to at least 64.**
  So 8-16 tokens per call cost the same ~4 ms per forward: speculative decoding (draft + verify of 4-8 tokens, accepted ~2-3 on average) or decoding several sequences at once would give ~2.5-8x the
  single-stream rate on the ANE (extrapolated from the probe, not measured: 4 sequences at ~4.5 ms per step is ~800 tok/s aggregate). That requires T>1 programs with constants (attention with a K/V cache and T new rows, not built here).
- Remaining single-stream levers, small: overlap the GPU head with the next group (needs a draft/other token), shrink the 1 ms host side (head 134 MB bf16 on the GPU ~0.5 ms), trim the non-matmul ops.

## Reproduce

```
swift build -c release --product ane-infer --product ane-decode-probe
.build/out/Products/Release/ane-decode-probe floor
SHARED=1 .build/out/Products/Release/ane-decode-probe blobs pal2t        # also fp16 | int8 | pal2; MM=1 matmul form; W=<tokens>
.build/out/Products/Release/ane-infer --ckpt DIR --decode-only --fuse 16 --wmode pal2conv   # wmode: input | fp16 | pal2 | fp16conv | pal2conv; --no-gpu-decode skips the GPU baselines
```
Code: `Sources/LFM2ANE/DecodeFused.swift` (new; `decodeProgram` untouched), additions to `Sources/LFM2ANE/Graph.swift` (blob, concat, conv1x1, lutToDense; existing emitted text unchanged),
`Sources/ane-decode-probe`, `--fuse/--wmode/--decode-only/--no-gpu-decode` in `Sources/ane-infer/main.swift`. Fused decode does not support colscale (precondition); the checkpoint has none.
Not done: handoff from prefill state to fused decode; the prefill programs still use runtime-input weights (palette constants would likely help them too).
