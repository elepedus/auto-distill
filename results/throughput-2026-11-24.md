# Throughput, one job at a time (2026-11-24 05:45 GMT, M2 Max, 256k tokens each)

Queue: `tp-ane.queue` (ane-train --bound, P=1 and P=8) then `tp-gpu.queue` (float-train --ternary and fp32). Raw lines:

## tp-ane-p1
```
time per sequence (ms): teacher 24.6, update 11.0, head 7.9, accumulate 6.9, backward 6.3, forward 3.6; teacher text generation 30.5
throughput excluding text generation: 4239 tok/s per member, 4239 member-tok/s
```
## tp-ane-p8
```
time per sequence (ms): update 56.4, head 46.9, backward 25.9, teacher 24.8, accumulate 14.1, forward 10.7; teacher text generation 31.6
throughput excluding text generation: 1432 tok/s per member, 11455 member-tok/s
```
## tp-gpu-tern
```
teacher text generation 30.4 ms per sequence
throughput excluding text generation: 5500 tok/s per member, 5500 member-tok/s
```
## tp-gpu-fp32
```
teacher text generation 30.5 ms per sequence
throughput excluding text generation: 5740 tok/s per member, 5740 member-tok/s
```
