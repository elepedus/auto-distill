// Stage 4: train a tiny ternary LFM2-shaped model from scratch, trunk on the ANE, against bf16 LFM2-350M.
//   ane-train [--steps 2000] [--accum 1] [--frozen-trits]
// The teacher writes the text (a from-scratch student can't yet) and scores it; the student learns the
// teacher's full next-token distribution (KL). Per step: embedding on the GPU, 4 layers forward on the ANE,
// head + loss on the GPU, 4 layers backward on the ANE, then TTF updates for the trits and Adam for the
// float parameters, written straight into the buffers the ANE programs read.
import ANE
import Foundation
import IOSurface
import LFM2ANE
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

let T = 256, D = 256, F = 768, H = 4, hd = 64, V = 65536
let kinds = [false, false, true, false]  // attention layer?
let steps = CommandLine.arguments.firstIndex(of: "--steps").map { Int(CommandLine.arguments[$0 + 1])! } ?? 2000
let frozenTrits = CommandLine.arguments.contains("--frozen-trits")  // ablation: only the float parts learn
let accum = CommandLine.arguments.firstIndex(of: "--accum").map { Int(CommandLine.arguments[$0 + 1])! } ?? 1  // sequences per update
let shape = LFM2Shape(tokens: T, dim: D, ff: F, heads: H, headDim: hd, ropeBase: 1e6)
MLXRandom.seed(0)  // same data, init and eval set across runs, so ablations compare like for like
let theta: Float = 16, leak: Float = 1 - 1.0 / 1000, beta2: Float = 0.95, lr: Float = 2e-3

// Zero-copy self-test: CPU writes are visible through the MLX view, GPU writes land in our memory.
do {
    let probe = SharedBuffer([1, 1, 1, 256])
    probe.pointer.assumingMemoryBound(to: Float16.self)[3] = 7
    let seen = probe.array[3].item(Float.self)
    eval(probe.write(MLXArray.full([256], values: MLXArray(Float(2)))))
    let landed = probe.pointer.assumingMemoryBound(to: Float16.self)[5]
    guard seen == 7, landed == 2 else { fatalError("zero-copy self-test failed: view saw \(seen), GPU write gave \(landed)") }
}

// Float parameter: fp32 master + Adam state, mirrored into an fp16 buffer the ANE reads (tiled per head if needed).
final class Param {
    var value: MLXArray, m: MLXArray, v: MLXArray
    let buffer: SharedBuffer?, heads: Int
    init(_ value: MLXArray, buffer: SharedBuffer? = nil, heads: Int = 1) {
        self.value = value; m = zeros(like: value); v = zeros(like: value); self.buffer = buffer; self.heads = heads
    }
    func sync() -> MLXArray? { buffer?.write(heads > 1 ? tiled(value, repetitions: [heads]) : value) }
    func update(_ grad: MLXArray, step t: Int, rate: Float) -> MLXArray? {
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad * grad
        let mHat = m / (1 - pow(0.9, Float(t))), vHat = v / (1 - pow(0.999, Float(t)))
        value = value - rate * mHat / (sqrt(vHat) + 1e-8)
        return sync()
    }
}

// Ternary matrix trained by TTF: trits live only in the shared buffer; per-row scale is a float parameter.
final class Ternary {
    let name: String, n: Int, k: Int, trits: SharedBuffer, scale: Param
    var acc: MLXArray, rows: MLXArray, cols: MLXArray
    init(_ name: String, n: Int, k: Int) {
        self.name = name; self.n = n; self.k = k
        trits = SharedBuffer([1, 1, n, k])
        scale = Param(MLXArray.full([n], values: MLXArray(1 / sqrt(Float(k) * 2 / 3))), buffer: SharedBuffer([1, 1, 1, n]))
        acc = MLXArray.zeros([n * k]); rows = MLXArray.zeros([n]); cols = MLXArray.zeros([k])
        eval(trits.write(MLXRandom.randInt(-1 ..< 2, [n * k]).asType(.float16)), scale.sync()!)
    }
    /// g: [n, k] dL/dtrits (gpᵀ·a, averaged over the update's sequences). Returns the number of trits moved.
    func update(g: MLXArray, rho: Float, step t: Int) -> MLXArray {
        let g2 = g * g
        rows = beta2 * rows + (1 - beta2) * g2.mean(axis: 1); cols = beta2 * cols + (1 - beta2) * g2.mean(axis: 0)
        let correction = 1 - pow(beta2, Float(t)), r = rows / correction
        let params = concatenated([MLXArray([rho, leak, theta]), r.mean().reshaped(1)])
        let out = ttfKernel([g.reshaped(-1), acc, r, cols / correction, params, trits.array],
                            grid: (n * k, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[n * k], [n * k]], outputDTypes: [.float32, .float32])
        acc = out[0]
        return out[1].sum()
    }
}

struct Layer {
    let attention: Bool, ternaries: [Ternary], floats: [String: Param]
    var fwd: Model! = nil, bwd: Model! = nil
}

// Build parameters.
let embedding = Param(MLXRandom.normal([V, D]) * 0.02), finalNorm = Param(MLXArray.ones([D]))
let embedBuffer = SharedBuffer([1, 1, T, D]), topGradient = SharedBuffer([1, 1, T, D])
var layers: [Layer] = kinds.map { attention in
    let mixer: [(String, Int, Int)] = attention ? [("q", H * hd, D), ("k", H * hd, D), ("v", H * hd, D), ("o", D, H * hd)]
                                                : [("b", D, D), ("c", D, D), ("x", D, D), ("o", D, D)]
    let ternaries = (mixer + [("w1", F, D), ("w3", F, D), ("w2", D, F)]).map { Ternary($0.0, n: $0.1, k: $0.2) }
    var floats = ["g_op": Param(MLXArray.ones([D]), buffer: SharedBuffer([1, 1, 1, D])),
                  "g_ffn": Param(MLXArray.ones([D]), buffer: SharedBuffer([1, 1, 1, D]))]
    if attention {
        for n in ["g_q", "g_k"] { floats[n] = Param(MLXArray.ones([hd]), buffer: SharedBuffer([1, H, 1, hd]), heads: H) }
    } else {
        for j in 0 ..< 3 { floats["k\(j)"] = Param(MLXArray.full([D], values: MLXArray(Float(j == 2 ? 1 : 0))), buffer: SharedBuffer([1, 1, 1, D])) }
    }
    eval(floats.values.compactMap { $0.sync() })
    return Layer(attention: attention, ternaries: ternaries, floats: floats)
}

// Compile and wire the ANE programs. Weights bind to our buffers; activations chain surface to surface.
@MainActor func output(_ model: Model, _ name: String) throws -> SurfaceBinding {
    try model.surfaceBindings().first { $0.role == .output && ($0.name == name || $0.name.hasPrefix(name + "@")) }!
}
@MainActor func link(_ producer: Model, _ name: String) throws -> ExternalInputBinding {
    let s = try output(producer, name)
    return try ExternalInputBinding(surface: s.surface, byteRange: 0 ..< IOSurfaceGetAllocSize(s.surface),
                                    layout: .compilerLayout(s.layout), retaining: producer, access: .readOnly)
}
@MainActor func compile(_ i: Int, backward: Bool, x: ExternalInputBinding, gy: ExternalInputBinding?) throws -> Model {
    let (program, constants) = try layerProgram(shape, attention: layers[i].attention, backward: backward, diagnostics: false)
    var bindings = ["x": x]
    if let gy { bindings["gy"] = gy }
    for t in layers[i].ternaries { bindings["w_\(t.name)"] = try t.trits.binding(); bindings["s_\(t.name)"] = try t.scale.buffer!.binding() }
    for (name, p) in layers[i].floats { bindings[name] = try p.buffer!.binding() }
    let model = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 2 << 30), inputBindings: bindings)
    _ = try model.prepareUpdating(inputs: constants)
    return model
}
let started = Date()
for i in layers.indices {
    layers[i].fwd = try compile(i, backward: false, x: i == 0 ? try embedBuffer.binding() : try link(layers[i - 1].fwd, "y"), gy: nil)
}
for i in layers.indices.reversed() {
    let x = i == 0 ? try embedBuffer.binding() : try link(layers[i - 1].fwd, "y")
    layers[i].bwd = try compile(i, backward: true, x: x, gy: i == layers.count - 1 ? try topGradient.binding() : try link(layers[i + 1].bwd, "dx"))
}
print(String(format: "compiled %d ANE programs in %.1fs; %d ternary weights", layers.count * 2, Date().timeIntervalSince(started),
             layers.flatMap(\.ternaries).map { $0.n * $0.k }.reduce(0, +)))

// Teacher: writes the training text (random first token, temperature 1) and gives the targets.
let (teacher, _) = try await loadLLM("LiquidAI/LFM2-350M")
var pool: [[Int32]] = []
@MainActor func generate(_ count: Int) throws -> [[Int32]] {
    var x = MLXRandom.randInt(0 ..< V, [count, 1]).asType(.int32), rows = [x]
    let cache = try teacher.newCache(parameters: nil)
    for _ in 1 ..< T {
        x = categorical(teacher(x, cache: cache)[0..., -1, 0...])[0..., .newAxis].asType(.int32)
        eval(x); rows.append(x)
    }
    let all = concatenated(rows, axis: 1)
    return (0 ..< count).map { all[$0].asArray(Int32.self) }
}
let evalSet = try generate(8)  // held out: never trained on

// Head and loss on the GPU: logits = rmsnorm(h)·Eᵀ (tied), loss = KL(teacher || student) per token.
let head = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in  // a = [h, E, final norm gain, teacher log-probs]
    let s = logSoftmax(matmul(MLXFast.rmsNorm(a[0], weight: a[2], eps: 1e-5), a[1].T), axis: -1)
    return [(exp(a[3]) * (a[3] - s)).sum(axis: -1).mean()]
}, argumentNumbers: [0, 1, 2])
var teacherLogp = MLXArray(0)

@MainActor func shared(_ model: Model, _ name: String) throws -> MLXArray { view(try output(model, name)).asType(.float32) }

/// Mean KL(teacher || student) per token over the held-out sequences, forward pass on the ANE.
@MainActor func evaluate() throws -> Float {
    var total: Float = 0
    for sequence in evalSet {
        let tokens = MLXArray(sequence)
        let t = logSoftmax(teacher(tokens[.newAxis], cache: nil)[0].asType(.float32), axis: -1)
        eval(embedBuffer.write(embedding.value[tokens]))
        for l in layers { _ = try l.fwd.evaluatePrepared(readOutputs: false, poisonOutputs: false) }
        let h = try shared(layers[layers.count - 1].fwd, "y").reshaped(T, D)
        let s = logSoftmax(matmul(MLXFast.rmsNorm(h, weight: finalNorm.value, eps: 1e-5), embedding.value.T), axis: -1)
        total += (exp(t) * (t - s)).sum(axis: -1).mean().item(Float.self)
    }
    return total / Float(evalSet.count)
}
print(String(format: "eval kl %.3f at 0 tokens (accum %d%@)", try evaluate(), accum, frozenTrits ? ", trits frozen" : ""))
var lossScale: Float = 1024, goodSequences = 0, recent: [Float] = [], timing = [String: Double](), generation = 0.0
var tokensSeen = 0, nextEval = 64000  // held-out eval every 64k tokens, comparable across --accum
@MainActor func timed<R>(_ phase: String, _ body: () throws -> R) rethrows -> R {
    let t0 = Date(); defer { timing[phase, default: 0] += Date().timeIntervalSince(t0) }; return try body()
}
let warmup = max(1, steps / 20), logEvery = max(1, 25 / accum)
for step in 1 ... steps {
    // Schedules: TTF step size rho (warm up 0.25 -> 1, cosine to 0.2) and Adam rate (warm up, cosine to 10%).
    let cosine = 0.5 * (1 + cos(Float.pi * max(0, Float(step - warmup)) / Float(max(1, steps - warmup))))
    let rho = step < warmup ? 0.25 + 0.75 * Float(step) / Float(warmup) : 0.2 + 0.8 * cosine
    let rate = lr * (step < warmup ? Float(step) / Float(warmup) : 0.1 + 0.9 * cosine)

    // Sum gradients over `accum` sequences in fp32 on the GPU, keyed by the parameter they belong to.
    var sums = [ObjectIdentifier: MLXArray](), used = 0
    for _ in 0 ..< accum {
        if pool.isEmpty { let t0 = Date(); pool = try generate(256); generation += Date().timeIntervalSince(t0) }
        let tokens = MLXArray(pool.removeLast())
        tokensSeen += T
        timed("teacher") { teacherLogp = logSoftmax(teacher(tokens[.newAxis], cache: nil)[0].asType(.float32), axis: -1); eval(teacherLogp) }
        timed("embed") { eval(embedBuffer.write(embedding.value[tokens])) }
        try timed("ane forward") { for l in layers { _ = try l.fwd.evaluatePrepared(readOutputs: false, poisonOutputs: false) } }
        let (loss, headGrads) = try timed("head") { () throws -> ([MLXArray], [MLXArray]) in
            let r = head([try shared(layers[layers.count - 1].fwd, "y").reshaped(T, D), embedding.value, finalNorm.value, teacherLogp])
            eval(r.0, r.1, topGradient.write(r.1[0] * lossScale))
            return r
        }
        try timed("ane backward") { for l in layers.reversed() { _ = try l.bwd.evaluatePrepared(readOutputs: false, poisonOutputs: false) } }

        try timed("accumulate") {
            let dx0 = try shared(layers[0].bwd, "dx").reshaped(T, D)
            let gps = try layers.flatMap { l in try l.ternaries.map { try shared(l.bwd, "gp_\($0.name)") } }
            guard gps.map({ $0.sum() }).reduce(dx0.sum(), { $0 + $1 }).item(Float.self).isFinite else {  // fp16 overflow: skip, shrink
                lossScale /= 2; goodSequences = 0; print("update \(step): gradient overflow, loss scale -> \(lossScale)"); return
            }
            var parts: [(AnyObject, MLXArray)] = [(embedding, headGrads[1].at[tokens].add(dx0 / lossScale)), (finalNorm, headGrads[2])]
            for l in layers {
                for t in l.ternaries {
                    let gp = try shared(l.bwd, "gp_\(t.name)").reshaped(T, t.n) / lossScale
                    let a = try shared(l.bwd, ["w1": "a_hn", "w3": "a_hn", "w2": "a_act", "o": "a_mix"][t.name] ?? "a_xn").reshaped(T, t.k)
                    if !frozenTrits { parts.append((t, matmul(gp.T, a))) }  // dL/dtrits
                    parts.append((t.scale, try shared(l.bwd, "ds_\(t.name)").reshaped(-1) / lossScale))
                }
                for (name, p) in l.floats {
                    let g = try shared(l.bwd, "d" + name) / lossScale  // dg_op, dg_ffn, dg_q, dg_k, dk0..2
                    parts.append((p, p.heads > 1 ? g.reshaped(H, hd).sum(axis: 0) : g.reshaped(-1)))
                }
            }
            for (owner, g) in parts { sums[ObjectIdentifier(owner)] = sums[ObjectIdentifier(owner)].map { $0 + g } ?? g }
            eval(Array(sums.values))  // before the next sequence overwrites the ANE's output surfaces
            recent.append(loss[0].item(Float.self)); used += 1; goodSequences += 1
            if goodSequences % 200 == 0, lossScale < 65536 { lossScale *= 2 }
        }
    }
    guard used > 0 else { continue }

    try timed("update") {
        func mean(_ owner: AnyObject) -> MLXArray { sums[ObjectIdentifier(owner)]! / Float(used) }
        var pending = [embedding.update(mean(embedding), step: step, rate: rate), finalNorm.update(mean(finalNorm), step: step, rate: rate)].compactMap { $0 }
        var moved = [MLXArray]()
        for l in layers {
            for t in l.ternaries {
                if !frozenTrits { moved.append(t.update(g: mean(t), rho: rho, step: step)) }
                pending += [t.scale.update(mean(t.scale), step: step, rate: rate)].compactMap { $0 }
            }
            for p in l.floats.values { pending += [p.update(mean(p), step: step, rate: rate)].compactMap { $0 } }
        }
        let flips = moved.reduce(MLXArray(Float(0)), { $0 + $1 })
        eval(pending + [flips] + layers.flatMap { $0.ternaries.map(\.acc) })
        if step % logEvery == 0 || step == 1 {
            print(String(format: "update %4d  tokens %8d  kl %.3f  flips %6.0f  rho %.2f  scale %.0f  %.0f tok/s", step, tokensSeen,
                         recent.reduce(0, +) / Float(max(1, recent.count)), flips.item(Float.self), rho, lossScale,
                         Double(tokensSeen) / Date().timeIntervalSince(started)))
            recent.removeAll()
        }
    }
    if tokensSeen >= nextEval { print(String(format: "eval kl %.3f at %d tokens", try evaluate(), tokensSeen)); nextEval += 64000 }
}
let total = timing.values.reduce(0, +)
let sequences = Double(tokensSeen / T)
print("time per sequence (ms): " + timing.sorted { $0.value > $1.value }.map { String(format: "%@ %.1f", $0.key, 1000 * $0.value / sequences) }.joined(separator: ", ")
      + String(format: "; teacher text generation %.1f", 1000 * generation / sequences))
print(String(format: "training throughput excluding text generation: %.0f tok/s", Double(tokensSeen) / total))
