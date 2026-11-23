// Train a population of tiny ternary LFM2-shaped students from scratch, trunks on the ANE, against bf16 LFM2-350M.
//   ane-train [--population 8] [--tokens 2048000] [--round 256000] [--accum 1] [--frozen-trits]
//
// The teacher writes the text (a from-scratch student can't yet) and scores it once for every member; each
// member learns the teacher's full next-token distribution (KL). All members run side by side in one ANE program
// per layer (population axis), start from the same initialisation and see the same data, and differ only in their
// hyperparameters: Adam rate, TTF theta, leak tau and a multiplier on the rho schedule. Every `round` tokens the
// members are scored on held-out sequences; the worse half copies the better half's full state and perturbs its
// hyperparameters (population-based training). With --population 1 it is a single run.
import ANE
import Foundation
import IOSurface
import LFM2ANE
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
setvbuf(stdout, nil, _IOLBF, 0)  // line-buffered, so a log survives fatalError

func option(_ name: String) -> String? { CommandLine.arguments.firstIndex(of: "--" + name).map { CommandLine.arguments[$0 + 1] } }
let T = 256, D = 256, F = 768, H = 4, hd = 64, V = 65536
let kinds = [false, false, true, false]  // attention layer?
let P = Int(option("population") ?? "8")!, accum = Int(option("accum") ?? "1")!
let tokenBudget = Int(option("tokens") ?? "2048000")!, roundTokens = Int(option("round") ?? "256000")!
let frozenTrits = CommandLine.arguments.contains("--frozen-trits")  // ablation: only the float parts learn
let updates = tokenBudget / (T * accum)
let shape = LFM2Shape(tokens: T, dim: D, ff: F, heads: H, headDim: hd, ropeBase: 1e6, population: P)
let beta2: Float = 0.95
let seed = Int(option("seed") ?? "0")!
let fixedScale = CommandLine.arguments.contains("--fixed-scale")  // never halve or double the loss scale
let prescale = CommandLine.arguments.contains("--grad-prescale")  // per-tensor power-of-two gradient scales inside the ANE backward
var factorWidth = [String: Int](), kExp = [String: [Int]](), gseen = [String: MLXArray](), dxSeen: Float = 0  // prescale state, below
let evalCount = Int(option("eval") ?? "8")!
MLXRandom.seed(UInt64(seed))  // init and training data follow --seed; the eval set is fixed (seed 424242, below)

// --jsonl PATH: one JSON object per line, flushed. Non-finite numbers become null.
let jsonl = option("jsonl").flatMap { path -> FileHandle? in
    if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
    let h = FileHandle(forWritingAtPath: path); _ = try? h?.seekToEnd(); return h
}
func num(_ x: Float) -> String { x.isFinite ? "\(x)" : "null" }
func nums(_ xs: [Float]) -> String { "[" + xs.map(num).joined(separator: ",") + "]" }
func emit(_ line: String) { try? jsonl?.write(contentsOf: Data((line + "\n").utf8)) }  // FileHandle writes are unbuffered

// Zero-copy self-test: CPU writes are visible through the MLX view, GPU writes land in our memory.
do {
    let probe = SharedBuffer([1, 1, 1, 256])
    probe.pointer.assumingMemoryBound(to: Float16.self)[3] = 7
    let seen = probe.array[3].item(Float.self)
    eval(probe.write(MLXArray.full([256], values: MLXArray(Float(2)))))
    let landed = probe.pointer.assumingMemoryBound(to: Float16.self)[5]
    guard seen == 7, landed == 2 else { fatalError("zero-copy self-test failed: view saw \(seen), GPU write gave \(landed)") }
}

// Searched hyperparameters, per member. Member 0 starts at the earlier hand-picked values.
struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state; z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9; z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
var rng = SplitMix(state: 1 &+ UInt64(seed))
struct Hyper { var lr: Float, theta: Float, tau: Float, rhoScale: Float }
let ranges: [(WritableKeyPath<Hyper, Float>, ClosedRange<Float>)] = [(\.lr, 3e-4 ... 2e-2), (\.theta, 2 ... 64), (\.tau, 50 ... 4000), (\.rhoScale, 0.1 ... 2)]
@MainActor func logUniform(_ r: ClosedRange<Float>) -> Float { exp(Float.random(in: log(r.lowerBound) ... log(r.upperBound), using: &rng)) }
var hyper = (0 ..< P).map { p -> Hyper in
    if p == 0 {
        guard let o = option("hyper") else { return Hyper(lr: 2e-3, theta: 16, tau: 1000, rhoScale: 1) }
        let v = o.split(separator: ",").map { Float($0)! }
        return Hyper(lr: v[0], theta: v[1], tau: v[2], rhoScale: v[3])  // --hyper lr,theta,tau,rho
    }
    var h = Hyper(lr: 0, theta: 0, tau: 0, rhoScale: 0)
    for (key, range) in ranges { h[keyPath: key] = logUniform(range) }
    return h
}
@MainActor func perturbed(_ h: Hyper) -> Hyper {
    var h = h
    for (key, range) in ranges { h[keyPath: key] = min(max(h[keyPath: key] * (Bool.random(using: &rng) ? 1.25 : 0.8), range.lowerBound), range.upperBound) }
    return h
}
@MainActor func perMember(_ value: (Hyper) -> Float) -> MLXArray { MLXArray(hyper.map(value)) }

/// Broadcasts a per-member [P] array against a [P, ...] parameter.
func members(_ a: MLXArray, like x: MLXArray) -> MLXArray { a.reshaped([P] + Array(repeating: 1, count: x.ndim - 1)) }

// Float parameter for all members: fp32 value [P, ...] + Adam state, optionally mirrored into an fp16 buffer the
// ANE reads (per-head copies for q/k norm gains).
final class Param {
    var value: MLXArray, m: MLXArray, v: MLXArray
    let buffer: SharedBuffer?, heads: Int
    init(_ value: MLXArray, buffer: SharedBuffer? = nil, heads: Int = 1) {
        self.value = value; m = zeros(like: value); v = zeros(like: value); self.buffer = buffer; self.heads = heads
    }
    func sync() -> MLXArray? {
        buffer?.write(heads > 1 ? repeated(value.expandedDimensions(axis: 1), count: heads, axis: 1) : value)
    }
    func update(_ grad: MLXArray, step t: Int, rate: MLXArray) -> MLXArray? {
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad * grad
        let mHat = m / (1 - pow(0.9, Float(t))), vHat = v / (1 - pow(0.999, Float(t)))
        value = value - members(rate, like: value) * mHat / (sqrt(vHat) + 1e-8)
        return sync()
    }
    func select(_ source: MLXArray) -> MLXArray? {  // slot q takes member source[q]
        value = value[source]; m = m[source]; v = v[source]; return sync()
    }
}

// Ternary [n, k] matrix for all members, trained by TTF: trits live only in the shared buffer.
final class Ternary {
    let name: String, n: Int, k: Int, trits: SharedBuffer, scale: Param
    var acc: MLXArray, rows: MLXArray, cols: MLXArray
    init(_ name: String, n: Int, k: Int) {
        self.name = name; self.n = n; self.k = k
        trits = SharedBuffer([1, P, n, k])
        scale = Param(MLXArray.full([P, n], values: MLXArray(1 / sqrt(Float(k) * 2 / 3))), buffer: SharedBuffer([1, P, 1, n]))
        acc = MLXArray.zeros([P * n * k]); rows = MLXArray.zeros([P, n]); cols = MLXArray.zeros([P, k])
        let one = MLXRandom.randInt(-1 ..< 2, [1, n * k]).asType(.float16)  // same initial trits for every member
        eval(trits.write(repeated(one, count: P, axis: 0)), scale.sync()!)
    }
    /// g: [P, n, k] dL/dtrits. Returns the number of trits moved per member.
    func update(g: MLXArray, rho: MLXArray, leak: MLXArray, theta: MLXArray, step t: Int) -> MLXArray {
        let g2 = g * g
        rows = beta2 * rows + (1 - beta2) * g2.mean(axis: 2); cols = beta2 * cols + (1 - beta2) * g2.mean(axis: 1)
        let correction = 1 - pow(beta2, Float(t)), r = rows / correction
        let params = stacked([rho, leak, theta, r.mean(axis: 1)], axis: 1).reshaped(-1)
        let out = ttfKernel([g.reshaped(-1), acc, r.reshaped(-1), (cols / correction).reshaped(-1), params, trits.array],
                            template: [("N", n), ("K", k)], grid: (P * n * k, 1, 1), threadGroup: (256, 1, 1),
                            outputShapes: [[P * n * k], [P * n * k]], outputDTypes: [.float32, .float32])
        acc = out[0]
        return out[1].reshaped(P, -1).sum(axis: 1)
    }
    func select(_ source: MLXArray) -> [MLXArray] {
        acc = acc.reshaped(P, -1)[source].reshaped(-1); rows = rows[source]; cols = cols[source]
        return [trits.write(trits.array.reshaped(P, -1)[source])] + [scale.select(source)].compactMap { $0 }
    }
}

struct Layer {
    let attention: Bool, ternaries: [Ternary], floats: [String: Param]
    var fwd: Model! = nil, bwd: Model! = nil
}

// Build parameters (identical across members).
let embedding = Param(repeated(MLXRandom.normal([1, V, D]) * 0.02, count: P, axis: 0)), finalNorm = Param(MLXArray.ones([P, D]))
let embedBuffer = SharedBuffer([1, P, T, D]), topGradient = SharedBuffer([1, P, T, D])
var layers: [Layer] = kinds.map { attention in
    let mixer: [(String, Int, Int)] = attention ? [("q", H * hd, D), ("k", H * hd, D), ("v", H * hd, D), ("o", D, H * hd)]
                                                : [("b", D, D), ("c", D, D), ("x", D, D), ("o", D, D)]
    let ternaries = (mixer + [("w1", F, D), ("w3", F, D), ("w2", D, F)]).map { Ternary($0.0, n: $0.1, k: $0.2) }
    var floats = ["g_op": Param(MLXArray.ones([P, D]), buffer: SharedBuffer([1, P, 1, D])),
                  "g_ffn": Param(MLXArray.ones([P, D]), buffer: SharedBuffer([1, P, 1, D]))]
    if attention {
        for n in ["g_q", "g_k"] { floats[n] = Param(MLXArray.ones([P, hd]), buffer: SharedBuffer([1, P * H, 1, hd]), heads: H) }
    } else {
        for j in 0 ..< 3 { floats["k\(j)"] = Param(MLXArray.full([P, D], values: MLXArray(Float(j == 2 ? 1 : 0))), buffer: SharedBuffer([1, P, 1, D])) }
    }
    eval(floats.values.compactMap { $0.sync() })
    return Layer(attention: attention, ternaries: ternaries, floats: floats)
}
let allParams: [Param] = [embedding, finalNorm] + layers.flatMap { l in l.ternaries.map(\.scale) + Array(l.floats.values) }

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
    let (program, constants) = try layerProgram(shape, attention: layers[i].attention, backward: backward, diagnostics: false, prescale: prescale)
    var bindings = ["x": x]
    if let gy { bindings["gy"] = gy }
    for t in layers[i].ternaries { bindings["w_\(t.name)"] = try t.trits.binding(); bindings["s_\(t.name)"] = try t.scale.buffer!.binding() }
    for (name, p) in layers[i].floats { bindings[name] = try p.buffer!.binding() }
    let model = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 4 << 30), inputBindings: bindings)
    for (name, t) in constants where name.hasPrefix("f_") { factorWidth[key(i, String(name.dropFirst(2)))] = t.shape.width }
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
print(String(format: "population %d: compiled %d ANE programs in %.1fs; %d ternary weights per member; %d updates of %d tokens",
             P, layers.count * 2, Date().timeIntervalSince(started), layers.flatMap(\.ternaries).map { $0.n * $0.k }.reduce(0, +), updates, T * accum))

// Teacher: writes the training text (random first token, temperature 1) and gives the targets.
let (teacher, _) = try await loadLLM("LiquidAI/LFM2-350M")
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
@MainActor func teacherLogProbs(_ tokens: MLXArray) -> MLXArray {
    let t = logSoftmax(teacher(tokens[.newAxis], cache: nil)[0].asType(.float32), axis: -1); eval(t); return t
}
MLXRandom.seed(424242)  // the held-out set is the same for every --seed
let evalSet = try generate(evalCount).map { MLXArray($0) }  // held out: never trained on
let evalTargets = evalSet.map(teacherLogProbs)
MLXRandom.seed(UInt64(seed))  // training data from here on
var pool: [[Int32]] = []

// Head and loss on the GPU for all members: logits = rmsnorm(h)·Eᵀ (tied), KL(teacher || member) per token.
func klPerMember(_ h: MLXArray, _ e: MLXArray, _ gain: MLXArray, _ target: MLXArray) -> MLXArray {
    let x = h * rsqrt((h * h).mean(axis: -1, keepDims: true) + 1e-5) * gain.expandedDimensions(axis: 1)
    let s = logSoftmax(matmul(x, e.transposed(0, 2, 1)), axis: -1)  // [P, T, V]
    return (exp(target) * (target - s)).sum(axis: -1).mean(axis: -1)  // [P]
}
let head = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in  // a = [h, E, final gain, teacher log-probs]
    let kl = klPerMember(a[0], a[1], a[2], a[3]); return [kl.sum(), kl]
}, argumentNumbers: [0, 1, 2])

/// Embedding gradient from the trunk: adds each member's dx rows [P, T, D] at the token ids, as [P, V, D].
func scatterRows(_ dx: MLXArray, _ tokens: MLXArray) -> MLXArray { MLXArray.zeros([P, V, D]).at[0..., tokens].add(dx) }

@MainActor func shared(_ model: Model, _ name: String) throws -> MLXArray { view(try output(model, name)).asType(.float32) }
/// Every backward output, top layer first: (layer, name, nonfinite count, max |finite value|).
@MainActor func overflowScan() -> [(Int, String, Int, Float)] {
    layers.indices.reversed().flatMap { i in
        ((try? layers[i].bwd.surfaceBindings()) ?? []).filter { $0.role == .output }.map { b -> (Int, String, Int, Float) in
            let a = view(b).asType(.float32), ok = (a - a) .== 0
            return (i, String(b.name.split(separator: "@")[0]), (a.size - ok.asType(.int32).sum().item(Int.self)), MLX.where(ok, abs(a), 0).max().item(Float.self))
        }
    }
}
@MainActor func forward(_ tokens: MLXArray) throws -> MLXArray {  // final hidden states [P, T, D]
    eval(embedBuffer.write(embedding.value.take(tokens, axis: 1)))
    for l in layers { _ = try l.fwd.evaluatePrepared(readOutputs: false, poisonOutputs: false) }
    return try shared(layers[layers.count - 1].fwd, "y").reshaped(P, T, D)
}
/// Mean held-out KL per member [P].
@MainActor func evaluate() throws -> [Float] {
    var total = MLXArray.zeros([P])
    for (tokens, target) in zip(evalSet, evalTargets) {
        total = total + klPerMember(try forward(tokens), embedding.value, finalNorm.value, target); eval(total)
    }
    return (total / Float(evalSet.count)).asArray(Float.self)
}
@MainActor func report(_ kl: [Float], tokens: Int) {
    print("eval at \(tokens) tokens (member: held-out kl, hyperparameters; noise/theta = rho x·sqrt(tau/2)/theta):")
    for p in kl.indices.sorted(by: { kl[$0] < kl[$1] }) {
        let h = hyper[p]
        emit("{\"type\":\"eval\",\"tokens\":\(tokens),\"member\":\(p),\"kl\":\(num(kl[p])),\"lr\":\(h.lr),\"theta\":\(h.theta),\"tau\":\(h.tau),\"rho\":\(h.rhoScale),\"scale\":\(lossScale)}")
        print(String(format: "  m%-2d %.3f  lr %.2e  theta %5.1f  tau %6.0f  rho x %.2f  noise/theta %.2f", p, kl[p], h.lr, h.theta, h.tau,
                     h.rhoScale, h.rhoScale * sqrt(h.tau / 2) / h.theta))
    }
}

/// Per member, whether every value in these [P, ...] arrays is finite.
@MainActor func finite(_ xs: [MLXArray]) -> [Bool] {
    let s = xs.map { $0.reshaped(P, -1).sum(axis: 1) }.reduce(MLXArray.zeros([P]), { $0 + $1 })
    return ((s - s) .== 0).asArray(Bool.self)
}
/// Slot q takes over member source[q]'s full state: weights, trits, optimiser and TTF state.
@MainActor func adopt(_ source: [Int]) {
    let index = MLXArray(source.map(Int32.init))
    var pending = allParams.compactMap { $0.select(index) } + layers.flatMap { $0.ternaries.flatMap { $0.select(index) } }
    pending += layers.flatMap { $0.ternaries.flatMap { [$0.acc, $0.rows, $0.cols] } } + allParams.flatMap { [$0.value, $0.m, $0.v] }
    eval(pending)
}
var ranking = Array(0 ..< P)  // best first, from the last eval
/// A member whose forward pass is no longer finite has diverged: it becomes a copy of the best healthy member.
@MainActor func replaceDiverged(_ alive: [Bool]) {
    guard let best = ranking.first(where: { alive[$0] }) else { fatalError("every member diverged") }
    adopt((0 ..< P).map { alive[$0] ? $0 : best })
    for p in 0 ..< P where !alive[p] { hyper[p] = perturbed(hyper[best]); print("  m\(p) diverged; now a copy of m\(best), perturbed") }
}

var lossScale: Float = Float(option("loss-scale") ?? "1024")!, goodSequences = 0, timing = [String: Double](), generation = 0.0
var gzero = [Float](repeating: 0, count: P), gmax = gzero, dxzero = gzero, gzmax = gzero
var tokensSeen = 0, nextRound = roundTokens, recent = MLXArray.zeros([P]), recentCount = 0
@MainActor func timed<R>(_ phase: String, _ body: () throws -> R) rethrows -> R {
    let t0 = Date(); defer { timing[phase, default: 0] += Date().timeIntervalSince(t0) }; return try body()
}
// --grad-prescale: output `name` of layer i's backward is multiplied by loss scale x 2^k[i.name][member] (an fp16
// runtime constant in the program, so k can change without a recompile); the trainer divides it out in fp32.
// k stays in [-14, 14] because 2^k must itself be an fp16 normal number.
func key(_ i: Int, _ name: String) -> String { "\(i).\(name)" }
@MainActor func pushFactors(_ i: Int) throws {
    let updates = try kExp.filter { $0.key.hasPrefix("\(i).") }.map { (k, e) -> (String, Tensor) in
        let name = String(k.dropFirst("\(i).".count)), copies = name == "dg_q" || name == "dg_k" ? H : 1
        let values = e.flatMap { [Float16](repeating: Float16(pow(2, Float($0))), count: copies * factorWidth[k]!) }
        return ("f_" + name, try Tensor(shape: TensorShape(batches: 1, channels: e.count * copies, height: 1, width: factorWidth[k]!), values: values))
    }
    try layers[i].bwd.update(inputs: Dictionary(uniqueKeysWithValues: updates))
}
/// Gradient output `name` of layer i as fp32 [P, ...] with the loss scale and its 2^k divided out; tracks its max.
@MainActor func gradient(_ i: Int, _ name: String) throws -> MLXArray {
    let raw = try shared(layers[i].bwd, name).reshaped(P, -1)
    guard prescale else { return raw / lossScale }
    let d = MLXArray((0 ..< P).map { lossScale * pow(2, Float(kExp[key(i, name)]![$0])) }).reshaped(P, 1)
    gseen[key(i, name)] = maximum(gseen[key(i, name)] ?? MLXArray.zeros([P]), (abs(raw).max(axis: 1, keepDims: true) / d).reshaped(P))
    return raw / d
}
/// Loss scale to `new`, with every k lowered by the same factor so the gradient outputs keep their size.
@MainActor func setLossScale(_ new: Float) throws {
    let shift = Int(log2(new / lossScale).rounded()); lossScale = new
    guard prescale, shift != 0 else { return }
    for k in kExp.keys { kExp[k] = kExp[k]!.map { min(14, max(-14, $0 - shift)) } }
    for i in layers.indices { try pushFactors(i) }
}
/// Every 50 updates: k so the largest observed gradient lands near 2^11; the loss scale likewise for dx0, within [256, 65536].
@MainActor func adaptPrescale() throws {
    if dxSeen > 0 { lossScale = min(65536, max(256, lossScale * pow(2, min(2, max(-2, log2(2048 / dxSeen).rounded()))))) }
    dxSeen = 0
    for (k, seen) in gseen {
        kExp[k] = zip(kExp[k]!, seen.asArray(Float.self)).map { old, g in g > 0 ? min(14, max(-14, Int(log2(2048 / (g * lossScale)).rounded()))) : old }
    }
    gseen = [:]
    for i in layers.indices { try pushFactors(i) }
}
if prescale {
    for (i, l) in layers.enumerated() {
        for t in l.ternaries { for n in ["gp_", "ds_"] { kExp[key(i, n + t.name)] = [Int](repeating: 0, count: P) } }
        for n in l.floats.keys { kExp[key(i, "d" + n)] = [Int](repeating: 0, count: P) }
    }
}
report(try evaluate(), tokens: 0)
let warmup = max(1, updates / 20), logEvery = max(1, 50 / accum)
for step in 1 ... updates {
    // Schedules: rho (warm up 0.25 -> 1, cosine to 0.2) times each member's multiplier; Adam rate likewise.
    let cosine = 0.5 * (1 + cos(Float.pi * max(0, Float(step - warmup)) / Float(max(1, updates - warmup))))
    let rhoBase = step < warmup ? 0.25 + 0.75 * Float(step) / Float(warmup) : 0.2 + 0.8 * cosine
    let rateBase = step < warmup ? Float(step) / Float(warmup) : 0.1 + 0.9 * cosine
    let rate = rateBase * perMember(\.lr), rho = rhoBase * perMember(\.rhoScale)
    let leak = 1 - 1 / perMember(\.tau), theta = perMember(\.theta)

    var sums = [ObjectIdentifier: MLXArray](), used = 0
    for _ in 0 ..< accum {
        if pool.isEmpty { let t0 = Date(); pool = try generate(256); generation += Date().timeIntervalSince(t0) }
        let tokens = MLXArray(pool.removeLast())
        tokensSeen += T
        let target = timed("teacher") { teacherLogProbs(tokens) }
        let h = try timed("forward") { try forward(tokens) }
        let alive = finite([h])
        if alive.contains(false) { replaceDiverged(alive); continue }  // ponytail: with --accum > 1 the slot keeps this update's earlier sums
        let (losses, headGrads) = timed("head") { () -> ([MLXArray], [MLXArray]) in
            let r = head([h, embedding.value, finalNorm.value, target])
            eval(r.0, r.1, topGradient.write(r.1[0] * lossScale)); return r
        }
        try timed("backward") { for l in layers.reversed() { _ = try l.bwd.evaluatePrepared(readOutputs: false, poisonOutputs: false) } }

        try timed("accumulate") {
            let dx0 = try shared(layers[0].bwd, "dx").reshaped(P, T, D)
            let gps = try layers.flatMap { l in try l.ternaries.map { try shared(l.bwd, "gp_\($0.name)") } }
            let dss = try layers.flatMap { l in try l.ternaries.map { try shared(l.bwd, "ds_\($0.name)") } }
            let dgs = try layers.flatMap { l in try l.floats.keys.map { try shared(l.bwd, "d" + $0) } }
            guard !finite((gps + dss + dgs).map { $0.reshaped(P, -1) } + [dx0]).contains(false) else {  // fp16 gradient overflow: skip, shrink
                let scan = overflowScan(), bad = scan.filter { $0.2 > 0 }.map { "L\($0.0).\($0.1)" }
                let finiteMax = scan.filter { $0.2 == 0 }.map { "L\($0.0).\($0.1)=\(String(format: "%.3g", $0.3))" }
                // The first layer (top down) with a nonfinite output is the culprit; below it, everything inherits its dx.
                let culprit = scan.filter { $0.2 > 0 }.map { $0.0 }.first.map { c in scan.filter { $0.0 == c && $0.2 > 0 }.map(\.1) } ?? []
                let top = scan.first { $0.2 > 0 }?.0 ?? 0
                for name in culprit where prescale && kExp[key(top, name)] != nil { kExp[key(top, name)] = kExp[key(top, name)]!.map { max(-14, $0 - 4) } }
                if !fixedScale, !prescale || culprit.contains(where: { kExp[key(top, $0)] == nil }) { try setLossScale(max(prescale ? 256 : 1, lossScale / 2)) }
                if prescale { for i in layers.indices { try pushFactors(i) } }
                goodSequences = 0
                print("update \(step): gradient overflow, loss scale -> \(lossScale)\n    bad: \(bad.joined(separator: " "))\n    max finite: \(finiteMax.joined(separator: " "))")
                emit("{\"type\":\"overflow\",\"update\":\(step),\"scale\":\(lossScale),\"bad\":[\(bad.map { "\"\($0)\"" }.joined(separator: ","))],\"max\":{"
                     + scan.filter { $0.2 == 0 }.map { "\"L\($0.0).\($0.1)\":\(num($0.3))" }.joined(separator: ",") + "}}")
                return
            }
            dxSeen = max(dxSeen, prescale ? abs(dx0).max().item(Float.self) : 0)
            if step % logEvery == 0 || step == 1 {  // gradient diagnostics per member, from the raw fp16 outputs
                let zeros = gps.map { ($0.reshaped(P, -1) .== 0).asType(.float32).sum(axis: 1) }.reduce(MLXArray.zeros([P]), +)
                gzero = (zeros / Float(gps.map { $0.size / P }.reduce(0, +))).asArray(Float.self)
                gzmax = gps.map { ($0.reshaped(P, -1) .== 0).asType(.float32).mean(axis: 1) }.reduce(MLXArray.zeros([P]), { maximum($0, $1) }).asArray(Float.self)
                gmax = gps.map { abs($0.reshaped(P, -1)).max(axis: 1) }.reduce(MLXArray.zeros([P]), { maximum($0, $1) }).asArray(Float.self)
                dxzero = (dx0.reshaped(P, -1) .== 0).asType(.float32).mean(axis: 1).asArray(Float.self)
            }
            var parts: [(AnyObject, MLXArray)] = [(embedding, headGrads[1] + scatterRows(dx0 / lossScale, tokens)), (finalNorm, headGrads[2])]
            for (i, l) in layers.enumerated() {
                for t in l.ternaries {
                    let gp = try gradient(i, "gp_\(t.name)").reshaped(P, T, t.n)
                    let a = try shared(l.bwd, ["w1": "a_hn", "w3": "a_hn", "w2": "a_act", "o": "a_mix"][t.name] ?? "a_xn").reshaped(P, T, t.k)
                    if !frozenTrits { parts.append((t, matmul(gp.transposed(0, 2, 1), a))) }  // dL/dtrits [P, n, k]
                    parts.append((t.scale, try gradient(i, "ds_\(t.name)").reshaped(P, t.n)))
                }
                for (name, p) in l.floats {
                    let g = try gradient(i, "d" + name)  // dg_op, dg_ffn, dg_q, dg_k, dk0..2
                    parts.append((p, p.heads > 1 ? g.reshaped(P, H, hd).sum(axis: 1) : g.reshaped(P, -1)))
                }
            }
            for (owner, g) in parts { sums[ObjectIdentifier(owner)] = sums[ObjectIdentifier(owner)].map { $0 + g } ?? g }
            recent = recent + losses[1]
            eval(Array(sums.values) + [recent])  // before the next sequence overwrites the ANE's output surfaces
            used += 1; recentCount += 1; goodSequences += 1
            if goodSequences % 200 == 0, lossScale < 65536, !fixedScale, !prescale { lossScale *= 2 }
        }
    }
    if prescale, !fixedScale, step == 1 || step % 50 == 0 { try adaptPrescale() }
    let flips = used == 0 ? MLXArray.zeros([P]) : timed("update") { () -> MLXArray in
        func mean(_ owner: AnyObject) -> MLXArray { sums[ObjectIdentifier(owner)]! / Float(used) }
        var pending = allParams.compactMap { p in p.update(mean(p), step: step, rate: rate) }
        var moved = MLXArray.zeros([P])
        if !frozenTrits {
            for t in layers.flatMap(\.ternaries) { moved = moved + t.update(g: mean(t), rho: rho, leak: leak, theta: theta, step: step) }
        }
        pending += [moved] + layers.flatMap { $0.ternaries.map(\.acc) } + allParams.map(\.value)
        eval(pending)
        return moved
    }
    if step % logEvery == 0 || step == 1 {
        let kl = (recent / Float(max(1, recentCount))).asArray(Float.self), f = flips.asArray(Float.self)
        print(String(format: "update %5d  tokens %8d  train kl best %.3f median %.3f  flips median %.0f  scale %.0f  %.0f tok/s per member",
                     step, tokensSeen, kl.min()!, kl.sorted()[P / 2], f.sorted()[P / 2], lossScale, Double(tokensSeen) / Date().timeIntervalSince(started)))
        print(String(format: "    gzero %.2f..%.2f (worst tensor %.2f)  gmax %.3g..%.3g  dxzero %.2f..%.2f", gzero.min()!, gzero.max()!, gzmax.max()!, gmax.min()!, gmax.max()!, dxzero.min()!, dxzero.max()!))
        emit("{\"type\":\"train\",\"update\":\(step),\"tokens\":\(tokensSeen),\"kl\":\(nums(kl)),\"flips\":\(nums(f)),\"scale\":\(lossScale),\"gzero\":\(nums(gzero)),\"gzmax\":\(nums(gzmax)),\"gmax\":\(nums(gmax)),\"dxzero\":\(nums(dxzero))}")
        recent = MLXArray.zeros([P]); recentCount = 0
    }
    if tokensSeen >= nextRound || step == updates {
        nextRound += roundTokens
        let kl = try evaluate().map { $0.isFinite ? $0 : .infinity }
        report(kl, tokens: tokensSeen)
        ranking = kl.indices.sorted { kl[$0] < kl[$1] }
        if P > 1, step < updates {  // population-based training: the worse half copies the better half, then explores
            let top = Array(ranking.prefix(P / 2))
            var source = Array(0 ..< P)
            for (i, slot) in ranking.suffix(P - P / 2).enumerated() { source[slot] = top[i % top.count] }
            adopt(source)
            for (slot, from) in source.enumerated() where slot != from { hyper[slot] = perturbed(hyper[from]); print("  m\(slot) <- m\(from), perturbed") }
        }
    }
}
let sequences = Double(tokensSeen / T), total = timing.values.reduce(0, +)
print("time per sequence (ms): " + timing.sorted { $0.value > $1.value }.map { String(format: "%@ %.1f", $0.key, 1000 * $0.value / sequences) }.joined(separator: ", ")
      + String(format: "; teacher text generation %.1f", 1000 * generation / sequences))
print(String(format: "throughput excluding text generation: %.0f tok/s per member, %.0f member-tok/s", Double(tokensSeen) / total, Double(tokensSeen * P) / total))
emit("{\"type\":\"done\",\"tokens\":\(tokensSeen),\"seconds\":\(Date().timeIntervalSince(started))}")
