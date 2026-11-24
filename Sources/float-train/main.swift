// Float GPU baseline for ane-train: the same LFM2-shaped student (d 256, conv/conv/attn/conv, tied head), plain fp32
// weights, Adam on everything, MLX autodiff, same teacher-generated data and held-out KL eval.
//   float-train [--tokens 2048000] [--round 256000] [--lr 1e-3] [--seed 0] [--eval 8] [--jsonl PATH] [--schedule]
//               [--ternary [--hyper lr,theta,tau,rho]] [--ternary-ste]
// --ternary: every linear ane-train makes ternary is trits x per-row scale (fp32 autodiff gives dL/dtrit, trits trained by the
// same TTF kernel as ane-train, scales/embedding/norms/taps by Adam); schedules are ane-train's. --schedule: ane-train's Adam
// warm-up + cosine on plain float too. --ternary-ste: the same linears keep a latent fp32 weight W (float init); the forward uses
// s * round(clip(W/s, -1, 1)) with a straight-through gradient, Adam on everything (--lr is the peak, ane-train's schedule).
import Foundation
import LFM2ANE
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

func option(_ name: String) -> String? { CommandLine.arguments.firstIndex(of: "--" + name).map { CommandLine.arguments[$0 + 1] } }
let T = 256, D = 256, F = 768, H = 4, hd = 64, V = 65536
let kinds = [false, false, true, false]  // attention layer?
let tokenBudget = Int(option("tokens") ?? "2048000")!, roundTokens = Int(option("round") ?? "256000")!
let ternary = CommandLine.arguments.contains("--ternary"), ste = CommandLine.arguments.contains("--ternary-ste"), scheduled = ternary || ste || CommandLine.arguments.contains("--schedule")
let hyp = (option("hyper") ?? "2e-3,16,1000,1").split(separator: ",").map { Float($0)! }  // lr, theta, tau, rho
let lr = ternary ? hyp[0] : Float(option("lr") ?? "1e-3")!, theta = hyp[1], tau = hyp[2], rhoScale = hyp[3], beta2: Float = 0.95
let seed = Int(option("seed") ?? "0")!, evalCount = Int(option("eval") ?? "8")!
let updates = tokenBudget / T
// On-policy data: lambda = fraction of each pool refill that the student writes (teacher scores it).
func pair(_ name: String) -> (Float, Float)? { option(name).map { let a = $0.split(separator: ",").map { Float($0)! }; return (a[0], a[1]) } }
let opConst = Float(option("onpolicy-const") ?? "0")!, opFrom = Float(option("onpolicy-from") ?? "0")!, opRamp = pair("onpolicy-ramp"), opAlpha = pair("onpolicy-alpha")

let jsonl = option("jsonl").flatMap { path -> FileHandle? in
    if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
    let h = FileHandle(forWritingAtPath: path); _ = try? h?.seekToEnd(); return h
}
func num(_ x: Float) -> String { x.isFinite ? "\(x)" : "null" }
func emit(_ line: String) { try? jsonl?.write(contentsOf: Data((line + "\n").utf8)) }

// Parameters, flat: [embed, final gain] then per layer
//   conv:  g_op g_ffn b c x o k0 k1 k2 w1 w3 w2     attn: g_op g_ffn q k v o g_q g_k w1 w3 w2
let paramCount = 49
/// Indices in the flat parameter list of the linears that are ternary in ane-train.
func ternaryIndices() -> [Int] {
    var out = [Int](), i = 2
    for attention in [false, false, true, false] {
        let ffn: Int = i + (attention ? 8 : 9)
        out += [i + 2, i + 3, i + 4, i + 5, ffn, ffn + 1, ffn + 2]
        i = ffn + 3
    }
    return out
}
/// Replaces each ternary linear's per-row scale [n] by trits [n, k] x scale.
func compose(_ p: [MLXArray], _ trits: [MLXArray]) -> [MLXArray] {
    var p = p
    for (j, i) in ternaryIndices().enumerated() { p[i] = trits[j] * p[i].expandedDimensions(axis: 1) }
    return p
}
/// BitNet-style STE: W_q = s * round(clip(W/s, -1, 1)), forward value W_q, gradient straight to W. The per-row scale s is
/// derived from W (absmean over the row, as BitNet b1.58 but per row), not a separate parameter.
func steQuantize(_ p: [MLXArray]) -> [MLXArray] {
    var p = p
    for i in ternaryIndices() {
        let w = p[i], s = abs(w).mean(axis: 1, keepDims: true) + 1e-5
        p[i] = w + stopGradient(s * round(clip(w / s, min: -1, max: 1)) - w)
    }
    return p
}
/// Fraction of zero trits over all quantised linears.
func zeroFraction(_ p: [MLXArray]) -> Float {
    var z = MLXArray(Float(0)), n = 0
    for i in ternaryIndices() {
        let w = p[i], s = abs(w).mean(axis: 1, keepDims: true) + 1e-5
        z = z + (round(clip(w / s, min: -1, max: 1)) .== 0).asType(.float32).sum(); n += w.size
    }
    return z.item(Float.self) / Float(n)
}
func initParams(ternary: Bool) -> [MLXArray] {
    func lin(_ n: Int, _ k: Int) -> MLXArray {  // ane-train: trits x scale, scale 1/sqrt(2k/3); float baseline: std 1/sqrt(k)
        ternary ? MLXArray.full([n], values: MLXArray(1 / sqrt(Float(k) * 2 / 3))) : MLXRandom.normal([n, k]) / sqrt(Float(k))
    }
    var p = [MLXRandom.normal([65536, 256]) * 0.02, MLXArray.ones([256])]
    for attention in [false, false, true, false] {
        p += [MLXArray.ones([256]), MLXArray.ones([256])]
        if attention {
            p += [lin(256, 256), lin(256, 256), lin(256, 256), lin(256, 256), MLXArray.ones([64]), MLXArray.ones([64])]
        } else {
            p += [lin(256, 256), lin(256, 256), lin(256, 256), lin(256, 256), MLXArray.zeros([256]), MLXArray.zeros([256]), MLXArray.ones([256])]
        }
        p += [lin(768, 256), lin(768, 256), lin(256, 768)]
    }
    return p
}

/// Logits [T, V] for one sequence of token ids [T]. Constants are local so valueAndGrad closures capture nothing.
func logits(_ p: [MLXArray], _ tokens: MLXArray) -> MLXArray {
    let T = 256, H = 4, hd = 64, eps: Float = 1e-5
    func lin(_ x: MLXArray, _ w: MLXArray) -> MLXArray { matmul(x, w.T) }
    var h = p[0][tokens], i = 2
    for attention in [false, false, true, false] {
        let xn = MLXFast.rmsNorm(h, weight: p[i], eps: eps), ffnGain = p[i + 1]
        let r: MLXArray
        if attention {
            func heads(_ z: MLXArray, _ gain: MLXArray?) -> MLXArray {
                var z = z.reshaped(1, T, H, hd)
                if let gain { z = MLXFast.rmsNorm(z, weight: gain, eps: eps) }
                return z.transposed(0, 2, 1, 3)
            }
            func rope(_ z: MLXArray) -> MLXArray { MLXFast.RoPE(z, dimensions: hd, traditional: false, base: 1e6, scale: 1, offset: 0) }
            let q = rope(heads(lin(xn, p[i + 2]), p[i + 6])), k = rope(heads(lin(xn, p[i + 3]), p[i + 7])), v = heads(lin(xn, p[i + 4]), nil)
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.125, mask: .causal)
            r = lin(o.transposed(0, 2, 1, 3).reshaped(T, H * hd), p[i + 5])
            i += 8
        } else {
            let bx = lin(xn, p[i + 2]) * lin(xn, p[i + 4])
            let padded = concatenated([MLXArray.zeros([2, 256]), bx], axis: 0)  // causal 3-tap conv: taps k0, k1, k2 on t-2, t-1, t
            let conv = padded[0 ..< T] * p[i + 6] + padded[1 ..< T + 1] * p[i + 7] + padded[2 ..< T + 2] * p[i + 8]
            r = lin(lin(xn, p[i + 3]) * conv, p[i + 5])
            i += 9
        }
        h = h + r
        let hn = MLXFast.rmsNorm(h, weight: ffnGain, eps: eps)
        h = h + lin(silu(lin(hn, p[i])) * lin(hn, p[i + 1]), p[i + 2])
        i += 3
    }
    return matmul(MLXFast.rmsNorm(h, weight: p[1], eps: eps), p[0].T)
}
/// [KL(teacher || student), alpha = sum_v min(p, q) = 1 - TV], both averaged over positions; alpha carries no gradient.
func klLoss(_ p: [MLXArray], _ tokens: MLXArray, _ target: MLXArray) -> [MLXArray] {
    let s = logSoftmax(logits(p, tokens), axis: -1)
    return [(exp(target) * (target - s)).sum(axis: -1).mean(), stopGradient(minimum(exp(target), exp(s)).sum(axis: -1).mean())]
}
/// One cached decode step for B sequences: x [B] token ids at position t. Per layer, s1 holds the conv state [B, 2, D] (inputs at
/// t-2, t-1) or the attention keys [B, H, t, hd]; s2 the attention values. Returns logits [B, V].
func decodeStep(_ p: [MLXArray], _ x: MLXArray, _ s1: inout [MLXArray], _ s2: inout [MLXArray], t: Int) -> MLXArray {
    let H = 4, hd = 64, eps: Float = 1e-5, B = x.dim(0)
    func lin(_ x: MLXArray, _ w: MLXArray) -> MLXArray { matmul(x, w.T) }
    var h = p[0][x], i = 2
    for (l, attention) in [false, false, true, false].enumerated() {
        let xn = MLXFast.rmsNorm(h, weight: p[i], eps: eps), ffnGain = p[i + 1]
        let r: MLXArray
        if attention {
            func heads(_ z: MLXArray, _ gain: MLXArray?) -> MLXArray {
                var z = z.reshaped(B, 1, H, hd)
                if let gain { z = MLXFast.rmsNorm(z, weight: gain, eps: eps) }
                return z.transposed(0, 2, 1, 3)
            }
            func rope(_ z: MLXArray) -> MLXArray { MLXFast.RoPE(z, dimensions: hd, traditional: false, base: 1e6, scale: 1, offset: t) }
            let q = rope(heads(lin(xn, p[i + 2]), p[i + 6])), k = rope(heads(lin(xn, p[i + 3]), p[i + 7])), v = heads(lin(xn, p[i + 4]), nil)
            s1[l] = t == 0 ? k : concatenated([s1[l], k], axis: 2); s2[l] = t == 0 ? v : concatenated([s2[l], v], axis: 2)
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: s1[l], values: s2[l], scale: 0.125, mask: .none)
            r = lin(o.transposed(0, 2, 1, 3).reshaped(B, H * hd), p[i + 5])
            i += 8
        } else {
            let bx = lin(xn, p[i + 2]) * lin(xn, p[i + 4]), s = s1[l]
            let conv = s[0..., 0] * p[i + 6] + s[0..., 1] * p[i + 7] + bx * p[i + 8]
            s1[l] = concatenated([s[0..., 1...], bx[0..., .newAxis, 0...]], axis: 1)
            r = lin(lin(xn, p[i + 3]) * conv, p[i + 5])
            i += 9
        }
        h = h + r
        let hn = MLXFast.rmsNorm(h, weight: ffnGain, eps: eps)
        h = h + lin(silu(lin(hn, p[i])) * lin(hn, p[i + 1]), p[i + 2])
        i += 3
    }
    return matmul(MLXFast.rmsNorm(h, weight: p[1], eps: eps), p[0].T)
}

// Teacher and data, as ane-train.
let (teacher, _) = try await loadLLM("LiquidAI/LFM2-350M")
@MainActor func generate(_ count: Int, keep: Bool = false) throws -> (tokens: [[Int32]], steps: [MLXArray]) {
    // keep: also keep the decode-step teacher log-probs as KL targets, one [count, V] bf16 array per step; position t predicts token t+1,
    // so the last position needs one extra decode step on the final token (its sample is discarded).
    var x = MLXRandom.randInt(0 ..< V, [count, 1]).asType(.int32), rows = [x], lps: [MLXArray] = []
    let cache = try teacher.newCache(parameters: nil)
    for i in 0 ..< T {
        let logits = teacher(x, cache: cache)[0..., -1, 0...]
        if keep { let l = logSoftmax(logits.asType(.float32), axis: -1).asType(.bfloat16); eval(l); lps.append(l) }
        if i == T - 1 { break }
        x = categorical(logits)[0..., .newAxis].asType(.int32)
        eval(x); rows.append(x)
    }
    let all = concatenated(rows, axis: 1)
    return ((0 ..< count).map { all[$0].asArray(Int32.self) }, lps)
}
@MainActor func teacherLogProbs(_ tokens: MLXArray) -> MLXArray {
    let t = logSoftmax(teacher(tokens[.newAxis], cache: nil)[0].asType(.float32), axis: -1); eval(t); return t
}
/// Teacher log-probs (fp32 logSoftmax, stored bf16) of given sequences, in chunks of 32: [32, T, V] stays under 2^31 elements.
@MainActor func teacherScore(_ seqs: [[Int32]]) -> [MLXArray] {
    stride(from: 0, to: seqs.count, by: 32).map { s in
        let part = seqs[s ..< min(s + 32, seqs.count)]
        let lp = logSoftmax(teacher(MLXArray(part.flatMap { $0 }, [part.count, T]), cache: nil).asType(.float32), axis: -1).asType(.bfloat16); eval(lp); return lp
    }
}
/// Teacher NLL per token of tokens 1... under the teacher's own next-token log-probs [T, V] (position t predicts token t+1).
func teacherNLL(_ target: MLXArray, _ tokens: MLXArray) -> Float {
    -takeAlong(target[0 ..< T - 1], tokens[1...].asType(.int32)[0..., .newAxis], axis: -1).mean().item(Float.self)
}
MLXRandom.seed(424242)  // held-out set is independent of --seed
let evalRaw = try generate(evalCount).tokens
let evalSet = evalRaw.map { MLXArray($0) }
let evalTargets = evalSet.map(teacherLogProbs)
print(String(format: "teacher NLL on held-out teacher text (reference, ~entropy): %.3f nats/token", zip(evalTargets, evalSet).map { teacherNLL($0, $1) }.reduce(0, +) / Float(evalCount)))
print("eval set checksum: \(evalRaw.flatMap { $0 }.reduce(0) { ($0 &* 31 &+ Int($1)) & 0xFFFF_FFFF }) (\(evalCount) sequences)")
MLXRandom.seed(UInt64(seed))
// Pool entries: (tokens, group, index in group). A group is the decode-step targets (T arrays [n, V]) or a scored chunk [n, T, V].
enum Targets { case steps([MLXArray]), chunk(MLXArray) }
var pool: [([Int32], Int, Int)] = [], poolGroups: [Targets] = []
let poolSize = Int(option("pool") ?? "256")!  // targets stay per step: one [pool, T, V] array would pass 2^31 elements (MLX indexing) at 128+

var params = initParams(ternary: ternary)
eval(params)
let count = params.reduce(0) { $0 + $1.size }
print("float-train: \(count) parameters; \(updates) updates of \(T) tokens; lr \(lr)" + (ste ? "; ternary STE" : "") + (ternary ? "; ternary TTF theta \(theta) tau \(tau) rho x \(rhoScale)" : "") + (scheduled ? "; scheduled" : ""))

// --acc fp32|fp16|int8|stateless: TTF accumulator storage (4 / 2 / 1 / 0 bytes per weight).
let accMode = option("acc") ?? "fp32", pscale = Float(option("pscale") ?? "1")!  // pscale: stateless move-probability factor
let accDType: DType = accMode == "fp16" ? .float16 : accMode == "int8" ? .int8 : .float32
precondition(["fp32", "fp16", "int8", "stateless"].contains(accMode), "--acc fp32|fp16|int8|stateless")
// Ternary state: trits live in fp16 shared buffers (ttfKernel updates them in place); MLX sees them as fp32 arguments.
final class Trits {
    let n: Int, k: Int, buffer: SharedBuffer
    var acc: MLXArray, rows: MLXArray, cols: MLXArray
    init(n: Int, k: Int) {
        self.n = n; self.k = k; buffer = SharedBuffer([1, 1, n, k])
        acc = MLXArray.zeros([accMode == "stateless" ? 1 : n * k], dtype: accDType); rows = MLXArray.zeros([1, n]); cols = MLXArray.zeros([1, k])
        eval(buffer.write(MLXRandom.randInt(-1 ..< 2, [n * k]).asType(.float16)))
    }
    var value: MLXArray { buffer.array.reshaped(n, k).asType(.float32) }
    /// g [n, k] = dL/dtrit. Returns the number of trits moved.
    func update(g: MLXArray, rho: Float, leak: Float, theta: Float, step t: Int) -> MLXArray {
        let g2 = g * g
        rows = beta2 * rows + (1 - beta2) * g2.mean(axis: 1).reshaped(1, n); cols = beta2 * cols + (1 - beta2) * g2.mean(axis: 0).reshaped(1, k)
        let correction = 1 - pow(beta2, Float(t)), r = rows / correction
        let params = stacked([MLXArray(accMode == "stateless" ? rho * pscale : rho), MLXArray(leak), MLXArray(theta), r.mean()], axis: 0)
        let rowsF = r.reshaped(-1), colsF = (cols / correction).reshaped(-1)
        let out = accMode == "fp32"
            ? ttfKernel([g.reshaped(-1), acc, rowsF, colsF, params, buffer.array],
                        template: [("N", n), ("K", k)], grid: (n * k, 1, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[n * k], [n * k]], outputDTypes: [.float32, .float32])
            : ttfLeanKernel([g.reshaped(-1), acc, rowsF, colsF, params, buffer.array, MLXArray(UInt32(t))],
                            template: [("N", n), ("K", k), ("MODE", accMode == "fp16" ? 1 : accMode == "int8" ? 2 : 3), ("T", accDType)],
                            grid: (n * k, 1, 1), threadGroup: (256, 1, 1),
                            outputShapes: [[accMode == "stateless" ? 1 : n * k], [n * k]], outputDTypes: [accDType, .float32])
        if accMode != "stateless" { acc = out[0] }
        return out[1].sum()
    }
}
let tritState: [Trits] = ternary ? (0 ..< 4).flatMap { _ in [(256, 256), (256, 256), (256, 256), (256, 256), (F, D), (F, D), (D, F)] }.map { Trits(n: $0.0, k: $0.1) } : []
var m = params.map { zeros(like: $0) }, v = m

// a = params (49; ternary linears hold their scale [n]) + trits (0 or 28, fp32) + [tokens, target]; grads for all but the last two.
let lossGrad = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in
    let p = Array(a[0 ..< 49]), trits = Array(a[49 ..< a.count - 2])
    return klLoss(trits.isEmpty ? (ste ? steQuantize(p) : p) : compose(p, trits), a[a.count - 2], a[a.count - 1])
}, argumentNumbers: Array(0 ..< params.count + tritState.count))
@MainActor func studentWeights() -> [MLXArray] { ternary ? compose(params, tritState.map(\.value)) : ste ? steQuantize(params) : params }
/// Autoregressive samples (temperature 1, random first token, T tokens) from the current student; key != nil makes it reproducible
/// and leaves the global RNG alone.
@MainActor func studentGenerate(_ count: Int, key: MLXArray? = nil) -> [[Int32]] {
    let p = studentWeights(), keys = key.map { MLXRandom.split(key: $0, into: T + 1) }
    var x = MLXRandom.randInt(0 ..< V, [count], key: keys?[T]).asType(.int32), rows = [x]
    var s1 = (0 ..< 4).map { kinds[$0] ? MLXArray.zeros([1]) : MLXArray.zeros([count, 2, D]) }, s2 = s1
    for t in 0 ..< T - 1 {
        x = categorical(decodeStep(p, x, &s1, &s2, t: t), key: keys?[t]).asType(.int32)
        eval(x); rows.append(x)
    }
    let all = stacked(rows, axis: 1)
    return (0 ..< count).map { all[$0].asArray(Int32.self) }
}
/// (kl, alpha) averaged over sequences.
@MainActor func evaluate(_ seqs: [MLXArray], _ targets: [MLXArray]) -> (kl: Float, alpha: Float) {
    var kl = MLXArray(Float(0)), a = kl
    let p = studentWeights()
    for (tokens, target) in zip(seqs, targets) { let r = klLoss(p, tokens, target); kl = kl + r[0]; a = a + r[1]; eval(kl, a) }
    return (kl.item(Float.self) / Float(seqs.count), a.item(Float.self) / Float(seqs.count))
}
@MainActor func evaluate() -> (kl: Float, alpha: Float) { evaluate(evalSet, evalTargets) }
@MainActor func report(_ held: (kl: Float, alpha: Float), tokens: Int) {
    let zeros = ste ? ",\"zeros\":\(zeroFraction(params))" : ""
    // On-policy: the student writes evalCount sequences (fixed key), the teacher scores them.
    let t0 = Date(), seqs = studentGenerate(evalCount, key: MLXRandom.key(7777))
    let scored = teacherScore(seqs).flatMap { c in (0 ..< c.dim(0)).map { c[$0].asType(.float32) } }
    let on = evaluate(seqs.map { MLXArray($0) }, scored)
    let nll = zip(scored, seqs).map { teacherNLL($0, MLXArray($1)) }.reduce(0, +) / Float(evalCount)
    emit("{\"type\":\"eval\",\"tokens\":\(tokens),\"member\":0,\"kl\":\(num(held.kl)),\"alpha\":\(num(held.alpha)),\"onpolicy_kl\":\(num(on.kl)),\"teacher_nll\":\(num(nll)),\"onpolicy_alpha\":\(num(on.alpha)),\"lr\":\(lr)\(zeros)}")
    print(String(format: "eval at %d tokens: held-out kl %.3f alpha %.3f | on-policy kl %.3f teacher nll %.3f alpha %.3f (%.1fs)  lr %.2e", tokens, held.kl, held.alpha, on.kl, nll, on.alpha, Date().timeIntervalSince(t0), lr) + (ste ? String(format: "  zero trits %.3f", zeroFraction(params)) : ""))
}

if CommandLine.arguments.contains("--gentest") {  // cached decode == full forward? and student sampling speed
    let p = studentWeights(), seq = studentGenerate(8)[0], tk = MLXArray(seq), full = logits(p, tk)
    var s1 = (0 ..< 4).map { kinds[$0] ? MLXArray.zeros([1]) : MLXArray.zeros([1, 2, D]) }, s2 = s1, worst: Float = 0
    for t in 0 ..< T { let l = decodeStep(p, tk[t ..< t + 1], &s1, &s2, t: t); worst = max(worst, abs(l[0] - full[t]).max().item(Float.self)) }
    print("gentest: max |cached - full| logit \(worst)")
    let t0 = Date(); _ = studentGenerate(256); print(String(format: "gentest: student sampling %.1f ms/seq at 256", 1000 * Date().timeIntervalSince(t0) / 256))
    exit(0)
}
let started = Date()
let warmup = max(1, updates / 20)
var alphaWindow: [Float] = [], recentAlpha = 0.0
var tokensSeen = 0, nextRound = roundTokens, recent = 0.0, recentCount = 0, flipsRecent: Float = 0, generation = 0.0, working = 0.0
report(evaluate(), tokens: 0)
for step in 1 ... updates {
    if pool.isEmpty {
        let t0 = Date()
        let abar = alphaWindow.isEmpty ? 0 : alphaWindow.reduce(0, +) / Float(alphaWindow.count), tk = Float(tokensSeen)
        func clamp01(_ x: Float) -> Float { min(1, max(0, x)) }
        let lambda = opRamp.map { clamp01((tk - $0.0) / ($0.1 - $0.0)) } ?? opAlpha.map { clamp01((abar - $0.0) / ($0.1 - $0.0)) } ?? (tk >= opFrom ? opConst : 0)
        let nStudent = Int((lambda * Float(poolSize)).rounded()), nTeacher = poolSize - nStudent
        pool = []; poolGroups = []
        if nTeacher > 0 { let g = try generate(nTeacher, keep: true); poolGroups.append(.steps(g.steps)); pool += (0 ..< nTeacher).map { (g.tokens[$0], 0, $0) } }
        if nStudent > 0 {
            let seqs = studentGenerate(nStudent), scored = teacherScore(seqs)
            for (c, chunk) in scored.enumerated() { poolGroups.append(.chunk(chunk)); pool += (0 ..< chunk.dim(0)).map { (seqs[c * 32 + $0], poolGroups.count - 1, $0) } }
        }
        pool.shuffle()
        generation += Date().timeIntervalSince(t0)
        print(String(format: "pool refill at %d tokens: lambda %.3f (alpha-bar %.3f), %d student + %d teacher sequences, %.1fs", tokensSeen, lambda, abar, nStudent, nTeacher, Date().timeIntervalSince(t0)))
        emit("{\"type\":\"refill\",\"tokens\":\(tokensSeen),\"lambda\":\(lambda),\"alpha_bar\":\(abar),\"student\":\(nStudent),\"teacher\":\(nTeacher)}")
    }
    let (poolTokens, g, i) = pool.removeLast(); let tokens = MLXArray(poolTokens)
    let target: MLXArray
    switch poolGroups[g] { case .steps(let s): target = stacked(s.map { $0[i] }).asType(.float32); case .chunk(let c): target = c[i].asType(.float32) }
    tokensSeen += T
    let t0 = Date()
    let (loss, grads) = lossGrad(params + tritState.map(\.value) + [tokens, target])
    // ane-train schedules: rho warm-up 0.25 -> 1 then cosine to 0.2 (times rhoScale); Adam rate warm-up then cosine to 0.1.
    let cosine = 0.5 * (1 + cos(Float.pi * max(0, Float(step - warmup)) / Float(max(1, updates - warmup))))
    let rhoNow = (step < warmup ? 0.25 + 0.75 * Float(step) / Float(warmup) : 0.2 + 0.8 * cosine) * rhoScale
    let rate = scheduled ? lr * (step < warmup ? Float(step) / Float(warmup) : 0.1 + 0.9 * cosine) : lr
    let b1 = pow(0.9, Float(step)), b2 = pow(0.999, Float(step))
    for j in params.indices {
        m[j] = 0.9 * m[j] + 0.1 * grads[j]; v[j] = 0.999 * v[j] + 0.001 * grads[j] * grads[j]
        params[j] = params[j] - rate * (m[j] / (1 - b1)) / (sqrt(v[j] / (1 - b2)) + 1e-8)
    }
    var moved = MLXArray(Float(0))
    for (j, t) in tritState.enumerated() { moved = moved + t.update(g: grads[params.count + j], rho: rhoNow, leak: 1 - 1 / tau, theta: theta, step: step) }
    var toEval: [MLXArray] = params + m + v; toEval += [loss[0], loss[1], moved]; toEval += tritState.map(\.acc)
    eval(toEval)
    flipsRecent += moved.item(Float.self)
    working += Date().timeIntervalSince(t0)
    recent += Double(loss[0].item(Float.self)); recentCount += 1
    let a = loss[1].item(Float.self); recentAlpha += Double(a); alphaWindow.append(a); if alphaWindow.count > 50 { alphaWindow.removeFirst() }
    if step % 25 == 0 || step == 1 {
        let kl = Float(recent / Double(recentCount))
        let flipsMean = flipsRecent / Float(recentCount)  // trits moved per update, as ane-train's flips
        print(String(format: "update %5d  tokens %8d  train kl %.3f alpha %.3f  flips %.0f  %.0f tok/s", step, tokensSeen, kl, Float(recentAlpha / Double(recentCount)), flipsMean, Double(tokensSeen) / Date().timeIntervalSince(started)))
        emit("{\"type\":\"train\",\"update\":\(step),\"tokens\":\(tokensSeen),\"kl\":[\(num(kl))],\"alpha\":[\(num(Float(recentAlpha / Double(recentCount))))],\"flips\":[\(num(flipsMean))]}")
        flipsRecent = 0; recent = 0; recentCount = 0; recentAlpha = 0
    }
    if tokensSeen >= nextRound || step == updates { nextRound += roundTokens; report(evaluate(), tokens: tokensSeen) }
}
print(String(format: "pool generation (teacher decode, student sampling, scoring) %.1f ms per sequence", 1000 * generation / Double(tokensSeen / T)))
print(String(format: "throughput excluding text generation: %.0f tok/s per member, %.0f member-tok/s", Double(tokensSeen) / working, Double(tokensSeen) / working))
emit("{\"type\":\"done\",\"tokens\":\(tokensSeen),\"seconds\":\(Date().timeIntervalSince(started))}")
