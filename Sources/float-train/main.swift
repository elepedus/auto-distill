// Float GPU baseline for ane-train: the same LFM2-shaped student (d 256, conv/conv/attn/conv, tied head), plain fp32
// weights, Adam on everything, MLX autodiff, same teacher-generated data and held-out KL eval.
//   float-train [--tokens 2048000] [--round 256000] [--lr 1e-3] [--seed 0] [--eval 8] [--jsonl PATH]
import Foundation
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

func option(_ name: String) -> String? { CommandLine.arguments.firstIndex(of: "--" + name).map { CommandLine.arguments[$0 + 1] } }
let T = 256, D = 256, F = 768, H = 4, hd = 64, V = 65536
let kinds = [false, false, true, false]  // attention layer?
let tokenBudget = Int(option("tokens") ?? "2048000")!, roundTokens = Int(option("round") ?? "256000")!
let lr = Float(option("lr") ?? "1e-3")!, seed = Int(option("seed") ?? "0")!, evalCount = Int(option("eval") ?? "8")!
let updates = tokenBudget / T

let jsonl = option("jsonl").flatMap { path -> FileHandle? in
    if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
    let h = FileHandle(forWritingAtPath: path); _ = try? h?.seekToEnd(); return h
}
func num(_ x: Float) -> String { x.isFinite ? "\(x)" : "null" }
func emit(_ line: String) { try? jsonl?.write(contentsOf: Data((line + "\n").utf8)) }

// Parameters, flat: [embed, final gain] then per layer
//   conv:  g_op g_ffn b c x o k0 k1 k2 w1 w3 w2     attn: g_op g_ffn q k v o g_q g_k w1 w3 w2
func initParams() -> [MLXArray] {
    func lin(_ n: Int, _ k: Int) -> MLXArray { MLXRandom.normal([n, k]) / sqrt(Float(k)) }  // ane-train: trits x scale, std 1/sqrt(k)
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
func klLoss(_ p: [MLXArray], _ tokens: MLXArray, _ target: MLXArray) -> MLXArray {
    let s = logSoftmax(logits(p, tokens), axis: -1)
    return (exp(target) * (target - s)).sum(axis: -1).mean()  // KL(teacher || student), mean over tokens
}

// Teacher and data, as ane-train.
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
MLXRandom.seed(424242)  // held-out set is independent of --seed
let evalRaw = try generate(evalCount)
let evalSet = evalRaw.map { MLXArray($0) }
let evalTargets = evalSet.map(teacherLogProbs)
print("eval set checksum: \(evalRaw.flatMap { $0 }.reduce(0) { ($0 &* 31 &+ Int($1)) & 0xFFFF_FFFF }) (\(evalCount) sequences)")
MLXRandom.seed(UInt64(seed))
var pool: [[Int32]] = []

var params = initParams()
eval(params)
let count = params.reduce(0) { $0 + $1.size }
print("float-train: \(count) parameters; \(updates) updates of \(T) tokens; lr \(lr)")
var m = params.map { zeros(like: $0) }, v = m

let lossGrad = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in [klLoss(Array(a.dropLast(2)), a[a.count - 2], a[a.count - 1])] },
                            argumentNumbers: Array(0 ..< params.count))
@MainActor func evaluate() -> Float {
    var total = MLXArray(Float(0))
    for (tokens, target) in zip(evalSet, evalTargets) { total = total + klLoss(params, tokens, target); eval(total) }
    return (total / Float(evalSet.count)).item(Float.self)
}
@MainActor func report(_ kl: Float, tokens: Int) {
    emit("{\"type\":\"eval\",\"tokens\":\(tokens),\"member\":0,\"kl\":\(num(kl)),\"lr\":\(lr)}")
    print(String(format: "eval at %d tokens: held-out kl %.3f  lr %.2e", tokens, kl, lr))
}

let started = Date()
var tokensSeen = 0, nextRound = roundTokens, recent = 0.0, recentCount = 0, generation = 0.0, working = 0.0
report(evaluate(), tokens: 0)
for step in 1 ... updates {
    if pool.isEmpty { let t0 = Date(); pool = try generate(256); generation += Date().timeIntervalSince(t0) }
    let tokens = MLXArray(pool.removeLast())
    tokensSeen += T
    let t0 = Date()
    let target = teacherLogProbs(tokens)
    let (loss, grads) = lossGrad(params + [tokens, target])
    let b1 = pow(0.9, Float(step)), b2 = pow(0.999, Float(step))
    for j in params.indices {
        m[j] = 0.9 * m[j] + 0.1 * grads[j]; v[j] = 0.999 * v[j] + 0.001 * grads[j] * grads[j]
        params[j] = params[j] - lr * (m[j] / (1 - b1)) / (sqrt(v[j] / (1 - b2)) + 1e-8)
    }
    eval(params + m + v + loss)
    working += Date().timeIntervalSince(t0)
    recent += Double(loss[0].item(Float.self)); recentCount += 1
    if step % 25 == 0 || step == 1 {
        let kl = Float(recent / Double(recentCount))
        print(String(format: "update %5d  tokens %8d  train kl %.3f  %.0f tok/s", step, tokensSeen, kl, Double(tokensSeen) / Date().timeIntervalSince(started)))
        emit("{\"type\":\"train\",\"update\":\(step),\"tokens\":\(tokensSeen),\"kl\":[\(num(kl))]}")
        recent = 0; recentCount = 0
    }
    if tokensSeen >= nextRound || step == updates { nextRound += roundTokens; report(evaluate(), tokens: tokensSeen) }
}
print(String(format: "teacher text generation %.1f ms per sequence", 1000 * generation / Double(tokensSeen / T)))
print(String(format: "throughput excluding text generation: %.0f tok/s per member, %.0f member-tok/s", Double(tokensSeen) / working, Double(tokensSeen) / working))
emit("{\"type\":\"done\",\"tokens\":\(tokensSeen),\"seconds\":\(Date().timeIntervalSince(started))}")
