// Ternarise a pretrained LFM2-350M and recover it by distillation against its own bf16 base.
//   ternarise measure [--init absmean|twn] [--keep-fp 0,15|all] [--text FILE] [--eval 8] [--jsonl PATH]
//   ternarise train   [measure options] [--ste] [--tokens 2048000] [--round 256000] [--seed 0] [--lr 1e-5]
//                     [--hyper lr,theta,tau,rho] [--acc-init zero|residual] [--kappa 1] [--pool 256] [--ppl-every 1024000]
// Every linear in every decoder layer becomes trits x per-row scale (embedding/tied head, norms, conv taps stay float).
// train: trits by TTF (float-train --ternary's kernel and maths; no master weights), the rest by Adam. --ste: fp32 master
// weights + absmean STE + Adam, as an upper-bound reference. Student compute: bf16 matmuls, fp32 everything else.
import Foundation
import LFM2ANE
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

func option(_ name: String) -> String? { CommandLine.arguments.firstIndex(of: "--" + name).map { CommandLine.arguments[$0 + 1] } }
func num(_ x: Float) -> String { x.isFinite ? "\(x)" : "null" }
let T = 256, V = 65536, beta2: Float = 0.95

/// Trits x row scale. `weight` holds the trits (float; TTF rewrites it each step), `scale` is trained by Adam.
final class TernaryLinear: Linear {
    @ParameterInfo var scale: MLXArray
    init(trits: MLXArray, scale: MLXArray) { _scale.wrappedValue = scale; super.init(weight: trits, bias: nil) }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        (matmul(x.asType(.bfloat16), weight.asType(.bfloat16).T) * scale.asType(.bfloat16)).asType(.float32)
    }
}
/// fp32 master weight; forward s * round(clip(W/s, -1, 1)) with s = row absmean, straight-through gradient.
final class STELinear: Linear {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = abs(weight).mean(axis: 1, keepDims: true) + 1e-5
        let q = weight + stopGradient(s * round(clip(weight / s, min: -1, max: 1)) - weight)
        return matmul(x.asType(.bfloat16), q.asType(.bfloat16).T).asType(.float32)
    }
}

/// w [n, k] fp32 -> (trits [n, k], scale [n]).
func quantise(_ w: MLXArray, twn: Bool) -> (MLXArray, MLXArray) {
    let a = abs(w)
    if twn {
        let mask = (a .> 0.7 * a.mean(axis: 1, keepDims: true)).asType(.float32)
        let s = (a * mask).sum(axis: 1, keepDims: true) / maximum(mask.sum(axis: 1, keepDims: true), 1)
        return (sign(w) * mask, s.squeezed(axis: 1))
    }
    let s = maximum(a.mean(axis: 1, keepDims: true), 1e-8)
    return (round(clip(w / s, min: -1, max: 1)), s.squeezed(axis: 1))
}

/// TTF state of one matrix, as float-train's Trits: trits in an fp16 shared buffer, fp32 accumulator, factored second moments.
final class Trits {
    let n: Int, k: Int, buffer: SharedBuffer
    var acc: MLXArray, rows: MLXArray, cols: MLXArray
    init(trits: MLXArray, acc: MLXArray) {
        n = trits.dim(0); k = trits.dim(1); buffer = SharedBuffer([1, 1, n, k]); self.acc = acc.reshaped(-1)
        rows = MLXArray.zeros([1, n]); cols = MLXArray.zeros([1, k])
        eval(buffer.write(trits.reshaped(-1).asType(.float16)), self.acc)
    }
    var value: MLXArray { buffer.array.reshaped(n, k).asType(.float32) }
    func update(g: MLXArray, rho: Float, leak: Float, theta: Float, step t: Int) -> MLXArray {
        let g2 = g * g
        rows = beta2 * rows + (1 - beta2) * g2.mean(axis: 1).reshaped(1, n); cols = beta2 * cols + (1 - beta2) * g2.mean(axis: 0).reshaped(1, k)
        let correction = 1 - pow(beta2, Float(t)), r = rows / correction
        let params = stacked([MLXArray(rho), MLXArray(leak), MLXArray(theta), r.mean()], axis: 0)
        let out = ttfKernel([g.reshaped(-1), acc, r.reshaped(-1), (cols / correction).reshaped(-1), params, buffer.array],
                            template: [("N", n), ("K", k)], grid: (n * k, 1, 1), threadGroup: (256, 1, 1),
                            outputShapes: [[n * k], [n * k]], outputDTypes: [.float32, .float32])
        acc = out[0]
        return out[1].sum()
    }
}

func klLoss(_ model: any LLMModel, _ tokens: MLXArray, _ target: MLXArray) -> [MLXArray] {
    let s = logSoftmax(model(tokens[.newAxis], cache: nil)[0].asType(.float32), axis: -1), p = exp(target)
    return [(p * (target - s)).sum(axis: -1).mean(), stopGradient(minimum(p, exp(s)).sum(axis: -1).mean())]
}

@MainActor func run() async throws {
    let mode = CommandLine.arguments.dropFirst().first ?? "", training = mode == "train"
    let ste = CommandLine.arguments.contains("--ste"), twn = (option("init") ?? "absmean") == "twn"
    let hyp = (option("hyper") ?? "1e-5,16,1000,0.25").split(separator: ",").map { Float($0)! }
    let lr = option("lr").map { Float($0)! } ?? hyp[0], theta = hyp[1], tau = hyp[2], rhoScale = hyp[3]
    let residual = (option("acc-init") ?? "zero") == "residual", kappa = Float(option("kappa") ?? "1")!
    let evalCount = Int(option("eval") ?? "8")!, seed = Int(option("seed") ?? "0")!
    let tokenBudget = Int(option("tokens") ?? "2048000")!, roundTokens = Int(option("round") ?? "256000")!
    let pplEvery = Int(option("ppl-every") ?? "1024000")!, poolSize = Int(option("pool") ?? "256")!
    let jsonl = option("jsonl").flatMap { path -> FileHandle? in
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        let h = FileHandle(forWritingAtPath: path); _ = try? h?.seekToEnd(); return h
    }
    func emit(_ line: String) { try? jsonl?.write(contentsOf: Data((line + "\n").utf8)) }

    let (teacher, dir) = try await loadLLM("LiquidAI/LFM2-350M")
    let (student, _) = try await loadLLM("LiquidAI/LFM2-350M")
    let keepArg = option("keep-fp") ?? ""
    let keep = Set(keepArg == "all" ? Array(0 ..< 64) : keepArg.split(separator: ",").map { Int($0)! })

    // Wikitext-2 test: BOS + 511 tokens x 64 windows, as `distill eval`.
    let tokenizer = try await AutoTokenizer.from(modelFolder: dir)
    let text = try String(contentsOf: URL(filePath: option("text") ?? "/tmp/wikitext2-test-nn.txt"), encoding: .utf8)
    let ids = tokenizer.encode(text: text, addSpecialTokens: false).prefix(64 * 511).map(Int32.init)
    let windows = concatenated([MLXArray.full([64, 1], values: MLXArray(Int32(tokenizer.bosTokenId ?? 1))), MLXArray(ids, [64, 511])], axis: 1)
    func perplexity(_ model: any LLMModel) -> Float {
        var nll = MLXArray(Float(0))
        for i in stride(from: 0, to: 64, by: 8) {
            let x = windows[i ..< i + 8], lp = logSoftmax(model(x, cache: nil).asType(.float32), axis: -1)
            nll = nll + takeAlong(lp[0..., ..<(-1)], x[0..., 1...][.ellipsis, .newAxis], axis: -1).sum(); eval(nll)
        }
        return exp(-nll / Float(64 * 511)).item(Float.self)
    }

    // Replace layer linears; TTF state is keyed by the flattened parameter name of the trits.
    var tritState: [String: Trits] = [:], ternaryCount = 0, keptCount = 0, zeroTrits = 0, replaced: [(String, Module)] = []
    for (path, m) in student.leafModules().flattened() {
        guard let lin = m as? Linear else { continue }
        let parts = path.split(separator: "."), li = parts.firstIndex(of: "layers").flatMap { Int(parts[$0 + 1]) }
        guard let layer = li else { continue }
        let w = lin.weight.asType(.float32)
        if keep.contains(layer) { keptCount += w.size; m.freeze(); continue }
        ternaryCount += w.size
        if ste { replaced.append((path, STELinear(weight: w, bias: nil))); continue }
        let (t, s) = quantise(w, twn: twn)
        zeroTrits += (t .== 0).sum().item(Int.self)
        let l = TernaryLinear(trits: t, scale: s)
        if training {
            let acc = residual ? clip(kappa * 2 * theta * (w / s.expandedDimensions(axis: 1) - t), min: -0.999 * theta, max: 0.999 * theta) : MLXArray.zeros(like: w)
            let tr = Trits(trits: t, acc: acc)
            tritState[path + ".weight"] = tr; l.weight = tr.value
        }
        eval(l)
        replaced.append((path, l))
    }
    student.update(modules: ModuleChildren.unflattened(replaced))
    student.update(parameters: student.parameters().mapValues { $0.asType(.float32) })  // norms, embedding, conv taps; trits are exact
    eval(student)
    let total = student.parameters().flattened().reduce(0) { $0 + $1.1.size }
    print("ternarise \(mode): \(ternaryCount) ternary of \(total) parameters" + (ste ? " (STE reference)" : " (\(twn ? "twn" : "absmean"), \(String(format: "%.3f", Double(zeroTrits) / Double(max(ternaryCount, 1)))) zero trits)") + "; keep-fp \(keep.sorted())")

    // Held-out set: teacher text under a fixed seed, targets from the teacher's full forward.
    @MainActor func generate(_ count: Int, keep: Bool = false) throws -> (tokens: [[Int32]], steps: [MLXArray]) {
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
    MLXRandom.seed(424242)
    let evalSet = try generate(evalCount).tokens.map { MLXArray($0) }
    let evalTargets = evalSet.map { t -> MLXArray in let l = logSoftmax(teacher(t[.newAxis], cache: nil)[0].asType(.float32), axis: -1); eval(l); return l }
    func heldOut() -> (kl: Float, alpha: Float) {
        var kl = MLXArray(Float(0)), a = kl
        for (t, y) in zip(evalSet, evalTargets) { let r = klLoss(student, t, y); kl = kl + r[0]; a = a + r[1]; eval(kl, a) }
        return (kl.item(Float.self) / Float(evalCount), a.item(Float.self) / Float(evalCount))
    }

    if !training {
        let h = heldOut(), ppl = perplexity(student), tppl = perplexity(teacher)
        print(String(format: "held-out kl %.4f alpha %.4f  wikitext2 ppl %.2f  (bf16 teacher ppl %.2f)", h.kl, h.alpha, ppl, tppl))
        emit("{\"type\":\"measure\",\"init\":\"\(twn ? "twn" : "absmean")\",\"keep_fp\":\"\(keepArg)\",\"kl\":\(num(h.kl)),\"alpha\":\(num(h.alpha)),\"ppl\":\(num(ppl)),\"teacher_ppl\":\(num(tppl)),\"zero_fraction\":\(Double(zeroTrits) / Double(max(ternaryCount, 1)))}")
        return
    }

    // Bytes per parameter. Float parameters (embedding, norms, scales, taps): fp32 weight + Adam m, v = 12.
    let other = total - ternaryCount - keptCount
    let perTernary = ste ? 12 : 6  // STE: fp32 master + Adam m, v. TTF: fp16 trits + fp32 accumulator (factored moments ~0)
    print(String(format: "state bytes per ternary parameter: %d (%@); float parameters 12; model total %.2f B/param (+4 transient gradient)", perTernary, ste ? "fp32 master, Adam m, v" : "fp16 trits, fp32 accumulator", Double(perTernary * ternaryCount + 12 * other) / Double(total)))

    let updates = tokenBudget / T, warmup = max(1, updates / 20)
    print("train: \(updates) updates of \(T) tokens; lr \(lr)" + (ste ? "; STE" : "; TTF theta \(theta) tau \(tau) rho x \(rhoScale) acc-init \(residual ? "residual k=\(kappa)" : "zero")"))
    let lossGrad = valueAndGrad(model: student) { model, a in klLoss(model as! any LLMModel, a[0], a[1]) }
    var m: [String: MLXArray] = [:], v: [String: MLXArray] = [:]

    MLXRandom.seed(UInt64(seed))
    var pool: [([Int32], Int, Int)] = [], steps: [MLXArray] = []
    var tokensSeen = 0, nextRound = roundTokens, nextPpl = pplEvery, recent = 0.0, recentAlpha = 0.0, recentCount = 0
    var flipsRecent: Float = 0, flipsTotal: Float = 0, generation = 0.0, working = 0.0
    let started = Date()
    func report(_ tokens: Int, withPpl: Bool) {
        let h = heldOut(), ppl = withPpl ? perplexity(student) : Float.nan
        let p = withPpl ? ",\"ppl\":\(num(ppl))" : ""
        emit("{\"type\":\"eval\",\"tokens\":\(tokens),\"member\":0,\"kl\":\(num(h.kl)),\"alpha\":\(num(h.alpha)),\"lr\":\(lr)\(p),\"flips_total\":\(flipsTotal)}")
        print(String(format: "eval at %d tokens: held-out kl %.4f alpha %.4f", tokens, h.kl, h.alpha) + (withPpl ? String(format: "  ppl %.2f", ppl) : "") + String(format: "  flips total %.0f", flipsTotal))
    }
    report(0, withPpl: true)
    for step in 1 ... updates {
        if pool.isEmpty {
            let t0 = Date(), g = try generate(poolSize, keep: true)
            steps = g.steps; pool = (0 ..< poolSize).map { (g.tokens[$0], 0, $0) }; pool.shuffle()
            generation += Date().timeIntervalSince(t0)
        }
        let (poolTokens, _, i) = pool.removeLast()
        let tokens = MLXArray(poolTokens), target = stacked(steps.map { $0[i] }).asType(.float32)
        tokensSeen += T
        let t0 = Date()
        let (loss, grads) = lossGrad(student, [tokens, target])
        let cosine = 0.5 * (1 + cos(Float.pi * max(0, Float(step - warmup)) / Float(max(1, updates - warmup))))
        let rhoNow = (step < warmup ? 0.25 + 0.75 * Float(step) / Float(warmup) : 0.2 + 0.8 * cosine) * rhoScale
        let rate = lr * (step < warmup ? Float(step) / Float(warmup) : 0.1 + 0.9 * cosine)
        let b1 = pow(0.9, Float(step)), b2 = pow(0.999, Float(step))
        let current = Dictionary(uniqueKeysWithValues: student.trainableParameters().flattened())
        var fresh: [(String, MLXArray)] = [], moved = MLXArray(Float(0)), toEval: [MLXArray] = [loss[0], loss[1]]
        for (k, g) in grads.flattened() {
            if let tr = tritState[k] { moved = moved + tr.update(g: g, rho: rhoNow, leak: 1 - 1 / tau, theta: theta, step: step); toEval.append(tr.acc); continue }
            let mk = 0.9 * (m[k] ?? zeros(like: g)) + 0.1 * g, vk = 0.999 * (v[k] ?? zeros(like: g)) + 0.001 * g * g
            m[k] = mk; v[k] = vk
            let p = current[k]! - rate * (mk / (1 - b1)) / (sqrt(vk / (1 - b2)) + 1e-8)
            fresh.append((k, p)); toEval += [p, mk, vk]
        }
        toEval.append(moved)
        eval(toEval)
        student.update(parameters: ModuleParameters.unflattened(fresh))
        for (path, mod) in replaced where !ste { (mod as! TernaryLinear).weight = tritState[path + ".weight"]!.value }
        working += Date().timeIntervalSince(t0)
        let f = moved.item(Float.self); flipsRecent += f; flipsTotal += f
        recent += Double(loss[0].item(Float.self)); recentAlpha += Double(loss[1].item(Float.self)); recentCount += 1
        if step % 25 == 0 || step == 1 {
            let kl = Float(recent / Double(recentCount)), al = Float(recentAlpha / Double(recentCount)), fl = flipsRecent / Float(recentCount)
            print(String(format: "update %5d  tokens %8d  train kl %.3f alpha %.3f  flips %.0f  %.0f tok/s", step, tokensSeen, kl, al, fl, Double(tokensSeen) / Date().timeIntervalSince(started)))
            emit("{\"type\":\"train\",\"update\":\(step),\"tokens\":\(tokensSeen),\"kl\":[\(num(kl))],\"alpha\":[\(num(al))],\"flips\":[\(num(fl))]}")
            flipsRecent = 0; recent = 0; recentAlpha = 0; recentCount = 0
        }
        if tokensSeen >= nextRound || step == updates {
            nextRound += roundTokens
            let withPpl = step == updates || tokensSeen >= nextPpl
            if withPpl { while nextPpl <= tokensSeen { nextPpl += pplEvery } }
            report(tokensSeen, withPpl: withPpl)
        }
    }
    print(String(format: "pool generation %.1f ms per sequence; throughput excluding generation %.0f tok/s, including %.0f tok/s; peak memory %.1f GB",
                 1000 * generation / Double(tokensSeen / T), Double(tokensSeen) / working, Double(tokensSeen) / Date().timeIntervalSince(started), Double(Memory.peakMemory) / 1e9))
    emit("{\"type\":\"done\",\"tokens\":\(tokensSeen),\"seconds\":\(Date().timeIntervalSince(started)),\"state_bytes_per_ternary_param\":\(perTernary),\"peak_gb\":\(Double(Memory.peakMemory) / 1e9)}")
}
try await run()
