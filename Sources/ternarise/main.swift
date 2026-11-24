// Ternarise a pretrained LFM2-350M and recover it by distillation against its own bf16 base.
//   ternarise measure [--init absmean|twn] [--keep-fp 0,15|all] [--text FILE] [--eval 8] [--jsonl PATH]
//   ternarise train   [measure options] [--ste] [--tokens 2048000] [--round 256000] [--seed 0] [--lr 1e-5]
//                     [--save DIR --save-every N --resume DIR --schedule cosine|wsd --decay-frac 0.15 --acc fp32|int8]
//                     [--hyper lr,theta,tau,rho] [--acc-init zero|residual] [--kappa 1] [--pool 256] [--ppl-every 1024000]
//                     [--colscale [--untied]] [--layerwise TOKENS]
// --colscale: W ~ diag(r) T diag(c), c per input channel (alternating-minimisation init, Adam-trained). --layerwise: before the
// end-to-end phase, TOKENS of per-layer reconstruction (student layer on the teacher's input -> teacher's output).
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
let int8Acc = option("acc") == "int8"  // accumulator in int8 units of theta/127, stochastic rounding (ttfLeanKernel MODE 2)

/// Trits x row scale. `weight` holds the trits (float; TTF rewrites it each step), `scale` is trained by Adam.
final class TernaryLinear: Linear {
    @ParameterInfo var scale: MLXArray
    @ParameterInfo var colscale: MLXArray?  // per input channel, optional: (x * c) @ trits^T * r
    var hook: ((MLXArray) -> MLXArray)?  // layerwise phase only: rewrites the output
    init(trits: MLXArray, scale: MLXArray, colscale: MLXArray? = nil) {
        _scale.wrappedValue = scale; _colscale.wrappedValue = colscale; super.init(weight: trits, bias: nil)
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xc = colscale.map { x * $0 } ?? x
        let y = (matmul(xc.asType(.bfloat16), weight.asType(.bfloat16).T) * scale.asType(.bfloat16)).asType(.float32)
        return hook?(y) ?? y
    }
}
/// RMSNorm that remembers its last input while `on` (layerwise phase: capture layer inputs / residuals).
final class Tap: RMSNorm {
    var on = false, last: MLXArray?
    init(_ n: RMSNorm) {
        super.init(dimensions: n.weight.dim(0), eps: n.eps)
        update(parameters: ModuleParameters.unflattened([("weight", n.weight)]))
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray { if on { last = x }; return super.callAsFunction(x) }
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

/// Two-sided init, W ~ diag(r) T diag(c): rounds of (T by the TWN rule on W/c, r by least squares; c by per-column least squares).
/// `fixedC`: tied scale, only T and r are fitted.
func quantiseTwoSided(_ w: MLXArray, rounds: Int = 4, fixedC: MLXArray? = nil) -> (MLXArray, MLXArray, MLXArray) {
    var c = fixedC ?? MLXArray.ones([w.dim(1)]), t = w, r = MLXArray.ones([w.dim(0)])
    for _ in 0 ..< (fixedC == nil ? rounds : 1) {
        t = quantise(w / c, twn: true).0
        let tc = t * c
        r = (w * tc).sum(axis: 1) / maximum((tc * tc).sum(axis: 1), 1e-12)
        if fixedC != nil { break }
        let u = t * r.expandedDimensions(axis: 1)
        c = (w * u).sum(axis: 0) / maximum((u * u).sum(axis: 0), 1e-12)
        c = maximum(c, 0.01 * c.mean())  // keep positive and non-degenerate
    }
    return (t, r, c)
}

/// Linears that share an input share one column scale (foldable into the preceding norm): q/k/v, and w1/w3.
func tieGroup(_ path: String) -> (group: String, members: [String])? {
    var p = path.split(separator: ".").map(String.init); guard let last = p.popLast() else { return nil }
    let sets = [["q_proj", "k_proj", "v_proj"], ["w1", "w3"]]
    guard let m = sets.first(where: { $0.contains(last) }) else { return nil }
    return (p.joined(separator: ".") + "." + m.joined(separator: "+"), m.map { (p + [$0]).joined(separator: ".") })
}
/// Sum the column-scale gradients within each tie group, so identical Adam updates keep tied copies identical.
func tieColscale(_ grads: [(String, MLXArray)]) -> [(String, MLXArray)] {
    var sums: [String: MLXArray] = [:]
    for (k, g) in grads where k.hasSuffix(".colscale") { if let t = tieGroup(String(k.dropLast(9))) { sums[t.group] = (sums[t.group] ?? MLXArray(Float(0))) + g } }
    return grads.map { k, g in k.hasSuffix(".colscale") ? (k, tieGroup(String(k.dropLast(9))).flatMap { sums[$0.group] } ?? g) : (k, g) }
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
        let rv = r.reshaped(-1), cv = (cols / correction).reshaped(-1)
        let out = int8Acc
            ? ttfLeanKernel([g.reshaped(-1), acc, rv, cv, params, buffer.array, MLXArray(UInt32(t))],
                            template: [("N", n), ("K", k), ("MODE", 2), ("T", DType.int8)], grid: (n * k, 1, 1), threadGroup: (256, 1, 1),
                            outputShapes: [[n * k], [n * k]], outputDTypes: [.int8, .float32])
            : ttfKernel([g.reshaped(-1), acc, rv, cv, params, buffer.array],
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
    let colscale = CommandLine.arguments.contains("--colscale"), untied = CommandLine.arguments.contains("--untied"), lwTokens = Int(option("layerwise") ?? "0")!
    let evalCount = Int(option("eval") ?? "8")!, seed = Int(option("seed") ?? "0")!
    let tokenBudget = Int(option("tokens") ?? "2048000")!, roundTokens = Int(option("round") ?? "256000")!
    let pplEvery = Int(option("ppl-every") ?? "1024000")!, poolSize = Int(option("pool") ?? "256")!
    let jsonl = option("jsonl").flatMap { path -> FileHandle? in
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        let h = FileHandle(forWritingAtPath: path); _ = try? h?.seekToEnd(); return h
    }
    func emit(_ line: String) { try? jsonl?.write(contentsOf: Data((line + "\n").utf8)) }

    let (teacher, baseDir) = try await loadLLM("LiquidAI/LFM2-350M")
    let (student, _) = try await loadLLM("LiquidAI/LFM2-350M")
    let keepArg = option("keep-fp") ?? ""
    let keep = Set(keepArg == "all" ? Array(0 ..< 64) : keepArg.split(separator: ",").map { Int($0)! })

    // Wikitext-2 test: BOS + 511 tokens x 64 windows, as `distill eval`.
    let tokenizer = try await AutoTokenizer.from(modelFolder: baseDir)
    let text = try String(contentsOf: URL(filePath: option("text") ?? "/tmp/wikitext2-test-join.txt"), encoding: .utf8)
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

    // --resume: checkpoint arrays (trits, scales, floats) and optimiser state.
    var ckpt: [String: MLXArray] = [:], opt: [String: MLXArray] = [:], resumeStep = 0, resumeTokens = 0
    if let r = option("resume") {
        ckpt = try loadArrays(url: URL(filePath: r).appending(path: "model.safetensors"))
        let (o, meta) = try loadArraysAndMetadata(url: URL(filePath: r).appending(path: "optim.safetensors"))
        opt = o; resumeStep = Int(meta["step"]!)!; resumeTokens = Int(meta["tokens"]!)!
        print("resume \(r): step \(resumeStep), \(resumeTokens) tokens")
    }
    // Replace layer linears; TTF state is keyed by the flattened parameter name of the trits.
    let leaves = Dictionary(uniqueKeysWithValues: student.leafModules().flattened().compactMap { p, m in (m as? Linear).map { (p, $0) } })
    var tiedC: [String: MLXArray] = [:], recon: [String: [Float]] = [:], tritState: [String: Trits] = [:], ternaryCount = 0, keptCount = 0, zeroTrits = 0, replaced: [(String, Module)] = []
    for (path, m) in student.leafModules().flattened() {
        guard let lin = m as? Linear else { continue }
        let parts = path.split(separator: "."), li = parts.firstIndex(of: "layers").flatMap { Int(parts[$0 + 1]) }
        guard let layer = li else { continue }
        let w = lin.weight.asType(.float32)
        if keep.contains(layer) { keptCount += w.size; m.freeze(); replaced.append((path, m)); continue }
        ternaryCount += w.size
        if ste { replaced.append((path, STELinear(weight: w, bias: nil))); continue }
        var (t, s) = quantise(w, twn: twn)
        var c: MLXArray?
        if colscale {
            let (t0, s0) = quantise(w, twn: true), sls = (w * t0).sum(axis: 1) / maximum((t0 * t0).sum(axis: 1), 1e-12)
            var cFix: MLXArray?
            if !untied, let g = tieGroup(path) {
                if tiedC[g.group] == nil { tiedC[g.group] = quantiseTwoSided(concatenated(g.members.map { leaves[$0]!.weight.asType(.float32) }, axis: 0)).2; eval(tiedC[g.group]!) }
                cFix = tiedC[g.group]
            }
            let (t2, s2, c2) = quantiseTwoSided(w, fixedC: cFix)
            func err(_ t: MLXArray, _ r: MLXArray, _ c: MLXArray?) -> Float { let d = w - t * r.expandedDimensions(axis: 1) * (c ?? MLXArray(Float(1))); return (d * d).sum().item(Float.self) }
            let key = parts.drop { $0 != "layers" }.dropFirst(2).joined(separator: ".")
            let e: [Float] = [err(t0, s0, nil), err(t0, sls, nil), err(t2, s2, c2), (w * w).sum().item(Float.self)]
            recon[key] = zip(recon[key] ?? [0, 0, 0, 0], e).map { $0 + $1 }
            (t, s, c) = (t2, s2, c2)
        }
        if let ct = ckpt[path + ".trits"] { t = ct.asType(.float32); s = ckpt[path + ".scale"]!; c = ckpt[path + ".colscale"] ?? c }
        zeroTrits += (t .== 0).sum().item(Int.self)
        var tr: Trits?
        if training {
            var acc = residual ? clip(kappa * 2 * theta * (w / (s.expandedDimensions(axis: 1) * (c ?? MLXArray(Float(1)))) - t), min: -0.999 * theta, max: 0.999 * theta) : MLXArray.zeros(like: w)
            if int8Acc { acc = round(acc * (127 / theta)).asType(.int8) }  // residual init in int8 units of theta/127
            if let a = opt["acc/" + path] { acc = a }
            tr = Trits(trits: t, acc: acc)
            if let r = opt["rows/" + path], let c = opt["cols/" + path] { tr!.rows = r; tr!.cols = c }
            tritState[path + ".weight"] = tr
        }
        let l = TernaryLinear(trits: tr?.value ?? t, scale: s, colscale: c)
        eval(l)
        replaced.append((path, l))
    }
    student.update(modules: ModuleChildren.unflattened(replaced))
    student.update(parameters: student.parameters().mapValues { $0.asType(.float32) })  // norms, embedding, conv taps; trits are exact
    if !ckpt.isEmpty {
        student.update(parameters: ModuleParameters.unflattened(ckpt.filter { !$0.key.hasSuffix(".trits") }.map { ($0.key, $0.value.asType(.float32)) }))
    }
    eval(student)
    if colscale {
        print("reconstruction ||W - What||/||W||  (twn | twn + LS row scale | two-sided)")
        for (k, e) in recon.sorted(by: { $0.key < $1.key }) {
            let q = e[0 ... 2].map { sqrt($0 / e[3]) }
            print(String(format: "  %-24@ %.4f | %.4f | %.4f", k as NSString, q[0], q[1], q[2]))
            emit("{\"type\":\"recon\",\"layer_type\":\"\(k)\",\"twn\":\(q[0]),\"twn_ls\":\(q[1]),\"two_sided\":\(q[2])}")
        }
        let e = recon.values.reduce([0, 0, 0, 0]) { zip($0, $1).map { $0 + $1 } }, q = e[0 ... 2].map { sqrt($0 / e[3]) }
        print(String(format: "  %-24@ %.4f | %.4f | %.4f", "all" as NSString, q[0], q[1], q[2]))
        emit("{\"type\":\"recon\",\"layer_type\":\"all\",\"twn\":\(q[0]),\"twn_ls\":\(q[1]),\"two_sided\":\(q[2])}")
    }
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
        emit("{\"type\":\"measure\",\"init\":\"\(twn ? "twn" : "absmean")\",\"colscale\":\(colscale),\"keep_fp\":\"\(keepArg)\",\"kl\":\(num(h.kl)),\"alpha\":\(num(h.alpha)),\"ppl\":\(num(ppl)),\"teacher_ppl\":\(num(tppl)),\"zero_fraction\":\(Double(zeroTrits) / Double(max(ternaryCount, 1)))}")
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
    for (k, a) in opt { if k.hasPrefix("m/") { m[String(k.dropFirst(2))] = a } else if k.hasPrefix("v/") { v[String(k.dropFirst(2))] = a } }
    let wsd = (option("schedule") ?? "cosine") == "wsd", decayFrac = Float(option("decay-frac") ?? "0.15")!
    let saveDir = option("save").map { URL(filePath: $0) }, saveEvery = Int(option("save-every") ?? "2048000")!
    var nextSave = (resumeTokens / saveEvery + 1) * saveEvery
    /// Atomic checkpoint: tmp dir, then rename over DIR. Trits as int8 [N,K], every other parameter under its module name.
    func save(_ step: Int, _ tokens: Int) throws {
        guard let dir = saveDir else { return }
        var model: [String: MLXArray] = [:], optim: [String: MLXArray] = [:]
        for (k, p) in student.parameters().flattened() {
            if let tr = tritState[k] { model[String(k.dropLast(7)) + ".trits"] = tr.value.asType(.int8); optim["acc/" + String(k.dropLast(7))] = tr.acc
                optim["rows/" + String(k.dropLast(7))] = tr.rows; optim["cols/" + String(k.dropLast(7))] = tr.cols } else { model[k] = p }
        }
        for (k, a) in m { optim["m/" + k] = a }
        for (k, a) in v { optim["v/" + k] = a }
        let fm = FileManager.default, tmp = dir.deletingLastPathComponent().appending(path: dir.lastPathComponent + ".tmp")
        try? fm.removeItem(at: tmp); try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        try Data(contentsOf: baseDir.appending(path: "config.json")).write(to: tmp.appending(path: "config.json"))  // resolves the hub symlink
        try MLX.save(arrays: model, url: tmp.appending(path: "model.safetensors"))
        try MLX.save(arrays: optim, metadata: ["step": String(step), "tokens": String(tokens)], url: tmp.appending(path: "optim.safetensors"))
        let old = dir.deletingLastPathComponent().appending(path: dir.lastPathComponent + ".old")
        try? fm.removeItem(at: old)
        if fm.fileExists(atPath: dir.path) { try fm.moveItem(at: dir, to: old) }
        try fm.moveItem(at: tmp, to: dir); try? fm.removeItem(at: old)
        print("checkpoint \(dir.path) at \(tokens) tokens")
    }

    // Layerwise reconstruction: each ternary layer, run on the teacher's input, matches the teacher layer's output. Taps record the
    // teacher's layer inputs and the student's residual h; the student's w2 output is rewritten to (teacher_out - h) so that every
    // layer sees the teacher's stream, with the +h and -h gradient paths cancelling (layers stay independent).
    var layerwiseSeconds = 0.0, layerwiseTokens = 0
    if lwTokens > 0 {
        precondition(keep.isEmpty && !ste, "--layerwise needs all layers ternary (no --keep-fp / --ste)")
        let lwStart = Date()
        func taps(_ model: Module, _ names: Set<String>) -> [Tap] {
            var found: [(Int, Tap)] = [], swaps: [(String, Module)] = []
            for (path, m) in model.leafModules().flattened() {
                guard let n = m as? RMSNorm, let name = path.split(separator: ".").last, names.contains(String(name)) else { continue }
                let tap = Tap(n); eval(tap); found.append((path.split(separator: ".").compactMap { Int($0) }.first ?? 1000, tap))
                swaps.append((path, tap))
            }
            model.update(modules: ModuleChildren.unflattened(swaps))
            return found.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
        let opTaps = taps(teacher, ["operator_norm", "embedding_norm"]), ffnTaps = taps(student, ["ffn_norm"])
        let layers = ffnTaps.count  // opTaps = operator_norm 0..<layers, then embedding_norm
        for tp in opTaps + ffnTaps { tp.on = true }
        func capture(_ t: MLXArray) -> [MLXArray] {  // teacher layer outputs = next layer's input
            _ = teacher(t[.newAxis], cache: nil)
            let outs = (1 ... layers).map { opTaps[$0].last!.asType(.float32) }
            eval(outs); for tp in opTaps { tp.last = nil }
            return outs
        }
        var losses = [MLXArray](repeating: MLXArray(Float(0)), count: layers), targets: [MLXArray] = []
        for (path, m) in replaced where path.hasSuffix("feed_forward.w2") {
            let i = path.split(separator: ".").compactMap { Int($0) }.first!, l = m as! TernaryLinear
            l.hook = { y in
                let h = ffnTaps[i].last!, t = targets[i], d = h + y - t
                losses[i] = (d * d).mean() / (t * t).mean()
                return stopGradient(t) - h
            }
        }
        func relMse(_ t: MLXArray) -> [Float] {
            targets = capture(t); _ = student(t[.newAxis], cache: nil)
            let l = stacked(losses); eval(l); return l.asArray(Float.self)
        }
        func evalLayers() -> [Float] {
            let r = evalSet.prefix(4).map { relMse($0) }
            return (0 ..< layers).map { i in r.map { $0[i] }.reduce(0, +) / Float(r.count) }
        }
        let before = evalLayers()
        let lwGrad = valueAndGrad(model: student) { model, a in
            targets = Array(a[1...]); _ = (model as! any LLMModel)(a[0][.newAxis], cache: nil)
            let l = stacked(losses); return [l.sum(), l]
        }
        MLXRandom.seed(UInt64(seed) &+ 1)
        let n = lwTokens / T, warmup = max(1, n / 20)
        print("layerwise: \(n) updates of \(T) tokens over \(layers) layers")
        var lm: [String: MLXArray] = [:], lv: [String: MLXArray] = [:], lwPool: [[Int32]] = [], recent = [Float](repeating: 0, count: layers), count = 0
        for step in 1 ... n {
            if lwPool.isEmpty { lwPool = try generate(poolSize).tokens }
            let tokens = MLXArray(lwPool.removeLast()), outs = capture(tokens)
            let (loss, grads) = lwGrad(student, [tokens] + outs)
            let cosine = 0.5 * (1 + cos(Float.pi * max(0, Float(step - warmup)) / Float(max(1, n - warmup))))
            let rhoNow = (step < warmup ? 0.25 + 0.75 * Float(step) / Float(warmup) : 0.2 + 0.8 * cosine) * rhoScale
            let rate = lr * (step < warmup ? Float(step) / Float(warmup) : 0.1 + 0.9 * cosine)
            let b1 = pow(0.9, Float(step)), b2 = pow(0.999, Float(step))
            let current = Dictionary(uniqueKeysWithValues: student.trainableParameters().flattened())
            var fresh: [(String, MLXArray)] = [], moved = MLXArray(Float(0)), toEval: [MLXArray] = [loss[0], loss[1]]
            for (k, g) in tieColscale(grads.flattened()) where k.contains("layers.") {
                if let tr = tritState[k] { moved = moved + tr.update(g: g, rho: rhoNow, leak: 1 - 1 / tau, theta: theta, step: step); toEval.append(tr.acc); continue }
                let mk = 0.9 * (lm[k] ?? zeros(like: g)) + 0.1 * g, vk = 0.999 * (lv[k] ?? zeros(like: g)) + 0.001 * g * g
                lm[k] = mk; lv[k] = vk
                let p = current[k]! - rate * (mk / (1 - b1)) / (sqrt(vk / (1 - b2)) + 1e-8)
                fresh.append((k, p)); toEval += [p, mk, vk]
            }
            toEval.append(moved); eval(toEval)
            student.update(parameters: ModuleParameters.unflattened(fresh))
            for (path, mod) in replaced { if let l = mod as? TernaryLinear { l.update(parameters: ModuleParameters.unflattened([("weight", tritState[path + ".weight"]!.value)])) } }
            layerwiseTokens += T
            recent = zip(recent, loss[1].asArray(Float.self)).map { $0 + $1 }; count += 1
            if step % 25 == 0 || step == 1 {
                print(String(format: "layerwise update %5d  tokens %8d  mean rel mse %.4f", step, layerwiseTokens, recent.reduce(0, +) / Float(count * layers)))
                recent = [Float](repeating: 0, count: layers); count = 0
            }
        }
        let after = evalLayers()
        for tp in opTaps + ffnTaps { tp.on = false; tp.last = nil }
        for (_, m) in replaced { (m as? TernaryLinear)?.hook = nil }
        for tr in tritState.values { tr.rows = MLXArray.zeros([1, tr.n]); tr.cols = MLXArray.zeros([1, tr.k]) }  // fresh second moments for the e2e step counter
        layerwiseSeconds = Date().timeIntervalSince(lwStart)
        let h = heldOut(), ppl = perplexity(student)
        let f = { (a: [Float]) in a.map { String(format: "%.4f", $0) }.joined(separator: " ") }
        print("layerwise rel mse per layer before: " + f(before) + "\n                              after:  " + f(after))
        print(String(format: "layerwise handover: %d tokens, %.1f s; held-out kl %.4f alpha %.4f ppl %.2f", layerwiseTokens, layerwiseSeconds, h.kl, h.alpha, ppl))
        emit("{\"type\":\"layerwise\",\"tokens\":\(layerwiseTokens),\"seconds\":\(layerwiseSeconds),\"rel_mse_before\":[\(before.map { num($0) }.joined(separator: ","))],\"rel_mse_after\":[\(after.map { num($0) }.joined(separator: ","))],\"kl\":\(num(h.kl)),\"alpha\":\(num(h.alpha)),\"ppl\":\(num(ppl))}")
    }

    MLXRandom.seed(UInt64(seed + resumeStep))
    var pool: [([Int32], Int, Int)] = [], steps: [MLXArray] = []
    var tokensSeen = resumeTokens, nextRound = (resumeTokens / roundTokens + 1) * roundTokens, nextPpl = (resumeTokens / pplEvery + 1) * pplEvery, recent = 0.0, recentAlpha = 0.0, recentCount = 0
    var flipsRecent: Float = 0, flipsTotal: Float = 0, generation = 0.0, working = 0.0
    let started = Date(), profile = CommandLine.arguments.contains("--profile")
    var tFwd = 0.0, tUpd = 0.0  // --profile: forward+backward vs TTF+Adam+student refresh (evaluated separately)
    func report(_ tokens: Int, withPpl: Bool) {
        let h = heldOut(), ppl = withPpl ? perplexity(student) : Float.nan
        let p = withPpl ? ",\"ppl\":\(num(ppl))" : ""
        emit("{\"type\":\"eval\",\"tokens\":\(tokens),\"member\":0,\"kl\":\(num(h.kl)),\"alpha\":\(num(h.alpha)),\"lr\":\(lr)\(p),\"flips_total\":\(flipsTotal)}")
        print(String(format: "eval at %d tokens: held-out kl %.4f alpha %.4f", tokens, h.kl, h.alpha) + (withPpl ? String(format: "  ppl %.2f", ppl) : "") + String(format: "  flips total %.0f", flipsTotal))
    }
    report(resumeTokens, withPpl: true)
    for step in (resumeStep + 1) ... max(updates, resumeStep + 1) {
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
        if profile { eval(loss, grads.flattened().map { $0.1 }); tFwd += Date().timeIntervalSince(t0) }
        let t1 = Date()
        let decayStart = max(warmup, Int(Float(updates) * (1 - decayFrac)))
        let cosine = wsd ? 1 - max(0, min(1, Float(step - decayStart) / Float(max(1, updates - decayStart)))) : 0.5 * (1 + cos(Float.pi * max(0, Float(step - warmup)) / Float(max(1, updates - warmup))))
        let rhoNow = (step < warmup ? 0.25 + 0.75 * Float(step) / Float(warmup) : 0.2 + 0.8 * cosine) * rhoScale
        let rate = lr * (step < warmup ? Float(step) / Float(warmup) : 0.1 + 0.9 * cosine)
        let b1 = pow(0.9, Float(step)), b2 = pow(0.999, Float(step))
        let current = Dictionary(uniqueKeysWithValues: student.trainableParameters().flattened())
        var fresh: [(String, MLXArray)] = [], moved = MLXArray(Float(0)), toEval: [MLXArray] = [loss[0], loss[1]]
        for (k, g) in tieColscale(grads.flattened()) {
            if let tr = tritState[k] { moved = moved + tr.update(g: g, rho: rhoNow, leak: 1 - 1 / tau, theta: theta, step: step); toEval.append(tr.acc); continue }
            let mk = 0.9 * (m[k] ?? zeros(like: g)) + 0.1 * g, vk = 0.999 * (v[k] ?? zeros(like: g)) + 0.001 * g * g
            m[k] = mk; v[k] = vk
            let p = current[k]! - rate * (mk / (1 - b1)) / (sqrt(vk / (1 - b2)) + 1e-8)
            fresh.append((k, p)); toEval += [p, mk, vk]
        }
        toEval.append(moved)
        eval(toEval)
        if profile { tUpd += Date().timeIntervalSince(t1) }
        student.update(parameters: ModuleParameters.unflattened(fresh))
        for (path, mod) in replaced { if let l = mod as? TernaryLinear { l.update(parameters: ModuleParameters.unflattened([("weight", tritState[path + ".weight"]!.value)])) } }
        working += Date().timeIntervalSince(t0)
        let f = moved.item(Float.self); flipsRecent += f; flipsTotal += f
        recent += Double(loss[0].item(Float.self)); recentAlpha += Double(loss[1].item(Float.self)); recentCount += 1
        if step % 25 == 0 || step == 1 {
            let kl = Float(recent / Double(recentCount)), al = Float(recentAlpha / Double(recentCount)), fl = flipsRecent / Float(recentCount)
            print(String(format: "update %5d  tokens %8d  train kl %.3f alpha %.3f  flips %.0f  %.0f tok/s", step, tokensSeen, kl, al, fl, Double(tokensSeen) / Date().timeIntervalSince(started)))
            emit("{\"type\":\"train\",\"update\":\(step),\"tokens\":\(tokensSeen),\"kl\":[\(num(kl))],\"alpha\":[\(num(al))],\"flips\":[\(num(fl))]}")
            flipsRecent = 0; recent = 0; recentAlpha = 0; recentCount = 0
        }
        if tokensSeen >= nextSave || step == updates { if tokensSeen >= nextSave { nextSave += saveEvery }; try save(step, tokensSeen) }
        if tokensSeen >= nextRound || step == updates {
            nextRound += roundTokens
            let withPpl = step == updates || tokensSeen >= nextPpl
            if withPpl { while nextPpl <= tokensSeen { nextPpl += pplEvery } }
            report(tokensSeen, withPpl: withPpl)
        }
    }
    if profile { let n = Double(tokensSeen / T); print(String(format: "profile per update (s): generation %.3f  forward+backward %.3f  TTF+Adam %.3f  total working %.3f", generation / n, tFwd / n, tUpd / n, working / n)) }
    print(String(format: "pool generation %.1f ms per sequence; throughput excluding generation %.0f tok/s, including %.0f tok/s; peak memory %.1f GB",
                 1000 * generation / Double(tokensSeen / T), Double(tokensSeen) / working, Double(tokensSeen) / Date().timeIntervalSince(started), Double(Memory.peakMemory) / 1e9))
    emit("{\"type\":\"done\",\"layerwise_tokens\":\(layerwiseTokens),\"layerwise_seconds\":\(layerwiseSeconds),\"tokens\":\(tokensSeen),\"seconds\":\(Date().timeIntervalSince(started)),\"state_bytes_per_ternary_param\":\(perTernary),\"peak_gb\":\(Double(Memory.peakMemory) / 1e9)}")
}
try await run()
