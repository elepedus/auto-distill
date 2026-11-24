// Exact export of a ternary LFM2-350M checkpoint as an MLX 2-bit affine-quantized model, plus parity and GPU benchmarks.
//   ternary-export export CKPT OUT [--embed-bits 4]   write OUT/config.json + model.safetensors, loadable by mlx-swift-lm as is
//   ternary-export quant4 OUT                         standard MLX 4-bit (group 64) quantization of the base, for reference
//   ternary-export parity CKPT MODEL [--seqs 4]       MODEL (standard load) vs CKPT evaluated as trits*scale in fp32
//   ternary-export bench MODEL [--reps 3]             prefill (T=256, B=1 and 8) and greedy decode (256 tokens) tok/s; MODEL may be a HF id
//
// Format. Linear weight W[n,k] = scale[n] * trits[n,k] becomes MLX affine 2-bit, group 64: w = s*q + b with q = t+1 in {0,1,2},
// s = row scale, b = -s for every group of the row. Storage: weight uint32 [n, k/16], scales/biases bf16 [n, k/64].
// Input-channel scales (colscale c, layer computes (x*c) @ (trits*scale)^T) are folded exactly:
//   * q/k/v share an input: their colscales must be identical; folded into operator_norm.weight.   (RMSNorm out = w * x/rms)
//   * conv in_proj colscale -> operator_norm.weight.
//   * w1/w3 share an input: identical colscales; folded into ffn_norm.weight.
//   * conv out_proj colscale c[d] -> scale of in_proj row D+d (the C chunk; in_proj chunks are B, C, x and y = C * conv(B*x), conv
//     is depthwise so channel d of y is linear in C[d]).
//   * attention out_proj colscale c[h*64+d] -> scale of v_proj row g*64+d (g = h/2, GQA 16 heads over 8 kv heads). Attention is linear
//     in V, so exact iff the two query heads sharing a kv head have equal colscale; else fail.
//   * w2 colscale c[j] -> scale of w3 row j (h = silu(w1 x) * (w3 x), w2 input is h).
// Row scales just multiply. Everything is multiplied in fp32 and stored bf16 (scale and bias get the same bf16 value, so the
// dequantized weight is exactly +-scale_bf16 or 0).
import Foundation
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
setvbuf(stdout, nil, _IOLBF, 0)

func option(_ name: String) -> String? { CommandLine.arguments.firstIndex(of: "--" + name).map { CommandLine.arguments[$0 + 1] } }
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
func fail(_ s: String) -> Never { print("error: " + s); exit(1) }
let T = 256, D = 1024
let args = CommandLine.arguments

/// Packs integers q [n, k] in 0..<4 (2 bits, little-endian within a uint32, 16 per word) as MLX's affine packing.
func pack2(_ q: MLXArray) -> MLXArray {
    let n = q.dim(0), k = q.dim(1)
    let shifts = MLXArray((0 ..< 16).map { UInt32($0 * 2) })
    return (q.asType(.uint32).reshaped(n, k / 16, 16) << shifts).sum(axis: -1).asType(.uint32)
}

func copyConfig(from dir: URL, to out: URL, quantization: [String: Any]?) throws {
    var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appending(path: "config.json"))) as! [String: Any]
    if let quantization { cfg["quantization"] = quantization }
    try JSONSerialization.data(withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys]).write(to: out.appending(path: "config.json"))
}

func exportCommand() async throws {
    let ckptDir = URL(filePath: args[2]), out = URL(filePath: args[3])
    let embedBits = Int(option("embed-bits") ?? "0")!
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let (base, _) = try await loadLLM("LiquidAI/LFM2-350M")
    let baseParams = Dictionary(uniqueKeysWithValues: base.parameters().flattened())
    let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: ckptDir.appending(path: "config.json"))) as! [String: Any]
    let attn = Set(cfg["full_attn_idxs"] as! [Int]), layers = cfg["num_hidden_layers"] as! Int
    let a = try loadArrays(url: ckptDir.appending(path: "model.safetensors"))
    var res: [String: MLXArray] = [:]
    var handled = Set<String>()
    func f32(_ k: String) -> MLXArray? { a[k]?.asType(.float32) }
    func tied(_ names: [String]) -> MLXArray? {  // identical colscale of linears sharing an input, or nil when none has one
        let cs = names.map { f32($0 + ".colscale") }
        guard let first = cs.compactMap({ $0 }).first else { return nil }
        for (n, c) in zip(names, cs) {
            guard let c, c.shape == first.shape, (c .== first).all().item(Bool.self)
            else { fail("colscales of \(names) are not identical (\(n)); cannot fold into the shared norm exactly") }
        }
        return first
    }
    var rowScale: [String: MLXArray] = [:], trits: [String: MLXArray] = [:]
    var paths: [String] = []
    for i in 0 ..< layers {
        let pre = "model.layers.\(i)."
        let names = (attn.contains(i) ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"] : ["conv.in_proj", "conv.out_proj"])
            + ["feed_forward.w1", "feed_forward.w3", "feed_forward.w2"]
        for n in names {
            let p = pre + n
            guard let t = a[p + ".trits"], let s = f32(p + ".scale") else { fail("checkpoint lacks \(p).trits/.scale") }
            trits[p] = t; rowScale[p] = s; paths.append(p)
            handled.formUnion([p + ".trits", p + ".scale", p + ".colscale", p + ".weight"])
        }
        var opNorm = f32(pre + "operator_norm.weight") ?? baseParams[pre + "operator_norm.weight"]!.asType(.float32)
        var ffnNorm = f32(pre + "ffn_norm.weight") ?? baseParams[pre + "ffn_norm.weight"]!.asType(.float32)
        if attn.contains(i) {
            if let c = tied(["q_proj", "k_proj", "v_proj"].map { pre + "self_attn." + $0 }) { opNorm = opNorm * c }
            if let c = f32(pre + "self_attn.out_proj.colscale") {
                let g = c.reshaped(8, 2, 64)  // [kv head, query head within the pair, d]
                guard (g[0..., 0] .== g[0..., 1]).all().item(Bool.self) else { fail("\(pre)self_attn.out_proj colscale differs between query heads sharing a kv head; cannot fold into v_proj") }
                rowScale[pre + "self_attn.v_proj"] = rowScale[pre + "self_attn.v_proj"]! * g[0..., 0].reshaped(-1)
            }
        } else {
            if let c = f32(pre + "conv.in_proj.colscale") { opNorm = opNorm * c }
            if let c = f32(pre + "conv.out_proj.colscale") {
                let s = rowScale[pre + "conv.in_proj"]!
                rowScale[pre + "conv.in_proj"] = concatenated([s[0 ..< D], s[D ..< 2 * D] * c, s[(2 * D)...]], axis: 0)
            }
        }
        if let c = tied(["w1", "w3"].map { pre + "feed_forward." + $0 }) { ffnNorm = ffnNorm * c }
        if let c = f32(pre + "feed_forward.w2.colscale") { rowScale[pre + "feed_forward.w3"] = rowScale[pre + "feed_forward.w3"]! * c }
        res[pre + "operator_norm.weight"] = opNorm.asType(.bfloat16); res[pre + "ffn_norm.weight"] = ffnNorm.asType(.bfloat16)
        handled.formUnion([pre + "operator_norm.weight", pre + "ffn_norm.weight"])
    }
    // Pack, and verify exactness against trits * (folded row scale) through MLX's own dequantize.
    var worstStore: Float = 0, mismatches = 0
    for p in paths {
        let t = trits[p]!, s = rowScale[p]!.asType(.bfloat16)
        let n = t.dim(0), k = t.dim(1), groups = k / 64
        let sb = broadcast(s.reshaped(n, 1), to: [n, groups])
        res[p + ".weight"] = pack2(t.asType(.int32) + 1); res[p + ".scales"] = sb; res[p + ".biases"] = -sb
        let deq = dequantized(res[p + ".weight"]!, scales: sb, biases: -sb, groupSize: 64, bits: 2).asType(.float32)
        let want = t.asType(.float32) * s.asType(.float32).reshaped(n, 1)  // exactly the bf16 row scale
        mismatches += (deq .!= want).sum().item(Int.self)
        let exact = t.asType(.float32) * rowScale[p]!.reshaped(n, 1)  // vs the unrounded fp32 scale
        worstStore = max(worstStore, (abs(deq - exact) / maximum(abs(exact), 1e-30)).max().item(Float.self))
    }
    // Cross-check pack2 against MLX quantized(): a group holding both 0 and 3 quantizes to scale -1, bias 3 (so q = 3 - value).
    let r = MLXRandom.randInt(low: 0, high: 4, [8, 128]).asType(.float32)
    let cols = MLXArray(Array(0 ..< 128)), keep = ((cols % 64) .> 1).reshaped(1, 128)
    let probe = MLX.where(keep, r, MLX.where((cols % 64 .== 0).reshaped(1, 128), MLXArray(Float(0)), MLXArray(Float(3))))
    let ref = quantized(probe, groupSize: 64, bits: 2)
    let packOK = (ref.wq .== pack2(3 - probe)).all().item(Bool.self)
    print("exactness: \(paths.count) linears, \(mismatches) elements where MLX-dequantized != trits * bf16(folded scale) (must be 0); max rel deviation of the bf16 scale from the fp32 scale \(worstStore) (bf16 eps 2^-8 = 0.0039); pack2 == MLX quantized() packing: \(packOK)")
    if mismatches != 0 || !packOK { fail("exactness check failed") }
    // Everything else, bf16.
    var embedQ: [String: Any] = [:]
    for (k, v) in baseParams where !handled.contains(k) {
        if a[k] == nil { print("note: \(k) not in checkpoint, using base") }
        let w = (a[k] ?? v).asType(.bfloat16)
        if embedBits > 0 && k == "model.embed_tokens.weight" {
            let q = quantized(w, groupSize: 64, bits: embedBits)
            res[k] = q.wq; res["model.embed_tokens.scales"] = q.scales; res["model.embed_tokens.biases"] = q.biases!
            embedQ["model.embed_tokens"] = ["group_size": 64, "bits": embedBits]
        } else { res[k] = w }
    }
    try save(arrays: res, url: out.appending(path: "model.safetensors"))
    try copyConfig(from: ckptDir, to: out, quantization: ["group_size": 64, "bits": 2].merging(embedQ) { $1 } as [String: Any])
    let size = try FileManager.default.attributesOfItem(atPath: out.appending(path: "model.safetensors").path)[.size] as! Int
    print("wrote \(res.count) tensors, \(Double(size) / 1e6) MB to \(out.path)")
}

func quant4Command() async throws {
    let out = URL(filePath: args[2])
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let (m, dir) = try await loadLLM("LiquidAI/LFM2-350M")
    quantize(model: m, groupSize: 64, bits: 4)
    try save(arrays: Dictionary(uniqueKeysWithValues: m.parameters().flattened()), url: out.appending(path: "model.safetensors"))
    try copyConfig(from: dir, to: out, quantization: ["group_size": 64, "bits": 4])
}

// ---- parity
final class Box { var a: MLXArray? }
final class TernaryLinear: Linear {  // after ternarise: (x * col) @ (trits * scale)^T in dtype dt
    @ParameterInfo var scale: MLXArray
    let col = Box(), dt: DType
    init(trits: MLXArray, scale: MLXArray, col: MLXArray?, dt: DType) { _scale.wrappedValue = scale; self.col.a = col; self.dt = dt; super.init(weight: trits, bias: nil) }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x.asType(dt)
        if let c = col.a { x = x * c.asType(dt) }
        return (matmul(x, weight.asType(dt).T) * scale.asType(dt)).asType(.float32)
    }
}

func parityCommand() async throws {
    let cmp = args[1] == "kl"  // kl MODEL REF: two standard-loaded models
    let a = cmp ? [:] : try loadArrays(url: URL(filePath: args[2]).appending(path: "model.safetensors"))
    let (exported, _) = try await loadLLM(cmp ? args[2] : args[3])
    func reference(_ dt: DType) async throws -> any LLMModel {
        let (m, _) = try await loadLLM("LiquidAI/LFM2-350M")
        var replaced: [(String, Module)] = []
        for k in a.keys where k.hasSuffix(".trits") {
            let p = String(k.dropLast(6))
            replaced.append((p, TernaryLinear(trits: a[k]!.asType(dt), scale: a[p + ".scale"]!.asType(.float32), col: a[p + ".colscale"]?.asType(.float32), dt: dt)))
        }
        m.update(modules: ModuleChildren.unflattened(replaced))
        var cur = Dictionary(uniqueKeysWithValues: m.parameters().flattened())
        for (k, v) in a where cur[k] != nil && !k.hasSuffix(".scale") { cur[k] = v.asType(.float32) }
        m.update(parameters: ModuleParameters.unflattened(cur))
        eval(m)
        return m
    }
    let ref32 = cmp ? try await loadLLM(args[3]).0 : try await reference(.float32), refBF = cmp ? ref32 : try await reference(.bfloat16)
    let tok = try await AutoTokenizer.from(modelFolder: try modelDirectory("LiquidAI/LFM2-350M"))
    let text = try String(contentsOf: URL(filePath: "/tmp/wikitext2-test-join.txt"), encoding: .utf8)
    let ids = tok.encode(text: text, addSpecialTokens: false).map(Int32.init)
    let nseq = Int(option("seqs") ?? "4")!
    func logits(_ m: any LLMModel, _ s: Int) -> MLXArray {
        let t = MLXArray([Int32(tok.bosTokenId ?? 1)] + Array(ids[(s * 4000) ..< (s * 4000 + T - 1)]))
        let l = m(t[.newAxis], cache: nil)[0].asType(.float32); eval(l); return l
    }
    func report(_ name: String, _ got: [MLXArray], _ want: [MLXArray]) {
        let g = concatenated(got, axis: 0), w = concatenated(want, axis: 0)
        let rel = (sqrt(((g - w) * (g - w)).sum()) / sqrt((w * w).sum())).item(Float.self)
        let agree = (g.argMax(axis: -1) .== w.argMax(axis: -1)).asType(.float32).mean().item(Float.self)
        let lg = logSoftmax(g, axis: -1), lw = logSoftmax(w, axis: -1)
        let kl = (exp(lw) * (lw - lg)).sum(axis: -1).mean().item(Float.self)
        print(String(format: "%@: rel-L2 %.5f, argmax agreement %.2f%%, KL(ref||model) %.6f nats/token (%d tokens)", name, rel, 100 * agree, kl, g.dim(0)))
    }
    let r32 = (0 ..< nseq).map { logits(ref32, $0) }, rbf = (0 ..< nseq).map { logits(refBF, $0) }, ex = (0 ..< nseq).map { logits(exported, $0) }
    report("exported (standard load, bf16 activations) vs fp32 ternary reference", ex, r32)
    report("noise floor: bf16-matmul ternary reference vs fp32 reference", rbf, r32)
    report("exported vs bf16-matmul ternary reference", ex, rbf)
}

// ---- benchmark
func median(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
func benchCommand() async throws {
    let (m, dir) = try await loadLLM(args[2])
    let reps = Int(option("reps") ?? "3")!
    let tok = try await AutoTokenizer.from(modelFolder: try modelDirectory("LiquidAI/LFM2-350M"))
    let text = try String(contentsOf: URL(filePath: "/tmp/wikitext2-test-join.txt"), encoding: .utf8)
    let ids = tok.encode(text: text, addSpecialTokens: false).prefix(T * 8).map(Int32.init)
    var res: [String] = []
    for b in [1, 8] {
        let x = MLXArray(Array(ids.prefix(b * T)), [b, T])
        for _ in 0 ..< 2 { eval(m(x, cache: nil)) }
        var rates: [Double] = []
        for _ in 0 ..< reps {
            let t0 = now(); for _ in 0 ..< 5 { eval(m(x, cache: nil)) }
            rates.append(Double(5 * b * T) / (now() - t0))
        }
        res.append(String(format: "prefill B=%d T=%d: %.0f tok/s (runs %@)", b, T, median(rates), rates.map { String(format: "%.0f", $0) }.joined(separator: " ")))
    }
    func decode(_ n: Int) throws -> Double {
        let cache = try m.newCache(parameters: nil)
        var y = m(MLXArray(Array(ids.prefix(16)), [1, 16]), cache: cache)[0..., -1, 0...].argMax(axis: -1)
        eval(y)
        let t0 = now()
        for _ in 0 ..< n { y = m(y.reshaped(1, 1), cache: cache)[0..., -1, 0...].argMax(axis: -1); eval(y) }
        return Double(n) / (now() - t0)
    }
    _ = try decode(16)
    let d = try (0 ..< reps).map { _ in try decode(256) }
    res.append(String(format: "decode B=1, 256 greedy tokens: %.1f tok/s (runs %@)", median(d), d.map { String(format: "%.1f", $0) }.joined(separator: " ")))
    var bytes = 0
    for f in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) where f.pathExtension == "safetensors" {
        bytes += try FileManager.default.attributesOfItem(atPath: f.resolvingSymlinksInPath().path)[.size] as! Int
    }
    res.append(String(format: "weights: %.1f MB", Double(bytes) / 1e6))
    print("bench \(args[2])\n  " + res.joined(separator: "\n  "))
}

switch args.dropFirst().first ?? "" {
case "export": try await exportCommand()
case "quant4": try await quant4Command()
case "parity", "kl": try await parityCommand()
case "bench": try await benchCommand()
default: print("usage: ternary-export export CKPT OUT [--embed-bits 4] | quant4 OUT | parity CKPT MODEL | bench MODEL")
}
