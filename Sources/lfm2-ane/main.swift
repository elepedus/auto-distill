// Stage 2: one ternary LFM2 layer on the ANE against an MLX reference, sharing memory zero-copy.
//   lfm2-ane TEXTFILE [LAYER...]   (default layers: 0 conv, 2 attention, 14 attention, 15 conv)
// Inputs are real hidden states of bf16 LFM2-350M on the text. The MLX reference reads the ANE's
// own input and output IOSurfaces, so both engines see exactly the same bytes.
import ANE
import Foundation
import IOSurface
import LMLoad
import MLX
import MLXNN
import Tokenizers

let T = 256
let text = try String(contentsOf: URL(filePath: CommandLine.arguments[1]), encoding: .utf8)
let layers = CommandLine.arguments.dropFirst(2).compactMap(Int.init)
let (model, dir) = try await loadLLM("LiquidAI/LFM2-350M")
let tokenizer = try await AutoTokenizer.from(modelFolder: dir)
let p = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) })
let config = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appending(path: "config.json"))) as! [String: Any]
let attentionLayers = Set(config["full_attn_idxs"] as! [Int])
let ff = p["model.layers.0.feed_forward.w1.weight"]!.dim(0)
let shape = LFM2Shape(tokens: T, dim: 1024, ff: ff, heads: 16, headDim: 64, ropeBase: Float(config["rope_theta"] as! Double))
let ids = [Int32(tokenizer.bosTokenId ?? 1)] + tokenizer.encode(text: text, addSpecialTokens: false).prefix(T - 1).map(Int32.init)

// MLX reference layer. `w(name)` is the effective [out, in] weight; `x` is [1, T, D].
func referenceLayer(_ i: Int, _ x: MLXArray, _ w: (String) -> MLXArray, _ v: (String) -> MLXArray) -> (y: MLXArray, r: MLXArray, m: MLXArray) {
    let pre = "model.layers.\(i)."
    func lin(_ z: MLXArray, _ name: String) -> MLXArray { matmul(z, w(pre + name).T) }
    let xn = MLXFast.rmsNorm(x, weight: v(pre + "operator_norm.weight"), eps: 1e-5)
    let r: MLXArray
    if attentionLayers.contains(i) {
        func heads(_ z: MLXArray, _ norm: String) -> MLXArray {
            let z = z.reshaped(1, T, -1, 64)
            return (norm.isEmpty ? z : MLXFast.rmsNorm(z, weight: v(pre + norm), eps: 1e-5)).transposed(0, 2, 1, 3)
        }
        let rope = { (z: MLXArray) in MLXFast.RoPE(z, dimensions: 64, traditional: false, base: shape.ropeBase, scale: 1, offset: 0) }
        let q = rope(heads(lin(xn, "self_attn.q_proj.weight"), "self_attn.q_layernorm.weight"))
        let k = rope(heads(lin(xn, "self_attn.k_proj.weight"), "self_attn.k_layernorm.weight"))
        let vv = heads(lin(xn, "self_attn.v_proj.weight"), "")
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: vv, scale: 0.125, mask: .causal)
        r = lin(o.transposed(0, 2, 1, 3).reshaped(1, T, -1), "self_attn.out_proj.weight")
    } else {
        let bcx = lin(xn, "conv.in_proj.weight").split(parts: 3, axis: -1)
        let bx = concatenated([MLXArray.zeros([1, 2, 1024]), bcx[0] * bcx[2]], axis: 1)
        let kernel = v(pre + "conv.conv.weight")  // [D, 3, 1]
        let taps = (0 ..< 3).map { j in bx[0..., j ..< j + T] * kernel[0..., j, 0] }
        let conv = taps[0] + taps[1] + taps[2]
        r = lin(bcx[1] * conv, "conv.out_proj.weight")
    }
    let h = x + r
    let hn = MLXFast.rmsNorm(h, weight: v(pre + "ffn_norm.weight"), eps: 1e-5)
    let m = lin(silu(lin(hn, "feed_forward.w1.weight")) * lin(hn, "feed_forward.w3.weight"), "feed_forward.w2.weight")
    return (h + m, r, m)
}

// Full bf16 forward with our reference layers, checked against mlx-swift-lm's own logits.
var hidden = [p["model.embed_tokens.weight"]![MLXArray(ids)][.newAxis]]
for i in 0 ..< 16 { hidden.append(referenceLayer(i, hidden[i], { p[$0]! }, { p[$0]! }).y) }
let logits = matmul(MLXFast.rmsNorm(hidden[16], weight: p["model.embedding_norm.weight"]!, eps: 1e-5), p["model.embed_tokens.weight"]!.T)
let official = model(MLXArray(ids)[.newAxis], cache: nil).asType(.float32)
func relL2(_ a: MLXArray, _ b: MLXArray) -> Float { (sqrt(((a - b) * (a - b)).sum()) / sqrt((b * b).sum())).item(Float.self) }
print(String(format: "reference vs mlx-swift-lm logits: rel L2 %.4f, argmax agreement %.1f%%", relL2(logits, official),
             100 * (logits.argMax(axis: -1) .== official.argMax(axis: -1)).asType(.float32).mean().item(Float.self)))
print("residual-stream max|h| per layer input: " + hidden.map { String(format: "%.0f", abs($0).max().item(Float.self)) }.joined(separator: " "))

/// Ternary in BitNet form: per output row, scale = mean|w|, trits = clip(round(w / scale), -1, 1).
func ternary(_ w: MLXArray) -> (trits: MLXArray, scale: MLXArray) {
    let s = abs(w).mean(axis: 1, keepDims: true)
    return (clip(round(w / s), min: -1, max: 1), s.reshaped(-1))
}

/// Zero-copy MLX view of an ANE IOSurface binding, honouring the compiler's strides.
func view(_ b: SurfaceBinding) -> MLXArray {
    let flat = MLXArray(rawPointer: IOSurfaceGetBaseAddress(b.surface), [IOSurfaceGetAllocSize(b.surface) / 2], dtype: .float16) { _ = b }
    let l = b.layout
    return asStrided(flat, [l.shape.batches, l.shape.channels, l.shape.height, l.shape.width],
                     strides: [l.batchStride / 2, l.planeStride / 2, l.rowStride / 2, 1], offset: b.startOffset / 2)
}
func tensor(_ a: MLXArray, _ dims: [Int]) throws -> Tensor {
    try Tensor(shape: TensorShape(batches: dims[0], channels: dims[1], height: dims[2], width: dims[3]),
               values: a.asType(.float16).asArray(Float16.self))
}

for i in layers.isEmpty ? [0, 2, 14, 15] : layers {
    let attention = attentionLayers.contains(i), pre = "model.layers.\(i)."
    let started = Date()
    let (program, constants) = try layerProgram(shape, attention: attention)
    let ane = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 2 << 30))
    defer { try? ane.close() }

    // Effective weights per ANE input name: (MLX weight key, rows to take, duplicate kv heads?)
    var weights: [String: String] = ["w1": "feed_forward.w1.weight", "w3": "feed_forward.w3.weight", "w2": "feed_forward.w2.weight"]
    if attention {
        weights.merge(["q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight", "o": "self_attn.out_proj.weight"]) { $1 }
    } else {
        weights.merge(["b": "conv.in_proj.weight", "c": "conv.in_proj.weight", "x": "conv.in_proj.weight", "o": "conv.out_proj.weight"]) { $1 }
    }
    var inputs: [String: Tensor] = constants.merging(["x": try tensor(hidden[i], [1, 1, T, 1024]),
                                    "g_op": try tensor(p[pre + "operator_norm.weight"]!, [1, 1, 1, 1024]),
                                    "g_ffn": try tensor(p[pre + "ffn_norm.weight"]!, [1, 1, 1, 1024])]) { $1 }
    for (name, key) in weights {
        var w = p[pre + key]!
        if !attention, let part = ["b": 0, "c": 1, "x": 2][name] { w = w[(part * 1024) ..< ((part + 1) * 1024)] }
        if attention, name == "k" || name == "v" {  // GQA: repeat each of the 8 kv heads twice
            w = repeated(w.reshaped(8, 1, 64, 1024), count: 2, axis: 1).reshaped(1024, 1024)
        }
        let (trits, scale) = ternary(w)
        inputs["w_\(name)"] = try tensor(trits, [1, 1, w.dim(0), w.dim(1)])
        inputs["s_\(name)"] = try tensor(scale, [1, 1, 1, w.dim(0)])
    }
    if attention {
        for n in ["q", "k"] { inputs["g_\(n)"] = try tensor(tiled(p[pre + "self_attn.\(n)_layernorm.weight"]!, repetitions: [16]), [1, 16, 1, 64]) }
    } else {
        for j in 0 ..< 3 { inputs["k\(j)"] = try tensor(p[pre + "conv.conv.weight"]![0..., j, 0], [1, 1, 1, 1024]) }
    }
    _ = try ane.prepare(inputs: inputs)  // the one upload; from here on both engines share these surfaces
    let compiled = Date()
    let surfaces = try ane.surfaceBindings()
    let observation = try ane.evaluatePrepared(readOutputs: false, poisonOutputs: true)
    func shared(_ name: String) -> MLXArray { view(surfaces.first { $0.name == name || $0.name.hasPrefix(name + "@") }!).asType(.float32) }

    // Reference from the same bytes: effective weight = trits * scale, rebuilt from the shared surfaces.
    let x = shared("x").reshaped(1, T, 1024)
    func effective(_ key: String) -> MLXArray {
        let name = key.split(separator: ".").dropLast().last.map(String.init) ?? ""
        let short = ["w1": "w1", "w3": "w3", "w2": "w2", "q_proj": "q", "k_proj": "k", "v_proj": "v", "out_proj": "o"][name]
        if let short { let w = shared("w_\(short)").squeezed() * shared("s_\(short)").reshaped(-1, 1)
            return attention && (short == "k" || short == "v") ? w.reshaped(8, 2, 64, 1024)[0..., 0].reshaped(512, 1024) : w }
        return concatenated(["b", "c", "x"].map { shared("w_\($0)").squeezed() * shared("s_\($0)").reshaped(-1, 1) }, axis: 0)  // in_proj
    }
    func vector(_ key: String) -> MLXArray {
        switch key.split(separator: ".").suffix(2).joined(separator: ".") {
        case "operator_norm.weight": shared("g_op").reshaped(-1)
        case "ffn_norm.weight": shared("g_ffn").reshaped(-1)
        case "q_layernorm.weight": shared("g_q")[0, 0, 0]
        case "k_layernorm.weight": shared("g_k")[0, 0, 0]
        default: stacked((0 ..< 3).map { shared("k\($0)").reshaped(-1) }, axis: 1)[0..., 0..., .newAxis]  // conv kernel [D, 3, 1]
        }
    }
    let ref = referenceLayer(i, x, effective, vector)
    var report = String(format: "layer %2d %@  compile %.1fs  eval %.2f ms |", i, attention ? "attn" : "conv",
                        compiled.timeIntervalSince(started), observation.elapsed * 1000)
    for (name, want) in [("r", ref.r), ("m", ref.m), ("y", ref.y)] {
        let got = shared(name).reshaped(1, T, 1024)
        let nonfinite = (isNaN(got).asType(.int32) + (abs(got) .> 65504).asType(.int32)).sum().item(Int.self)
        report += String(format: "  %@: rel L2 %.5f, max|ref| %.3g, nonfinite %d", name, relL2(got, want),
                         abs(want).max().item(Float.self), nonfinite)
    }
    print(report)
}
