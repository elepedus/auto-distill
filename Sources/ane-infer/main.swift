// Whole ternarised LFM2-350M inference: embedding (GPU) -> 16 decoder layers (ANE, chained zero-copy) -> final norm + tied head (GPU).
//   ane-infer [--ckpt DIR] [--text FILE] [--seqs 4] [--reps 20]     parity vs the same ternary model in MLX + prefill benchmark
//   ane-infer write-ckpt DIR [--colscale [--tied]]                           write a TWN checkpoint of the base model (--colscale: random input-channel scales, for testing)
// Weights: base LFM2-350M ternarised with TWN, or --ckpt DIR (config.json + model.safetensors with <path>.trits int8 [N,K],
// <path>.scale fp32 [N], optional <path>.colscale fp32 [K]; every other parameter under its mlx-swift-lm name).
import ANE
import Foundation
import IOSurface
import LFM2ANE
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
setvbuf(stdout, nil, _IOLBF, 0)

func option(_ name: String) -> String? { CommandLine.arguments.firstIndex(of: "--" + name).map { CommandLine.arguments[$0 + 1] } }
let T = 256, D = 1024, HEADS = 16, HD = 64
nonisolated(unsafe) var matDtype = DType.float32  // dtype of the GPU ternary model's matmuls

/// Trits x row scale (x input-channel scale): (x ⊙ col) @ (trits ⊙ scale)ᵀ. After ternarise's TernaryLinear.
final class Box { var a: MLXArray? }
final class TernaryLinear: Linear {
    @ParameterInfo var scale: MLXArray
    let col = Box()
    init(trits: MLXArray, scale: MLXArray, col: MLXArray?) { _scale.wrappedValue = scale; self.col.a = col; super.init(weight: trits, bias: nil) }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dt = matDtype
        var x = x.asType(dt)
        if let c = col.a { x = x * c.asType(dt) }
        return (matmul(x, weight.asType(dt).T) * scale.asType(dt)).asType(.float32)
    }
}

/// w [n, k] fp32 -> (trits [n, k], scale [n]). Copied from ternarise's quantise(twn: true).
func quantise(_ w: MLXArray) -> (MLXArray, MLXArray) {
    let a = abs(w)
    let mask = (a .> 0.7 * a.mean(axis: 1, keepDims: true)).asType(.float32)
    let s = (a * mask).sum(axis: 1, keepDims: true) / maximum(mask.sum(axis: 1, keepDims: true), 1)
    return (sign(w) * mask, s.squeezed(axis: 1))
}

func relL2(_ a: MLXArray, _ b: MLXArray) -> Float { (sqrt(((a - b) * (a - b)).sum()) / sqrt((b * b).sum())).item(Float.self) }
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let (base, baseDir) = try await loadLLM("LiquidAI/LFM2-350M")
let p = Dictionary(uniqueKeysWithValues: base.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) })
let config = try JSONSerialization.jsonObject(with: Data(contentsOf: baseDir.appending(path: "config.json"))) as! [String: Any]
let attentionLayers = Set(config["full_attn_idxs"] as! [Int])
let FF = p["model.layers.0.feed_forward.w1.weight"]!.dim(0)
let shape = LFM2Shape(tokens: T, dim: D, ff: FF, heads: HEADS, headDim: HD, ropeBase: Float(config["rope_theta"] as! Double))

/// Every ternary linear's module path.
let linearPaths: [String] = (0 ..< 16).flatMap { i -> [String] in
    let pre = "model.layers.\(i)."
    return (attentionLayers.contains(i) ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"] : ["conv.in_proj", "conv.out_proj"])
        .map { pre + $0 } + ["feed_forward.w1", "feed_forward.w3", "feed_forward.w2"].map { pre + $0 }
}
let linearWeightKeys = Set(linearPaths.map { $0 + ".weight" })

if CommandLine.arguments.dropFirst().first == "write-ckpt" {
    let out = URL(filePath: CommandLine.arguments[2]), colscale = CommandLine.arguments.contains("--colscale")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    var arrays: [String: MLXArray] = [:]
    MLXRandom.seed(7)
    for path in linearPaths {
        let (t, s) = quantise(p[path + ".weight"]!)
        arrays[path + ".trits"] = t.asType(.int8); arrays[path + ".scale"] = s
        if colscale { arrays[path + ".colscale"] = 0.5 + MLXRandom.uniform(low: 0, high: 1, [t.dim(1)]) }
    }
    if CommandLine.arguments.contains("--tied") {  // colscales foldable into norms: q=k=v, w1=w3, out_proj equal across the two query heads of a kv head
        for path in linearPaths {
            let pre = String(path[..<path.range(of: ".", options: .backwards)!.lowerBound]), name = path.split(separator: ".").last!
            if name == "k_proj" || name == "v_proj" { arrays[path + ".colscale"] = arrays[pre + ".q_proj.colscale"]! }
            if name == "w3" { arrays[path + ".colscale"] = arrays[pre + ".w1.colscale"]! }
            if name == "out_proj" && pre.hasSuffix("self_attn") { arrays[path + ".colscale"] = repeated(arrays[path + ".colscale"]!.reshaped(8, 2, 64)[0..., 0 ..< 1], count: 2, axis: 1).reshaped(-1) }
        }
    }
    for (k, v) in p where !linearWeightKeys.contains(k) { arrays[k] = v }
    try save(arrays: arrays, url: out.appending(path: "model.safetensors"))
    try? FileManager.default.removeItem(at: out.appending(path: "config.json"))
    try Data(contentsOf: baseDir.appending(path: "config.json")).write(to: out.appending(path: "config.json"))  // contents: the snapshot file is a relative symlink
    print("wrote \(arrays.count) tensors to \(out.path)")
    exit(0)
}

// ---- Weights: trits [N,K], scale [N], optional colscale [K], per linear; and the float parameters.
struct Tern { let trits: MLXArray, scale: MLXArray, col: MLXArray? }
var tern: [String: Tern] = [:]
var floats = p.filter { !linearWeightKeys.contains($0.key) }
if let ckpt = option("ckpt") {
    let a = try loadArrays(url: URL(filePath: ckpt).appending(path: "model.safetensors"))
    for path in linearPaths {
        guard let t = a[path + ".trits"], let s = a[path + ".scale"] else { fatalError("checkpoint lacks \(path).trits/.scale") }
        tern[path] = Tern(trits: t, scale: s.asType(.float32), col: a[path + ".colscale"]?.asType(.float32))
    }
    for k in floats.keys { if let v = a[k] { floats[k] = v.asType(.float32) } else { print("note: \(k) not in checkpoint, using base") } }
} else {
    for path in linearPaths { let (t, s) = quantise(p[path + ".weight"]!); tern[path] = Tern(trits: t, scale: s, col: nil) }
}
print("weights: " + (option("ckpt") ?? "base + TWN") + "; colscale on \(tern.values.filter { $0.col != nil }.count) of \(tern.count) linears")

// ---- GPU ternary model in MLX (ternarise's approach).
let (gpuTern, _) = try await loadLLM("LiquidAI/LFM2-350M")
var replaced: [(String, Module)] = []
for path in linearPaths { let t = tern[path]!; replaced.append((path, TernaryLinear(trits: t.trits.asType(.float16), scale: t.scale, col: t.col))) }
gpuTern.update(modules: ModuleChildren.unflattened(replaced))
var cur = Dictionary(uniqueKeysWithValues: gpuTern.parameters().flattened())
for (k, v) in floats where cur[k] != nil { cur[k] = v }
gpuTern.update(parameters: ModuleParameters.unflattened(cur))
eval(gpuTern)
@MainActor func setMatDtype(_ dt: DType) {
    matDtype = dt
    for (_, m) in replaced { let l = m as! TernaryLinear; l.update(parameters: ModuleParameters.unflattened([("weight", l.weight.asType(dt))])); eval(l.weight) }
}

// ---- ANE programs, chained zero-copy.
let embed = floats["model.embed_tokens.weight"]!, finalGain = floats["model.embedding_norm.weight"]!
let embedBuffer = SharedBuffer([1, 1, T, D])
func buf(_ a: MLXArray, _ shape: [Int]) -> SharedBuffer { let b = SharedBuffer(shape); eval(b.write(a.asType(.float16))); return b }
@MainActor func output(_ model: Model, _ name: String) throws -> SurfaceBinding {
    try model.surfaceBindings().first { $0.role == .output && ($0.name == name || $0.name.hasPrefix(name + "@")) }!
}
@MainActor func link(_ producer: Model, _ name: String) throws -> ExternalInputBinding {
    let s = try output(producer, name)
    return try ExternalInputBinding(surface: s.surface, byteRange: 0 ..< IOSurfaceGetAllocSize(s.surface), layout: .compilerLayout(s.layout), retaining: producer, access: .readOnly)
}
@MainActor func linearsHaveCol(_ i: Int) -> Bool { linearPaths.filter { $0.hasPrefix("model.layers.\(i).") }.contains { tern[$0]!.col != nil } }

/// Weight, scale, norm-gain and colscale bindings of layer i, shared by its prefill and decode programs.
@MainActor func weightBindings(_ i: Int) throws -> [String: ExternalInputBinding] {
    let attention = attentionLayers.contains(i), pre = "model.layers.\(i)."
    var bindings: [String: ExternalInputBinding] = [:]
    var colParts: [String: MLXArray] = [:]
    func add(_ name: String, _ path: String, rows: Range<Int>? = nil, gqa: Bool = false) throws {
        let t = tern[pre + path]!
        var w = t.trits, s = t.scale
        if let rows { w = w[rows]; s = s[rows] }
        if gqa {  // 8 kv heads repeated to 16
            w = repeated(w.reshaped(8, 1, 64, D), count: 2, axis: 1).reshaped(D, D); s = repeated(s.reshaped(8, 1, 64), count: 2, axis: 1).reshaped(D)
        }
        bindings["w_\(name)"] = try buf(w, [1, 1, w.dim(0), w.dim(1)]).binding()
        bindings["s_\(name)"] = try buf(s, [1, 1, 1, w.dim(0)]).binding()
        colParts[name] = t.col ?? MLXArray.ones([w.dim(1)])
    }
    if attention {
        try add("q", "self_attn.q_proj"); try add("k", "self_attn.k_proj", gqa: true); try add("v", "self_attn.v_proj", gqa: true); try add("o", "self_attn.out_proj")
        for n in ["q", "k"] { bindings["g_\(n)"] = try buf(tiled(floats[pre + "self_attn.\(n)_layernorm.weight"]!, repetitions: [16]), [1, 16, 1, HD]).binding() }
    } else {
        for (j, n) in ["b", "c", "x"].enumerated() { try add(n, "conv.in_proj", rows: (j * D) ..< ((j + 1) * D)) }
        try add("o", "conv.out_proj")
        let k = floats[pre + "conv.conv.weight"]!  // [D, 3, 1]
        for j in 0 ..< 3 { bindings["k\(j)"] = try buf(k[0..., j, 0], [1, 1, 1, D]).binding() }
    }
    try add("w1", "feed_forward.w1"); try add("w3", "feed_forward.w3"); try add("w2", "feed_forward.w2")
    bindings["g_op"] = try buf(floats[pre + "operator_norm.weight"]!, [1, 1, 1, D]).binding()
    bindings["g_ffn"] = try buf(floats[pre + "ffn_norm.weight"]!, [1, 1, 1, D]).binding()
    if linearsHaveCol(i) {
        let packed = concatenated(colscaleNames(attention: attention, dim: D, ff: FF).map { colParts[$0.name]! }, axis: 0)
        bindings["cs"] = try buf(packed, [1, 1, 1, packed.dim(0)]).binding()
    }
    return bindings
}
nonisolated(unsafe) var weights: [[String: ExternalInputBinding]] = []

@MainActor func compileLayer(_ i: Int, x: ExternalInputBinding) throws -> Model {
    let attention = attentionLayers.contains(i), useCol = linearsHaveCol(i)
    let bindings = weights[i].merging(["x": x]) { $1 }
    let (program, constants) = try layerProgram(shape, attention: attention, diagnostics: false, colscale: useCol)
    let model = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 4 << 30), inputBindings: bindings)
    _ = try model.prepareUpdating(inputs: constants)
    return model
}

let compileStart = now()
weights = try (0 ..< 16).map(weightBindings)
var layers: [Model] = []
for i in 0 ..< 16 { layers.append(try compileLayer(i, x: i == 0 ? try embedBuffer.binding() : try link(layers[i - 1], "y"))) }
print(String(format: "compiled 16 ANE programs in %.1fs", now() - compileStart))

/// Final hidden state [T, D] fp32 from the last layer's output surface.
@MainActor func aneHidden() throws -> MLXArray { view(try output(layers[15], "y")).asType(.float32).reshaped(T, D) }
@MainActor func aneForward(_ tokens: MLXArray, poison: Bool = false) throws -> [Double] {
    eval(embedBuffer.write(embed[tokens]))
    return try layers.map { try $0.evaluatePrepared(readOutputs: false, poisonOutputs: poison).elapsed }
}
let embedBF = embed.asType(.bfloat16)
@MainActor func head(_ h: MLXArray, fp32: Bool) -> MLXArray {
    let n = MLXFast.rmsNorm(h, weight: finalGain, eps: 1e-5)
    return fp32 ? matmul(n, embed.T) : matmul(n.asType(.bfloat16), embedBF.T).asType(.float32)
}

// ---- Sequences: BOS + 255 wikitext tokens, 4 windows.
let tokenizer = try await AutoTokenizer.from(modelFolder: baseDir)
let text = try String(contentsOf: URL(filePath: option("text") ?? "/tmp/wikitext2-test-join.txt"), encoding: .utf8)
let nseq = Int(option("seqs") ?? "4")!, reps = Int(option("reps") ?? "20")!
let allIds = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
let seqs = (0 ..< nseq).map { s -> MLXArray in MLXArray([Int32(tokenizer.bosTokenId ?? 1)] + Array(allIds[(s * 4000) ..< (s * 4000 + T - 1)])) }

@MainActor func gpuLogits(_ m: any LLMModel, _ t: MLXArray) -> MLXArray { let l = m(t[.newAxis], cache: nil)[0].asType(.float32); eval(l); return l }
@MainActor func gpuRate(_ m: any LLMModel) -> Double {
    let x = seqs[0][.newAxis]
    for _ in 0 ..< 3 { eval(m(x, cache: nil)) }
    let t0 = now()
    for _ in 0 ..< reps { eval(m(x, cache: nil)) }
    return Double(T * reps) / (now() - t0)
}

// ---- Decode: one token per step, 16 single-token programs chained zero-copy; conv state and the K/V caches are updated by the host.
let L = Int(option("max-len") ?? "512")!
@MainActor protocol StepDecoder: AnyObject {
    func step(_ token: Int) throws -> Double
    func hidden() -> MLXArray
    func reset()
}
@MainActor final class Decoder: StepDecoder {
    let xBuf = SharedBuffer([1, 1, 1, D]), onehot = SharedBuffer([1, 1, 1, L]), mask = SharedBuffer([1, 1, 1, L])
    let cosBuf = SharedBuffer([1, 1, 1, HD]), sinBuf = SharedBuffer([1, 1, 1, HD])
    var models: [Model] = [], kc: [SharedBuffer?] = [], vc: [SharedBuffer?] = [], s0: [SharedBuffer?] = [], s1: [SharedBuffer?] = []
    var out: [[String: SurfaceBinding]] = []
    var pos = 0
    let embed16: [Float16] = embed.asType(.float16).asArray(Float16.self)  // CPU copy: the embedding lookup is a 2 KB memcpy
    init() throws {
        for i in 0 ..< 16 {
            let attention = attentionLayers.contains(i)
            var b = weights[i]
            b["x"] = i == 0 ? try xBuf.binding() : try link(models[i - 1], "y")
            if attention {
                let k = SharedBuffer([1, HEADS, L, HD]), v = SharedBuffer([1, HEADS, L, HD])
                kc.append(k); vc.append(v); s0.append(nil); s1.append(nil)
                b["kc"] = try k.binding(); b["vc"] = try v.binding(); b["onehot"] = try onehot.binding(); b["mask"] = try mask.binding()
                b["cos"] = try cosBuf.binding(); b["sin"] = try sinBuf.binding()
            } else {
                let a = SharedBuffer([1, 1, 1, D]), c = SharedBuffer([1, 1, 1, D])
                kc.append(nil); vc.append(nil); s0.append(a); s1.append(c)
                b["s0"] = try a.binding(); b["s1"] = try c.binding()
            }
            let (program, constants) = try decodeProgram(shape, attention: attention, maxLength: L, colscale: linearsHaveCol(i))
            let m = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 4 << 30), inputBindings: b)
            _ = try m.prepareUpdating(inputs: constants)
            models.append(m)
            let names = attention ? ["y", "k_new", "v_new"] : ["y", "ns0", "ns1"]
            out.append(Dictionary(uniqueKeysWithValues: try names.map { ($0, try output(m, $0)) }))
        }
        reset()
    }
    func reset() {
        pos = 0
        for b in kc + vc + s0 + s1 { if let b { memset(b.pointer, 0, b.bytes) } }
        memset(onehot.pointer, 0, onehot.bytes)
        let m = mask.pointer.assumingMemoryBound(to: Float16.self)
        for j in 0 ..< L { m[j] = -30000 }
    }
    /// Copies `rows` rows of a program output (one per plane, `rowBytes` each) to dst + r * stride.
    func copy(_ b: SurfaceBinding, rows: Int, rowBytes: Int, to dst: UnsafeMutableRawPointer, stride: Int) {
        let base = IOSurfaceGetBaseAddress(b.surface) + b.startOffset
        for r in 0 ..< rows { memcpy(dst + r * stride, base + r * b.layout.planeStride, rowBytes) }
    }
    /// Runs one token; returns the summed ANE time of the layers. The hidden state is in `hidden()`.
    func step(_ token: Int) throws -> Double {
        precondition(pos < L)
        embed16.withUnsafeBytes { _ = memcpy(xBuf.pointer, $0.baseAddress! + token * D * 2, D * 2) }
        let oh = onehot.pointer.assumingMemoryBound(to: Float16.self), m = mask.pointer.assumingMemoryBound(to: Float16.self)
        if pos > 0 { oh[pos - 1] = 0 }
        oh[pos] = 1; m[pos] = 0
        let c = cosBuf.pointer.assumingMemoryBound(to: Float16.self), sn = sinBuf.pointer.assumingMemoryBound(to: Float16.self)
        for i in 0 ..< HD {
            let angle = Double(pos) * pow(Double(shape.ropeBase), -Double(2 * (i % (HD / 2))) / Double(HD))
            c[i] = Float16(cos(angle)); sn[i] = Float16(sin(angle))
        }
        var total = 0.0
        for i in 0 ..< 16 {
            total += try models[i].evaluatePrepared(readOutputs: false, poisonOutputs: false).elapsed
            if let k = kc[i], let v = vc[i] {
                copy(out[i]["k_new"]!, rows: HEADS, rowBytes: HD * 2, to: k.pointer + pos * HD * 2, stride: L * HD * 2)
                copy(out[i]["v_new"]!, rows: HEADS, rowBytes: HD * 2, to: v.pointer + pos * HD * 2, stride: L * HD * 2)
            } else {
                copy(out[i]["ns0"]!, rows: 1, rowBytes: D * 2, to: s0[i]!.pointer, stride: 0)
                copy(out[i]["ns1"]!, rows: 1, rowBytes: D * 2, to: s1[i]!.pointer, stride: 0)
            }
        }
        pos += 1
        return total
    }
    func hidden() -> MLXArray { view(out[15]["y"]!).asType(.float32).reshaped(1, D) }
}

// ---- Fused decode (--fuse k [--wmode input|fp16|pal2]): k consecutive layers per program, weights as packed inputs or constants.
@MainActor func layerData(_ i: Int) -> LayerData {
    let attention = attentionLayers.contains(i), pre = "model.layers.\(i)."
    var tr: [String: [Int8]] = [:], sc: [String: [Float16]] = [:], sm: [String: [Float16]] = [:]
    func add(_ name: String, _ path: String, rows: Range<Int>? = nil, gqa: Bool = false) {
        let t = tern[pre + path]!
        precondition(t.col == nil, "fused decode does not support colscale")
        var w = t.trits, s = t.scale
        if let rows { w = w[rows]; s = s[rows] }
        if gqa { w = repeated(w.reshaped(8, 1, 64, D), count: 2, axis: 1).reshaped(D, D); s = repeated(s.reshaped(8, 1, 64), count: 2, axis: 1).reshaped(D) }
        tr[name] = w.asType(.int8).asArray(Int8.self); sc[name] = s.asType(.float16).asArray(Float16.self)
    }
    func half(_ a: MLXArray) -> [Float16] { a.asType(.float16).asArray(Float16.self) }
    if attention {
        add("q", "self_attn.q_proj"); add("k", "self_attn.k_proj", gqa: true); add("v", "self_attn.v_proj", gqa: true); add("o", "self_attn.out_proj")
        for n in ["q", "k"] { sm["g_\(n)"] = half(tiled(floats[pre + "self_attn.\(n)_layernorm.weight"]!, repetitions: [16])) }
    } else {
        for (j, n) in ["b", "c", "x"].enumerated() { add(n, "conv.in_proj", rows: (j * D) ..< ((j + 1) * D)) }
        add("o", "conv.out_proj")
        let k = floats[pre + "conv.conv.weight"]!
        for j in 0 ..< 3 { sm["k\(j)"] = half(k[0..., j, 0]) }
    }
    add("w1", "feed_forward.w1"); add("w3", "feed_forward.w3"); add("w2", "feed_forward.w2")
    sm["g_op"] = half(floats[pre + "operator_norm.weight"]!); sm["g_ffn"] = half(floats[pre + "ffn_norm.weight"]!)
    return LayerData(trits: tr, scale: sc, small: sm)
}

@MainActor final class FusedDecoder: StepDecoder {
    struct Group { let range: Range<Int>, model: Model, cst: SharedBuffer?, kc: SharedBuffer?, vc: SharedBuffer?, out: [String: SurfaceBinding], nConv: Int, nAttn: Int }
    let xBuf = SharedBuffer([1, 1, 1, D]), onehot = SharedBuffer([1, 1, 1, L]), mask = SharedBuffer([1, 1, 1, L])
    let cosBuf = SharedBuffer([1, 1, 1, HD]), sinBuf = SharedBuffer([1, 1, 1, HD])
    var groups: [Group] = []
    var pos = 0
    let embed16: [Float16] = embed.asType(.float16).asArray(Float16.self)
    init(k: Int, mode: WeightMode) throws {
        var prev: Model?
        for start in stride(from: 0, to: 16, by: k) {
            let range = start ..< min(start + k, 16), attn = range.map { attentionLayers.contains($0) }
            let data = range.map(layerData)
            let (program, constants) = try fusedDecodeProgram(shape, attention: attn, maxLength: L, mode: mode, layers: mode == .input ? nil : data)
            var b: [String: ExternalInputBinding] = ["x": prev == nil ? try xBuf.binding() : try link(prev!, "y")]
            let nConv = attn.filter { !$0 }.count, nAttn = attn.count - nConv
            var cst: SharedBuffer?, kc: SharedBuffer?, vc: SharedBuffer?
            if nConv > 0 { cst = SharedBuffer([1, nConv, 2, D]); b["cst"] = try cst!.binding() }
            if nAttn > 0 {
                kc = SharedBuffer([1, nAttn * HEADS, L, HD]); vc = SharedBuffer([1, nAttn * HEADS, L, HD])
                b["kc"] = try kc!.binding(); b["vc"] = try vc!.binding(); b["onehot"] = try onehot.binding(); b["mask"] = try mask.binding()
                b["cos"] = try cosBuf.binding(); b["sin"] = try sinBuf.binding()
            }
            if mode == .input {
                let p = packInputs(data, attention: attn, dim: D, ff: FF)
                func put(_ name: String, _ shape: [Int], _ v: [Float16]) throws {
                    let buf = SharedBuffer(shape); v.withUnsafeBytes { _ = memcpy(buf.pointer, $0.baseAddress!, $0.count) }; b[name] = try buf.binding()
                }
                try put("wA", [1, attn.count, 4 * D + 2 * FF, D], p.wA); try put("wB", [1, attn.count, D, FF], p.wB); try put("sm", [1, attn.count, fusedSmallRows, FF], p.sm)
            }
            let t0 = now()
            let m = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 4 << 30), inputBindings: b)
            _ = try m.prepareUpdating(inputs: constants)
            print(String(format: "  group %@ compiled in %.1fs", "\(range)", now() - t0))
            let names = ["y"] + (nConv > 0 ? ["ns"] : []) + (nAttn > 0 ? ["kn", "vn"] : [])
            groups.append(Group(range: range, model: m, cst: cst, kc: kc, vc: vc, out: Dictionary(uniqueKeysWithValues: try names.map { ($0, try output(m, $0)) }), nConv: nConv, nAttn: nAttn))
            prev = m
        }
        reset()
    }
    func reset() {
        pos = 0
        for g in groups { for b in [g.cst, g.kc, g.vc] { if let b { memset(b.pointer, 0, b.bytes) } } }
        memset(onehot.pointer, 0, onehot.bytes)
        let m = mask.pointer.assumingMemoryBound(to: Float16.self)
        for j in 0 ..< L { m[j] = -30000 }
    }
    /// Copies planes x rows of a program output to dst (plane p, row r at dst + p * planeStride + r * rowStride).
    func copy(_ b: SurfaceBinding, planes: Int, rows: Int, rowBytes: Int, to dst: UnsafeMutableRawPointer, planeStride: Int, rowStride: Int) {
        let base = IOSurfaceGetBaseAddress(b.surface) + b.startOffset
        for p in 0 ..< planes { for r in 0 ..< rows { memcpy(dst + p * planeStride + r * rowStride, base + p * b.layout.planeStride + r * b.layout.rowStride, rowBytes) } }
    }
    func step(_ token: Int) throws -> Double {
        precondition(pos < L)
        embed16.withUnsafeBytes { _ = memcpy(xBuf.pointer, $0.baseAddress! + token * D * 2, D * 2) }
        let oh = onehot.pointer.assumingMemoryBound(to: Float16.self), m = mask.pointer.assumingMemoryBound(to: Float16.self)
        if pos > 0 { oh[pos - 1] = 0 }
        oh[pos] = 1; m[pos] = 0
        let c = cosBuf.pointer.assumingMemoryBound(to: Float16.self), sn = sinBuf.pointer.assumingMemoryBound(to: Float16.self)
        for i in 0 ..< HD {
            let angle = Double(pos) * pow(Double(shape.ropeBase), -Double(2 * (i % (HD / 2))) / Double(HD))
            c[i] = Float16(cos(angle)); sn[i] = Float16(sin(angle))
        }
        var total = 0.0
        for g in groups {
            total += try g.model.evaluatePrepared(readOutputs: false, poisonOutputs: false).elapsed
            if let cst = g.cst { copy(g.out["ns"]!, planes: g.nConv, rows: 2, rowBytes: D * 2, to: cst.pointer, planeStride: 2 * D * 2, rowStride: D * 2) }
            if let kc = g.kc, let vc = g.vc {
                copy(g.out["kn"]!, planes: g.nAttn * HEADS, rows: 1, rowBytes: HD * 2, to: kc.pointer + pos * HD * 2, planeStride: L * HD * 2, rowStride: 0)
                copy(g.out["vn"]!, planes: g.nAttn * HEADS, rows: 1, rowBytes: HD * 2, to: vc.pointer + pos * HD * 2, planeStride: L * HD * 2, rowStride: 0)
            }
        }
        pos += 1
        return total
    }
    func hidden() -> MLXArray { view(groups.last!.out["y"]!).asType(.float32).reshaped(1, D) }
}

@MainActor func runDecode() throws {
    print("# decode (single-token programs, max length \(L))")
    let t0 = now()
    let fuse = Int(option("fuse") ?? "0")!
    let dec: any StepDecoder = fuse > 0 ? try FusedDecoder(k: fuse, mode: WeightMode(rawValue: option("wmode") ?? "input")!) : try Decoder()
    print("decode mode: " + (fuse > 0 ? "fused k=\(fuse), weights \(option("wmode") ?? "input")" : "one program per layer, fp16 runtime inputs"))
    print(String(format: "compiled 16 decode programs in %.1fs", now() - t0))
    // Parity: teacher-forced, the decoder's logits at each position against the GPU ternary model's prefill logits.
    setMatDtype(.float32)
    let n = Int(option("decode-parity") ?? "128")!
    let seq = seqs[0], ref = gpuLogits(gpuTern, seq)
    var rows: [MLXArray] = []
    for t in 0 ..< n { _ = try dec.step(Int(seq[t].item(Int32.self))); let r = head(dec.hidden(), fp32: true)[0]; eval(r); rows.append(r) }
    let got = stacked(rows, axis: 0), want = ref[0 ..< n]
    eval(got)
    let rels = (0 ..< n).map { relL2(got[$0], want[$0]) }
    let agree = (got.argMax(axis: -1) .== want.argMax(axis: -1)).asType(.int32).sum().item(Int.self)
    let lpA = logSoftmax(got, axis: -1), lpG = logSoftmax(want, axis: -1)
    print(String(format: "decode vs GPU ternary, %d teacher-forced positions: logits rel-L2 %.4f (all together), worst per position %.4f; argmax agreement %.2f%%; KL(GPU || ANE) %.5f",
                 n, relL2(got, want), rels.max()!, 100 * Double(agree) / Double(n), (exp(lpG) * (lpG - lpA)).sum(axis: -1).mean().item(Float.self)))
    // Speed: greedy generation from BOS, `steps` tokens, head and argmax on the GPU.
    let steps = Int(option("decode-steps") ?? "128")!
    func aneSpeed() throws -> (Double, Double) {
        dec.reset()
        var tok = Int(tokenizer.bosTokenId ?? 1), aneTime = 0.0
        let t0 = now()
        for _ in 0 ..< steps {
            aneTime += try dec.step(tok)
            let a = head(dec.hidden(), fp32: false).argMax(axis: -1); eval(a)
            tok = a.item(Int.self)
        }
        return (Double(steps) / (now() - t0), Double(steps) / aneTime)
    }
    func gpuSpeed(_ m: any LLMModel) throws -> Double {
        let bos = Int32(tokenizer.bosTokenId ?? 1)
        func go(_ n: Int) throws {
            let cache = try m.newCache(parameters: nil)
            var x = MLXArray([bos])[.newAxis]
            for _ in 0 ..< n { x = m(x, cache: cache)[0..., -1, 0...].argMax(axis: -1)[.newAxis].asType(.int32); eval(x) }
        }
        try go(4)
        let t0 = now()
        try go(steps)
        return Double(steps) / (now() - t0)
    }
    setMatDtype(.bfloat16)
    let noGpu = CommandLine.arguments.contains("--no-gpu-decode")
    for pass in 1 ... 2 {
        let (e2e, aneOnly) = try aneSpeed()
        print(String(format: "pass %d: ANE decode %.1f tok/s end to end (%.1f ms/token; layers only %.1f tok/s); GPU bf16 base %.1f tok/s; GPU ternary (bf16 matmuls) %.1f tok/s",
                     pass, e2e, 1000 / e2e, aneOnly, noGpu ? 0 : try gpuSpeed(base), noGpu ? 0 : try gpuSpeed(gpuTern)))
    }
}

@MainActor func run() throws {
    print("# parity (ANE vs the same ternary model in MLX on the GPU, fp32 matmuls), \(nseq) sequences of \(T) tokens")
    setMatDtype(.float32)
    var layerTime = [Double](repeating: 0, count: 16)
    var layerMax = [Float](repeating: 0, count: 16)
    var agree = [Float](), rel = [Float](), kl = [Float](), klBase = [Float](), argBase = [Float](), argGpuBase = [Float](), maxAbs: Float = 0, bad = 0
    for t in seqs {
        let times = try aneForward(t, poison: true)
        _ = times
        for (li, l) in layers.enumerated() {
            let y = view(try output(l, "y")).asType(.float32)
            bad += (isNaN(y).asType(.int32) + (abs(y) .> 65000).asType(.int32)).sum().item(Int.self)
            let m = MLX.where(isNaN(y), 0, abs(y)).max().item(Float.self)
            maxAbs = max(maxAbs, m); layerMax[li] = max(layerMax[li], m)
        }
        let a = head(try aneHidden(), fp32: true), g = gpuLogits(gpuTern, t), b = gpuLogits(base, t)
        eval(a)
        rel.append(relL2(a, g))
        agree.append((a.argMax(axis: -1) .== g.argMax(axis: -1)).asType(.float32).mean().item(Float.self))
        let lpA = logSoftmax(a, axis: -1), lpG = logSoftmax(g, axis: -1), lpB = logSoftmax(b, axis: -1)
        kl.append((exp(lpG) * (lpG - lpA)).sum(axis: -1).mean().item(Float.self))
        klBase.append((exp(lpB) * (lpB - lpA)).sum(axis: -1).mean().item(Float.self))
        argBase.append((a.argMax(axis: -1) .== b.argMax(axis: -1)).asType(.float32).mean().item(Float.self))
        argGpuBase.append((g.argMax(axis: -1) .== b.argMax(axis: -1)).asType(.float32).mean().item(Float.self))
    }
    func mean(_ x: [Float]) -> Float { x.reduce(0, +) / Float(x.count) }
    func list(_ x: [Float], _ f: String) -> String { x.map { String(format: f, $0) }.joined(separator: " ") }
    print("logits rel-L2 ANE vs GPU ternary:   mean \(String(format: "%.4f", mean(rel)))  per sequence: \(list(rel, "%.4f"))")
    print("argmax agreement ANE vs GPU ternary: mean \(String(format: "%.2f%%", 100 * mean(agree)))  per sequence: \(list(agree.map { 100 * $0 }, "%.2f"))")
    print("KL(GPU ternary || ANE) per token:    mean \(String(format: "%.5f", mean(kl)))  per sequence: \(list(kl, "%.5f"))")
    print("context: KL(bf16 base || ANE) \(String(format: "%.4f", mean(klBase))), argmax ANE vs base \(String(format: "%.1f%%", 100 * mean(argBase))), GPU ternary vs base \(String(format: "%.1f%%", 100 * mean(argGpuBase)))")
    print("overflow: \(bad) nonfinite or |y| > 65000 in any layer output over all sequences; max |y| over all layers \(String(format: "%.1f", maxAbs)); per layer: " + layerMax.map { String(format: "%.1f", $0) }.joined(separator: " "))

    print("# prefill speed, T=\(T), \(reps) reps")
    let tokens = seqs[0]
    for _ in 0 ..< 3 { _ = try aneForward(tokens) }
    var e2e = 0.0, aneOnly = 0.0
    for _ in 0 ..< reps {
        let t0 = now()
        for (i, x) in try aneForward(tokens).enumerated() { layerTime[i] += x / Double(reps) }
        let t1 = now()
        let l = head(try aneHidden(), fp32: false); eval(l)
        e2e += now() - t0; aneOnly += t1 - t0
    }
    print("per-layer ANE time (mean ms, \(reps) reps): " + layerTime.enumerated().map { String(format: "%d%@ %.2f", $0.offset, attentionLayers.contains($0.offset) ? "a" : "c", $0.element * 1000) }.joined(separator: " ")
          + String(format: "; sum %.2f ms", layerTime.reduce(0, +) * 1000))
    print(String(format: "ANE end to end (embed GPU + 16 layers ANE + head GPU): %.1f ms per sequence, %.0f tok/s;  embed+layers only: %.1f ms, %.0f tok/s",
                 e2e / Double(reps) * 1000, Double(T * reps) / e2e, aneOnly / Double(reps) * 1000, Double(T * reps) / aneOnly))
    setMatDtype(.bfloat16)
    print(String(format: "GPU bf16 base (mlx-swift-lm):   %.0f tok/s", gpuRate(base)))
    print(String(format: "GPU ternary (bf16 matmuls):     %.0f tok/s", gpuRate(gpuTern)))
    print("(GPU numbers, second pass)")
    print(String(format: "GPU bf16 base: %.0f tok/s, GPU ternary: %.0f tok/s", gpuRate(base), gpuRate(gpuTern)))
}
if !CommandLine.arguments.contains("--decode-only") { try run() }
if !CommandLine.arguments.contains("--no-decode") { try runDecode() }
