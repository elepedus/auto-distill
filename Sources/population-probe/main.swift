// Does a population of P models in one ANE program cost less than P programs, and does each member compute
// exactly what it would alone? Tiny ternary LFM2 layers (d 256), random weights, T = 256.
import ANE
import Foundation
import LFM2ANE
import MLX

let T = 256, D = 256, F = 768, H = 4, hd = 64
func shape(_ p: Int) -> LFM2Shape { LFM2Shape(tokens: T, dim: D, ff: F, heads: H, headDim: hd, ropeBase: 1e6, population: p) }

/// Random value for an input of the given name and [1, C, h, w] shape.
func random(_ name: String, _ dims: [Int]) -> MLXArray {
    switch name.prefix(2) {
    case "w_": MLXRandom.randInt(-1 ..< 2, dims).asType(.float32)
    case "s_": MLXArray.full(dims, values: MLXArray(1 / sqrt(Float(dims[3]) * 2 / 3)))  // per-row scale ~ 1/sqrt(2K/3)
    case "g_": 1 + 0.1 * MLXRandom.normal(dims)
    case "k0", "k1", "k2": 0.3 * MLXRandom.normal(dims)
    default: MLXRandom.normal(dims)  // x, gy
    }
}

func run(_ p: Int, attention: Bool, backward: Bool, inputs: [String: MLXArray]) throws -> (ms: Double, outputs: [String: MLXArray]) {
    let (program, constants) = try layerProgram(shape(p), attention: attention, backward: backward, diagnostics: false)
    let model = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 2 << 30))
    defer { try? model.close() }
    var feed = constants
    for l in model.inputs where feed[l.name] == nil {
        feed[l.name] = try tensor(inputs[l.name]!, [1, l.shape.channels, l.shape.height, l.shape.width])
    }
    _ = try model.prepare(inputs: feed)
    var times = [Double]()
    for _ in 0 ..< 12 { times.append(try model.evaluatePrepared(readOutputs: false, poisonOutputs: false).elapsed) }
    let outs = try model.surfaceBindings().filter { $0.role == .output }
    return (times.dropFirst(2).sorted()[5] * 1000, Dictionary(uniqueKeysWithValues: outs.map {
        ($0.name.split(separator: "@").first.map(String.init)!, view($0).asType(.float32))
    }))
}

func relL2(_ a: MLXArray, _ b: MLXArray) -> Float { (sqrt(((a - b) * (a - b)).sum()) / sqrt((b * b).sum())).item(Float.self) }

MLXRandom.seed(0)
print("# population-probe: \(Date()), \(ProcessInfo.processInfo.operatingSystemVersionString). Tiny ternary LFM2 layers d=\(D), ff=\(F), \(H) heads, T=\(T).")
for attention in [false, true] {
    var solo = [Bool: Double]()
    for p in [1, 4, 8, 16] {
        var line = String(format: "%@ P=%2d", attention ? "attn" : "conv", p)
        for backward in [false, true] {
            // Population inputs: channel axis holds the members (P·H channels for per-head gains).
            let (program, _) = try layerProgram(shape(p), attention: attention, backward: backward, diagnostics: false)
            let probe = try Model(program: program, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 2 << 30))
            let layouts = probe.inputs; try probe.close()
            var inputs = [String: MLXArray]()
            for l in layouts where !l.name.hasPrefix("c") || l.name.count > 3 {
                inputs[l.name] = random(l.name, [1, l.shape.channels, l.shape.height, l.shape.width])
            }
            let (ms, outputs) = try run(p, attention: attention, backward: backward, inputs: inputs)
            if p == 1 { solo[backward] = ms }
            line += String(format: "  %@ %6.2f ms (%.2f ms/member, %.1fx vs solo)", backward ? "bwd" : "fwd", ms, ms / Double(p), solo[backward]! * Double(p) / ms)
            if p > 1 {  // each member alone must match its slice of the population
                var worst: Float = 0
                for j in [0, p - 1] {
                    let mine = inputs.mapValues { a -> MLXArray in
                        let c = a.dim(1) / p
                        return a[0..., (j * c) ..< ((j + 1) * c)]
                    }
                    let (_, alone) = try run(1, attention: attention, backward: backward, inputs: mine)
                    for (name, out) in alone {
                        let c = outputs[name]!.dim(1) / p
                        worst = max(worst, relL2(outputs[name]![0..., (j * c) ..< ((j + 1) * c)], out))
                    }
                }
                line += String(format: " [members vs solo: max rel L2 %.1e]", worst)
            }
        }
        print(line)
    }
}
