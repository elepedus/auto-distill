// One LFM2 decoder layer as an ANE program. Ternary weights and all scales are runtime inputs,
// so the same surfaces can later be rewritten in place by the trainer.
import ANE
import Foundation

struct V { let name: String; let shape: [Int] }

/// Flat-SSA MIL text emitter (fp16), after swift-ane's study graphs.
final class MILText {
    private var lines: [String] = [], signature: [String] = [], serial = 0
    /// Constant tensors, fed as runtime inputs at prepare time.
    private(set) var constants: [String: Tensor] = [:]

    private func fresh() -> String { serial += 1; return "v\(serial)" }
    private static func dims(_ s: [Int]) -> String { "[" + s.map(String.init).joined(separator: ", ") + "]" }
    private func scalar(_ type: String, _ literal: String) -> String {
        let n = fresh(); lines.append("\(type) \(n) = const()[name = string(\"\(n)\"), val = \(type)(\(literal))];"); return n
    }
    private func ints(_ v: [Int]) -> String {
        let n = fresh(), t = "tensor<int32, [\(v.count)]>"
        lines.append("\(t) \(n) = const()[name = string(\"\(n)\"), val = \(t)(\(Self.dims(v)))];"); return n
    }
    private func half(_ x: Float) -> String { scalar("fp16", String(format: "%a", Double(Float16(x)))) }
    private func op(_ name: String, _ args: String, _ shape: [Int], as out: String? = nil) -> V {
        let n = out ?? fresh()
        lines.append("tensor<fp16, \(Self.dims(shape))> \(n) = \(name)(\(args))[name = string(\"\(n)\")];")
        return V(name: n, shape: shape)
    }

    func input(_ name: String, _ shape: [Int]) -> V {
        signature.append("tensor<fp16, \(Self.dims(shape))> \(name)"); return V(name: name, shape: shape)
    }
    /// A constant as a runtime input. Several BLOBFILE constants in one in-memory program fail to
    /// load ("verifyBundleAtPath: invalid model") on macOS 27.2; one works, inputs always work.
    func const(_ values: [Float16], _ shape: [Int]) throws -> V {
        let name = "c\(constants.count)"
        constants[name] = try Tensor(shape: TensorShape(batches: shape[0], channels: shape[1], height: shape[2], width: shape[3]), values: values)
        return input(name, shape)
    }
    func binary(_ name: String, _ x: V, _ y: V) -> V {
        op(name, "x = \(x.name), y = \(y.name)", zip(x.shape, y.shape).map { max($0, $1) })
    }
    func mul(_ x: V, _ y: V) -> V { binary("mul", x, y) }
    func add(_ x: V, _ y: V) -> V { binary("add", x, y) }
    func scale(_ x: V, _ s: Float) -> V { op("mul", "x = \(x.name), y = \(half(s))", x.shape) }
    func offset(_ x: V, _ c: Float) -> V { op("add", "x = \(x.name), y = \(half(c))", x.shape) }
    func divide(_ c: Float, by x: V) -> V { op("real_div", "x = \(half(c)), y = \(x.name)", x.shape) }
    func unary(_ name: String, _ x: V) -> V { op(name, "x = \(x.name)", x.shape) }
    func matmul(_ x: V, _ y: V, transposeY: Bool = false) -> V {
        let n = transposeY ? y.shape[2] : y.shape[3]
        return op("matmul", "transpose_x = \(scalar("bool", "false")), transpose_y = \(scalar("bool", "\(transposeY)")), x = \(x.name), y = \(y.name)",
                  Array(x.shape.dropLast()) + [n])
    }
    func rowMax(_ x: V) -> V {
        op("reduce_max", "axes = \(ints([-1])), keep_dims = \(scalar("bool", "true")), x = \(x.name)", Array(x.shape.dropLast()) + [1])
    }
    func floor(_ x: V, _ value: Float) -> V { op("maximum", "x = \(x.name), y = \(half(value))", x.shape) }
    func rsqrt(_ x: V) -> V { op("rsqrt", "epsilon = \(half(0x1p-24)), x = \(x.name)", x.shape) }
    func reshape(_ x: V, _ shape: [Int]) -> V { op("reshape", "shape = \(ints(shape)), x = \(x.name)", shape) }
    func transpose(_ x: V, _ perm: [Int]) -> V { op("transpose", "perm = \(ints(perm)), x = \(x.name)", perm.map { x.shape[$0] }) }
    func output(_ x: V, _ name: String) -> V { op("identity", "x = \(x.name)", x.shape, as: name) }

    func program(outputs: [V]) throws -> Program {
        let text = "program(1.3)\n{\n    func main<ios18>(\(signature.joined(separator: ", "))) {\n"
            + lines.map { "        " + $0 }.joined(separator: "\n")
            + "\n    } -> (\(outputs.map(\.name).joined(separator: ", ")));\n}\n"
        return try Program(mil: text)
    }
}

struct LFM2Shape { let tokens: Int, dim: Int, ff: Int, heads: Int, headDim: Int, ropeBase: Float }

/// Builds layer `y = h + mlp(norm(h))`, `h = x + mixer(norm(x))`, with outputs y, r (mixer) and m (mlp).
func layerProgram(_ s: LFM2Shape, attention: Bool) throws -> (Program, constants: [String: Tensor]) {
    let g = MILText(), T = s.tokens, D = s.dim, H = s.heads, hd = s.headDim
    let onesD = try g.const(.init(repeating: 1, count: D), [1, 1, D, 1])

    // x / rms(x) * gamma, as xs / rms(xs) with xs = x / max|x|: no finite input can overflow the squares,
    // and the sum of squares goes through a matmul (wide accumulator), not reduce_sum.
    func norm(_ x: V, _ gamma: V, ones: V) -> V {
        let m = g.floor(g.rowMax(g.unary("abs", x)), 0x1p-14), xs = g.binary("real_div", x, m)
        let meanSquare = g.scale(g.matmul(g.mul(xs, xs), ones), 1 / Float(x.shape[3]))
        let eps = g.divide(1e-5, by: g.mul(m, m))  // RMSNorm's eps, rescaled with x: mean(x²) + eps = m²(mean(xs²) + eps/m²)
        return g.mul(g.mul(xs, g.rsqrt(g.add(meanSquare, eps))), gamma)
    }
    func linear(_ x: V, _ name: String, out: Int) -> V {
        let w = g.input("w_\(name)", [1, 1, out, x.shape[3]]), scale = g.input("s_\(name)", [1, 1, 1, out])
        return g.mul(g.matmul(x, w, transposeY: true), scale)  // trits in MLX's [out, in] layout
    }

    let x = g.input("x", [1, 1, T, D])
    let xn = norm(x, g.input("g_op", [1, 1, 1, D]), ones: onesD)
    let r: V
    if attention {
        func heads(_ z: V) -> V { g.transpose(g.reshape(z, [1, T, H, hd]), [0, 2, 1, 3]) }
        let onesH = try g.const(.init(repeating: 1, count: H * hd), [1, H, hd, 1])
        var cosT = [Float16](), sinT = [Float16](), rot = [Float16](repeating: 0, count: H * hd * hd)
        for _ in 0 ..< H {
            for t in 0 ..< T {
                for i in 0 ..< hd {
                    let angle = Float(t) * pow(s.ropeBase, -Float(2 * (i % (hd / 2))) / Float(hd))
                    cosT.append(Float16(cos(angle))); sinT.append(Float16(sin(angle)))
                }
            }
        }
        for h in 0 ..< H {  // rotate_half(q) = q @ R: column i takes -q[i + hd/2] (first half) or q[i - hd/2]
            for i in 0 ..< hd / 2 {
                rot[h * hd * hd + (i + hd / 2) * hd + i] = -1
                rot[h * hd * hd + i * hd + (i + hd / 2)] = 1
            }
        }
        let cosC = try g.const(cosT, [1, H, T, hd]), sinC = try g.const(sinT, [1, H, T, hd])
        let rotC = try g.const(rot, [1, H, hd, hd])
        var mask = [Float16]()
        for _ in 0 ..< H { for i in 0 ..< T { for j in 0 ..< T { mask.append(j <= i ? 0 : -30000) } } }
        let maskC = try g.const(mask, [1, H, T, T]), onesT = try g.const(.init(repeating: 1, count: H * T), [1, H, T, 1])
        func rope(_ z: V) -> V { g.add(g.mul(z, cosC), g.mul(g.matmul(z, rotC), sinC)) }

        // GQA as 16 full heads: the caller duplicates each kv head's weight rows.
        let q = rope(norm(heads(linear(xn, "q", out: H * hd)), g.input("g_q", [1, H, 1, hd]), ones: onesH))
        let k = rope(norm(heads(linear(xn, "k", out: H * hd)), g.input("g_k", [1, H, 1, hd]), ones: onesH))
        let v = heads(linear(xn, "v", out: H * hd))
        let scores = g.add(g.scale(g.matmul(q, k, transposeY: true), 1 / sqrt(Float(hd))), maskC)
        let e = g.unary("exp", g.binary("sub", scores, g.rowMax(scores)))
        // V * 2^8 keeps small probability-weighted terms out of fp16 underflow (swift-ane attention study).
        let o = g.scale(g.binary("real_div", g.matmul(e, g.scale(v, 0x1p8)), g.matmul(e, onesT)), 0x1p-8)
        r = linear(g.reshape(g.transpose(o, [0, 2, 1, 3]), [1, 1, T, H * hd]), "o", out: D)
    } else {
        let b = linear(xn, "b", out: D), c = linear(xn, "c", out: D), xx = linear(xn, "x", out: D)
        // Causal 3-tap conv over tokens. Row shifts are matmuls with 0/1 matrices: one term per row, so exact.
        let bx = g.mul(b, xx)
        func shift(_ by: Int) throws -> V {
            var m = [Float16](repeating: 0, count: T * T)
            for t in by ..< T { m[t * T + t - by] = 1 }
            return g.matmul(try g.const(m, [1, 1, T, T]), bx)
        }
        let shifted = [try shift(2), try shift(1), bx]
        let taps = (0 ..< 3).map { j in g.mul(shifted[j], g.input("k\(j)", [1, 1, 1, D])) }
        r = linear(g.mul(c, g.add(g.add(taps[0], taps[1]), taps[2])), "o", out: D)
    }
    let h = g.add(x, r)
    let hn = norm(h, g.input("g_ffn", [1, 1, 1, D]), ones: onesD)
    let a = linear(hn, "w1", out: s.ff)
    let silu = g.binary("real_div", a, g.offset(g.unary("exp", g.scale(a, -1)), 1))
    let m = linear(g.mul(silu, linear(hn, "w3", out: s.ff)), "w2", out: D)
    return (try g.program(outputs: [g.output(g.add(h, m), "y"), g.output(r, "r"), g.output(m, "m")]), g.constants)
}
