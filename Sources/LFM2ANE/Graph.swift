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
    func const(_ values: [Float16], _ shape: [Int], name: String? = nil) throws -> V {
        let name = name ?? "c\(constants.count)"
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
    func matmul(_ x: V, _ y: V, transposeX: Bool = false, transposeY: Bool = false) -> V {
        let m = transposeX ? x.shape[3] : x.shape[2], n = transposeY ? y.shape[2] : y.shape[3]
        return op("matmul", "transpose_x = \(scalar("bool", "\(transposeX)")), transpose_y = \(scalar("bool", "\(transposeY)")), x = \(x.name), y = \(y.name)",
                  Array(x.shape.prefix(2)) + [m, n])
    }
    func rowMax(_ x: V) -> V {
        op("reduce_max", "axes = \(ints([-1])), keep_dims = \(scalar("bool", "true")), x = \(x.name)", Array(x.shape.dropLast()) + [1])
    }
    func floor(_ x: V, _ value: Float) -> V { op("maximum", "x = \(x.name), y = \(half(value))", x.shape) }
    func rsqrt(_ x: V) -> V { op("rsqrt", "epsilon = \(half(0x1p-24)), x = \(x.name)", x.shape) }
    func reshape(_ x: V, _ shape: [Int]) -> V { op("reshape", "shape = \(ints(shape)), x = \(x.name)", shape) }
    func slice(_ x: V, begin: [Int], size: [Int]) -> V { op("slice_by_size", "x = \(x.name), begin = \(ints(begin)), size = \(ints(size))", size) }
    func transpose(_ x: V, _ perm: [Int]) -> V { op("transpose", "perm = \(ints(perm)), x = \(x.name)", perm.map { x.shape[$0] }) }
    func output(_ x: V, _ name: String) -> V { op("identity", "x = \(x.name)", x.shape, as: name) }

    func program(outputs: [V]) throws -> Program {
        let text = "program(1.3)\n{\n    func main<ios18>(\(signature.joined(separator: ", "))) {\n"
            + lines.map { "        " + $0 }.joined(separator: "\n")
            + "\n    } -> (\(outputs.map(\.name).joined(separator: ", ")));\n}\n"
        return try Program(mil: text)
    }
}

/// Layer shape. `population` runs that many independent models side by side in one program: every tensor
/// gets a population axis (the channel axis, which matmuls batch over), each member with its own weights.
public struct LFM2Shape: Sendable {
    public let tokens: Int, dim: Int, ff: Int, heads: Int, headDim: Int, ropeBase: Float, population: Int
    public init(tokens: Int, dim: Int, ff: Int, heads: Int, headDim: Int, ropeBase: Float, population: Int = 1) {
        self.tokens = tokens; self.dim = dim; self.ff = ff; self.heads = heads; self.headDim = headDim; self.ropeBase = ropeBase
        self.population = population
    }
}

/// One LFM2 decoder layer: `h = x + mixer(norm(x))`, `y = h + mlp(norm(h))`. Outputs y, r (mixer), m (mlp).
///
/// With `backward`, the program also takes `gy` = dL/dy, recomputes the forward pass and returns `dx`, plus for
/// each ternary linear its two gradient factors, input activation `a_*` and output gradient times scale `gp_*`,
/// whose product gpᵀ·a is dL/dtrits, and the float-parameter gradients `ds_*`, `dg_*`, `dk*`.
///
/// Names and widths (columns per member) of the prescale factors, packed into the one runtime input `f` [1, P, 1, sum of widths].
public func prescaleNames(attention: Bool, heads: Int) -> [(name: String, width: Int)] {
    let linears = (attention ? ["q", "k", "v", "o"] : ["b", "c", "x", "o"]) + ["w1", "w3", "w2"]
    return linears.flatMap { [("gp_" + $0, 1), ("ds_" + $0, 1)] } + [("dg_op", 1), ("dg_ffn", 1)]
        + (attention ? [("dg_q", heads), ("dg_k", heads)] : (0 ..< 3).map { ("dk\($0)", 1) })
}
/// With `prescale`, each of those gradient outputs is multiplied by its own slice of the runtime input `f` (default 1), applied to the upstream gradient before any reduction.
public func layerProgram(_ s: LFM2Shape, attention: Bool, backward: Bool = false, diagnostics: Bool = true, prescale: Bool = false) throws
    -> (Program, constants: [String: Tensor]) {
    let g = MILText(), T = s.tokens, D = s.dim, H = s.heads, hd = s.headDim, P = s.population, PH = P * H
    func ones(_ n: Int) -> [Float16] { .init(repeating: 1, count: n) }
    let onesD = try g.const(ones(P * D), [1, P, D, 1])
    let tokenSum = backward ? try g.const(ones(P * T), [1, P, 1, T]) : onesD  // [1, T] @ z sums z over tokens
    var outputs: [V] = []
    var factor = [String: V]()  // output name -> its prescale constant
    if prescale && backward {  // all factors travel in one input: the ANE refuses programs with more than 32 inputs
        let packed = try g.const(ones(P * prescaleNames(attention: attention, heads: H).map(\.width).reduce(0, +)), [1, P, 1, prescaleNames(attention: attention, heads: H).map(\.width).reduce(0, +)], name: "f")
        var offset = 0
        for (name, width) in prescaleNames(attention: attention, heads: H) {
            let part = g.slice(packed, begin: [0, 0, 0, offset], size: [1, P, 1, width])
            factor[name] = width == 1 ? part : g.reshape(part, [1, PH, 1, 1])  // dg_q, dg_k: one factor per member and head
            offset += width
        }
    }
    func scaled(_ x: V, _ f: V?) -> V { f.map { g.mul(x, $0) } ?? x }
    func emit(_ x: V, _ name: String) { outputs.append(g.output(x, name)) }

    // RMSNorm as xs / rms(xs) with xs = x / max|x|: no finite input can overflow the squares, and the sum
    // of squares goes through a matmul (wide accumulator), not reduce_sum. eps is rescaled to match.
    struct Norm { let out, xhat, r, m, gamma, ones: V }
    func norm(_ x: V, _ gamma: V, ones: V) -> Norm {
        let m = g.floor(g.rowMax(g.unary("abs", x)), 0x1p-14), xs = g.binary("real_div", x, m)
        let meanSquare = g.scale(g.matmul(g.mul(xs, xs), ones), 1 / Float(x.shape[3]))
        let r = g.rsqrt(g.add(meanSquare, g.divide(1e-5, by: g.mul(m, m))))
        let xhat = g.mul(xs, r)
        return Norm(out: g.mul(xhat, gamma), xhat: xhat, r: r, m: m, gamma: gamma, ones: ones)
    }
    // d/dx of gamma·x·rs, rs = 1/sqrt(mean(x²) + eps) = r/m:  rs·(gxhat − xhat·mean(gxhat·xhat)).
    func normBack(_ n: Norm, _ gout: V, tokens: V, prescale f: V? = nil) -> (gx: V, dgamma: V) {
        let gxhat = g.mul(gout, n.gamma)
        let dot = g.scale(g.matmul(g.mul(gxhat, n.xhat), n.ones), 1 / Float(n.xhat.shape[3]))
        let gx = g.mul(g.binary("real_div", n.r, n.m), g.binary("sub", gxhat, g.mul(n.xhat, dot)))
        return (gx, g.matmul(tokens, g.mul(scaled(gout, f), n.xhat)))
    }
    struct Lin { let name: String; let x, w, s, pre, y: V }
    func linear(_ x: V, _ name: String, out: Int) -> Lin {
        let w = g.input("w_\(name)", [1, P, out, x.shape[3]]), scale = g.input("s_\(name)", [1, P, 1, out])
        let pre = g.matmul(x, w, transposeY: true)  // trits in MLX's [out, in] layout
        return Lin(name: name, x: x, w: w, s: scale, pre: pre, y: g.mul(pre, scale))
    }
    func linearBack(_ l: Lin, _ gy: V) -> V {  // emits gp_ and ds_, returns dL/dx (an exact ternary matmul)
        let gp = g.mul(gy, l.s)
        emit(factor["gp_" + l.name].map { g.mul(gy, g.mul(l.s, $0)) } ?? gp, "gp_\(l.name)")  // gy·(s·f): neither a premature underflow nor an overflow
        emit(g.matmul(tokenSum, g.mul(scaled(gy, factor["ds_" + l.name]), l.pre)), "ds_\(l.name)")
        return g.matmul(gp, l.w)
    }
    func sum(_ xs: [V]) -> V { xs.dropFirst().reduce(xs[0], g.add) }

    let x = g.input("x", [1, P, T, D]), gy = backward ? g.input("gy", [1, P, T, D]) : x
    let xn = norm(x, g.input("g_op", [1, P, 1, D]), ones: onesD)
    let r: V, mixerBack: (V) throws -> V  // dL/dr -> dL/dxn
    if attention {
        // Heads of all members share the channel axis: [1, P, T, H·hd] <-> [1, P·H, T, hd].
        func heads(_ z: V) -> V { g.reshape(g.transpose(g.reshape(z, [P, T, H, hd]), [0, 2, 1, 3]), [1, PH, T, hd]) }
        func merge(_ z: V) -> V { g.reshape(g.transpose(g.reshape(z, [P, H, T, hd]), [0, 2, 1, 3]), [1, P, T, H * hd]) }
        let onesH = try g.const(ones(PH * hd), [1, PH, hd, 1])
        var cosT = [Float16](), sinT = [Float16](), rot = [Float16](repeating: 0, count: PH * hd * hd)
        for _ in 0 ..< PH {
            for t in 0 ..< T {
                for i in 0 ..< hd {
                    let angle = Float(t) * pow(s.ropeBase, -Float(2 * (i % (hd / 2))) / Float(hd))
                    cosT.append(Float16(cos(angle))); sinT.append(Float16(sin(angle)))
                }
            }
        }
        for h in 0 ..< PH {  // rotate_half(q) = q @ R: column i takes -q[i + hd/2] (first half) or q[i - hd/2]
            for i in 0 ..< hd / 2 {
                rot[h * hd * hd + (i + hd / 2) * hd + i] = -1
                rot[h * hd * hd + i * hd + (i + hd / 2)] = 1
            }
        }
        let cosC = try g.const(cosT, [1, PH, T, hd]), sinC = try g.const(sinT, [1, PH, T, hd])
        let rotC = try g.const(rot, [1, PH, hd, hd])
        var mask = [Float16]()
        for _ in 0 ..< PH { for i in 0 ..< T { for j in 0 ..< T { mask.append(j <= i ? 0 : -30000) } } }
        let maskC = try g.const(mask, [1, PH, T, T]), onesT = try g.const(ones(PH * T), [1, PH, T, 1])
        func rope(_ z: V) -> V { g.add(g.mul(z, cosC), g.mul(g.matmul(z, rotC), sinC)) }

        // GQA as 16 full heads: the caller duplicates each kv head's weight rows.
        let lq = linear(xn.out, "q", out: H * hd), lk = linear(xn.out, "k", out: H * hd), lv = linear(xn.out, "v", out: H * hd)
        let qn = norm(heads(lq.y), g.input("g_q", [1, PH, 1, hd]), ones: onesH)
        let kn = norm(heads(lk.y), g.input("g_k", [1, PH, 1, hd]), ones: onesH)
        let q = rope(qn.out), k = rope(kn.out), v = heads(lv.y)
        let scores = g.add(g.scale(g.matmul(q, k, transposeY: true), 1 / sqrt(Float(hd))), maskC)
        let e = g.unary("exp", g.binary("sub", scores, g.rowMax(scores))), den = g.matmul(e, onesT)
        // V * 2^8 keeps small probability-weighted terms out of fp16 underflow (swift-ane attention study).
        let o = g.scale(g.binary("real_div", g.matmul(e, g.scale(v, 0x1p8)), den), 0x1p-8)
        let lo = linear(merge(o), "o", out: D)
        r = lo.y
        mixerBack = { gr in
            let headTokens = try g.const(ones(PH * T), [1, PH, 1, T])
            emit(lo.x, "a_mix")
            let gO = heads(linearBack(lo, gr)), p = g.binary("real_div", e, den)
            let gV = g.matmul(p, gO, transposeX: true), gP = g.matmul(gO, v, transposeY: true)
            let gS = g.scale(g.mul(p, g.binary("sub", gP, g.matmul(g.mul(gP, p), onesT))), 1 / sqrt(Float(hd)))
            func ropeBack(_ gz: V) -> V { g.binary("sub", g.mul(gz, cosC), g.matmul(g.mul(gz, sinC), rotC)) }  // Rᵀ = −R
            let (gq, dgq) = normBack(qn, ropeBack(g.matmul(gS, k)), tokens: headTokens, prescale: factor["dg_q"])
            let (gk, dgk) = normBack(kn, ropeBack(g.matmul(gS, q, transposeX: true)), tokens: headTokens, prescale: factor["dg_k"])
            emit(dgq, "dg_q"); emit(dgk, "dg_k")
            return sum([linearBack(lq, merge(gq)), linearBack(lk, merge(gk)), linearBack(lv, merge(gV))])
        }
    } else {
        let lb = linear(xn.out, "b", out: D), lc = linear(xn.out, "c", out: D), lx = linear(xn.out, "x", out: D)
        // Causal 3-tap conv over tokens. Row shifts are matmuls with 0/1 matrices: one term per row, so exact.
        func shiftMatrix(_ by: Int, transposed: Bool = false) throws -> V {
            var m = [Float16](repeating: 0, count: T * T)
            for t in by ..< T { if transposed { m[(t - by) * T + t] = 1 } else { m[t * T + t - by] = 1 } }
            return try g.const(Array([[Float16]](repeating: m, count: P).joined()), [1, P, T, T])
        }
        let bx = g.mul(lb.y, lx.y)
        let shifted = [g.matmul(try shiftMatrix(2), bx), g.matmul(try shiftMatrix(1), bx), bx]
        let k = (0 ..< 3).map { g.input("k\($0)", [1, P, 1, D]) }
        let conv = sum((0 ..< 3).map { g.mul(shifted[$0], k[$0]) })
        let lo = linear(g.mul(lc.y, conv), "o", out: D)
        r = lo.y
        mixerBack = { gr in
            emit(lo.x, "a_mix")
            let gz = linearBack(lo, gr), gconv = g.mul(gz, lc.y)
            for j in 0 ..< 3 { emit(g.matmul(tokenSum, g.mul(scaled(gconv, factor["dk\(j)"]), shifted[j])), "dk\(j)") }
            let gbx = sum([g.matmul(try shiftMatrix(2, transposed: true), g.mul(gconv, k[0])),
                           g.matmul(try shiftMatrix(1, transposed: true), g.mul(gconv, k[1])), g.mul(gconv, k[2])])
            return sum([linearBack(lb, g.mul(gbx, lx.y)), linearBack(lx, g.mul(gbx, lb.y)), linearBack(lc, g.mul(gz, conv))])
        }
    }
    let h = g.add(x, r)
    let hn = norm(h, g.input("g_ffn", [1, P, 1, D]), ones: onesD)
    let l1 = linear(hn.out, "w1", out: s.ff), l3 = linear(hn.out, "w3", out: s.ff)
    let silu = g.binary("real_div", l1.y, g.offset(g.unary("exp", g.scale(l1.y, -1)), 1))  // explicit: native silu is inaccurate
    let l2 = linear(g.mul(silu, l3.y), "w2", out: D)
    let y = g.add(h, l2.y)
    emit(y, "y")
    guard backward else {
        if diagnostics { emit(r, "r"); emit(l2.y, "m") }
        return (try g.program(outputs: outputs), g.constants)
    }
    emit(xn.out, "a_xn"); emit(hn.out, "a_hn"); emit(l2.x, "a_act")
    let gact = linearBack(l2, gy)
    let sig = g.divide(1, by: g.offset(g.unary("exp", g.scale(l1.y, -1)), 1))
    let dsilu = g.add(sig, g.mul(g.mul(l1.y, sig), g.offset(g.scale(sig, -1), 1)))  // σ(1 + a(1 − σ))
    let ghn = g.add(linearBack(l1, g.mul(g.mul(gact, l3.y), dsilu)), linearBack(l3, g.mul(gact, silu)))
    let (ghNorm, dgFfn) = normBack(hn, ghn, tokens: tokenSum, prescale: factor["dg_ffn"])
    emit(dgFfn, "dg_ffn")
    let gh = g.add(gy, ghNorm)
    let (gxNorm, dgOp) = normBack(xn, try mixerBack(gh), tokens: tokenSum, prescale: factor["dg_op"])
    emit(dgOp, "dg_op")
    emit(g.add(gh, gxNorm), "dx")
    return (try g.program(outputs: outputs), g.constants)
}
