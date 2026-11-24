// One LFM2 decoder layer for a single new token (T = 1), with its state as inputs and outputs.
// Weights, scales and norm gains are the same runtime inputs as `layerProgram`, so the same buffers serve both.
//   conv layer:      inputs s0, s1 [1,1,1,D] = bx of the two previous tokens; outputs ns0 = s1, ns1 = bx (the host copies them back).
//   attention layer: inputs kc, vc [1,H,L,hd] (post-rope keys and values, 16 heads, rows beyond the position zero), onehot [1,1,1,L]
//                    (1 at the current position), mask [1,1,1,L] (0 up to and including it, -30000 after), cos, sin [1,1,1,hd];
//                    outputs k_new, v_new [1,H,1,hd] (the host writes them into the caches at the position).
// All layers output y [1,1,1,D].
import ANE
import Foundation

public func decodeProgram(_ s: LFM2Shape, attention: Bool, maxLength L: Int, colscale: Bool = false) throws -> (Program, constants: [String: Tensor]) {
    let g = MILText(), D = s.dim, H = s.heads, hd = s.headDim
    func ones(_ n: Int) -> [Float16] { .init(repeating: 1, count: n) }
    let onesD = try g.const(ones(D), [1, 1, D, 1])
    var outputs: [V] = []
    var colSlice = [String: V]()
    if colscale {
        let names = colscaleNames(attention: attention, dim: D, ff: s.ff), total = names.map(\.width).reduce(0, +)
        let packed = g.input("cs", [1, 1, 1, total])
        var offset = 0
        for (name, width) in names { colSlice[name] = g.slice(packed, begin: [0, 0, 0, offset], size: [1, 1, 1, width]); offset += width }
    }
    // RMSNorm exactly as in layerProgram.
    func norm(_ x: V, _ gamma: V, ones: V) -> V {
        let m = g.floor(g.rowMax(g.unary("abs", x)), 0x1p-14), xs = g.binary("real_div", x, m)
        let meanSquare = g.scale(g.matmul(g.mul(xs, xs), ones), 1 / Float(x.shape[3]))
        let r = g.rsqrt(g.add(meanSquare, g.divide(1e-5, by: g.mul(m, m))))
        return g.mul(g.mul(xs, r), gamma)
    }
    func linear(_ x: V, _ name: String, out: Int) -> V {
        let w = g.input("w_\(name)", [1, 1, out, x.shape[3]]), scale = g.input("s_\(name)", [1, 1, 1, out])
        return g.mul(g.matmul(colSlice[name].map { g.mul(x, $0) } ?? x, w, transposeY: true), scale)
    }
    let x = g.input("x", [1, 1, 1, D])
    let xn = norm(x, g.input("g_op", [1, 1, 1, D]), ones: onesD)
    let r: V
    if attention {
        let onesH = try g.const(ones(H * hd), [1, H, hd, 1]), onesL = try g.const(ones(H * L), [1, H, L, 1])
        var rot = [Float16](repeating: 0, count: H * hd * hd)
        for h in 0 ..< H { for i in 0 ..< hd / 2 { rot[h * hd * hd + (i + hd / 2) * hd + i] = -1; rot[h * hd * hd + i * hd + (i + hd / 2)] = 1 } }
        let rotC = try g.const(rot, [1, H, hd, hd])
        let cosI = g.input("cos", [1, 1, 1, hd]), sinI = g.input("sin", [1, 1, 1, hd])
        func heads(_ z: V) -> V { g.reshape(z, [1, H, 1, hd]) }
        func rope(_ z: V) -> V { g.add(g.mul(z, cosI), g.mul(g.matmul(z, rotC), sinI)) }
        let q = rope(norm(heads(linear(xn, "q", out: H * hd)), g.input("g_q", [1, H, 1, hd]), ones: onesH))
        let k = rope(norm(heads(linear(xn, "k", out: H * hd)), g.input("g_k", [1, H, 1, hd]), ones: onesH))
        let v = heads(linear(xn, "v", out: H * hd))
        outputs.append(g.output(k, "k_new")); outputs.append(g.output(v, "v_new"))
        let onehot = g.transpose(g.input("onehot", [1, 1, 1, L]), [0, 1, 3, 2]), mask = g.input("mask", [1, 1, 1, L])
        let kf = g.add(g.input("kc", [1, H, L, hd]), g.mul(onehot, k)), vf = g.add(g.input("vc", [1, H, L, hd]), g.mul(onehot, v))
        let scores = g.add(g.scale(g.matmul(q, kf, transposeY: true), 1 / sqrt(Float(hd))), mask)
        let e = g.unary("exp", g.binary("sub", scores, g.rowMax(scores))), den = g.matmul(e, onesL)
        let o = g.scale(g.binary("real_div", g.matmul(e, g.scale(vf, 0x1p8)), den), 0x1p-8)
        r = linear(g.reshape(o, [1, 1, 1, H * hd]), "o", out: D)
    } else {
        let lb = linear(xn, "b", out: D), lc = linear(xn, "c", out: D), lx = linear(xn, "x", out: D)
        let bx = g.mul(lb, lx), s0 = g.input("s0", [1, 1, 1, D]), s1 = g.input("s1", [1, 1, 1, D])
        let k = (0 ..< 3).map { g.input("k\($0)", [1, 1, 1, D]) }
        let conv = g.add(g.add(g.mul(s0, k[0]), g.mul(s1, k[1])), g.mul(bx, k[2]))
        outputs.append(g.output(s1, "ns0")); outputs.append(g.output(bx, "ns1"))
        r = linear(g.mul(lc, conv), "o", out: D)
    }
    let h = g.add(x, r)
    let hn = norm(h, g.input("g_ffn", [1, 1, 1, D]), ones: onesD)
    let l1 = linear(hn, "w1", out: s.ff), l3 = linear(hn, "w3", out: s.ff)
    let silu = g.binary("real_div", l1, g.offset(g.unary("exp", g.scale(l1, -1)), 1))
    outputs.insert(g.output(g.add(h, linear(g.mul(silu, l3), "w2", out: D)), "y"), at: 0)
    return (try g.program(outputs: outputs), g.constants)
}
