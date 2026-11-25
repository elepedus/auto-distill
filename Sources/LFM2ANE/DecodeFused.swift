// Several consecutive LFM2 decoder layers for a single new token (T = 1) in ONE ANE program, with the weights either
//   .input: packed fp16 runtime inputs (the decode path of `decodeProgram`, but a few big inputs instead of ~12 per layer),
//   .fp16:  fp16 BLOBFILE constants (one shared weight file),
//   .pal2:  2-bit palettised constants: uint2 indices {0,1,2} = trit + 1 through one shared lookup table {-1, 0, 1, 0};
//           the per-row scale is applied after the matmul exactly as in `decodeProgram`.
// Inputs:  x [1,1,1,D]; cst [1,nConv,2,D] (row 0 = s0, row 1 = s1 of each conv layer of the group, in order);
//          kc, vc [1,nAttn*H,L,hd] (caches of the group's attention layers, H planes each); onehot, mask, cos, sin as in `decodeProgram`;
//          with `.input`: wA [1,n,rowsA,D], wB [1,n,D,FF], sm [1,n,smRows,FF] (see `packInputs`).
// Outputs: y [1,1,1,D]; ns [1,nConv,2,D] (rows s1, bx of each conv layer: the next cst); kn, vn [1,nAttn*H,1,hd].
// Slices of packed inputs are along channels (layer), height (row block) and a width offset of 0 only: width offsets above 4094 give inf on the M2.
import ANE
import Foundation

/// pal2conv / fp16conv: the same constants consumed by a 1x1 convolution (x reshaped to [1,K,1,1]) instead of a matmul.
public enum WeightMode: String, Sendable {
    case input, fp16, pal2, fp16conv, pal2conv
    var conv: Bool { self == .fp16conv || self == .pal2conv }
    var palette: Bool { self == .pal2 || self == .pal2conv }
}

/// One layer's parameters. Linear names: conv b c x o, attention q k v o (GQA already expanded to 16 heads), then w1 w3 w2.
public struct LayerData: Sendable {
    public var trits: [String: [Int8]], scale: [String: [Float16]], small: [String: [Float16]]
    public init(trits: [String: [Int8]], scale: [String: [Float16]], small: [String: [Float16]]) { self.trits = trits; self.scale = scale; self.small = small }
}

public func fusedLinearNames(attention: Bool) -> [String] { (attention ? ["q", "k", "v", "o"] : ["b", "c", "x", "o"]) + ["w1", "w3", "w2"] }
/// Row order of the packed small vectors `sm`: the seven scales, the two norm gains, then conv taps k0..k2 or the q/k norm gains.
public func fusedSmallKeys(attention: Bool) -> [String] {
    fusedLinearNames(attention: attention).map { "s_" + $0 } + ["g_op", "g_ffn"] + (attention ? ["g_q", "g_k"] : ["k0", "k1", "k2"])
}
public let fusedSmallRows = 12

/// Packs the group's layers for `.input` mode: wA rows [q|b, k|c, v|x, o, w1, w3] (4 D + 2 FF rows of D), wB = w2 [D, FF], sm rows per `fusedSmallKeys`.
public func packInputs(_ layers: [LayerData], attention: [Bool], dim D: Int, ff FF: Int) -> (wA: [Float16], wB: [Float16], sm: [Float16]) {
    var wA = [Float16](), wB = [Float16](), sm = [Float16]()
    for (data, attn) in zip(layers, attention) {
        let names = fusedLinearNames(attention: attn)
        for name in names.dropLast() { wA += data.trits[name]!.map { Float16($0) } }
        wB += data.trits["w2"]!.map { Float16($0) }
        var rows = 0
        for key in fusedSmallKeys(attention: attn) {
            let v = key.hasPrefix("s_") ? data.scale[String(key.dropFirst(2))]! : data.small[key]!
            sm += v + [Float16](repeating: 0, count: FF - v.count); rows += 1
        }
        sm += [Float16](repeating: 0, count: (fusedSmallRows - rows) * FF)
    }
    return (wA, wB, sm)
}

private func pack2(_ trits: [Int8]) -> Data {
    var out = [UInt8](repeating: 0, count: (trits.count + 3) / 4)
    for (i, t) in trits.enumerated() { out[i >> 2] |= UInt8(t + 1) << UInt8((i & 3) * 2) }
    return Data(out)
}

public func fusedDecodeProgram(_ s: LFM2Shape, attention: [Bool], maxLength L: Int, mode: WeightMode, layers: [LayerData]? = nil) throws
    -> (Program, constants: [String: Tensor]) {
    let g = MILText(), D = s.dim, FF = s.ff, H = s.heads, hd = s.headDim, n = attention.count
    let nConv = attention.filter { !$0 }.count, nAttn = n - nConv
    func ones(_ k: Int) -> [Float16] { .init(repeating: 1, count: k) }
    func constant(_ values: [Float16], _ shape: [Int]) throws -> V { mode == .input ? try g.const(values, shape) : g.blobFP16(values, shape) }
    let onesD = try constant(ones(D), [1, 1, D, 1])
    let x0 = g.input("x", [1, 1, 1, D])
    let cst = nConv > 0 ? g.input("cst", [1, nConv, 2, D]) : nil
    var kc: V?, vc: V?, onehot: V?, mask: V?, cosI: V?, sinI: V?, onesH: V?, onesL: V?, rotC: V?
    if nAttn > 0 {
        kc = g.input("kc", [1, nAttn * H, L, hd]); vc = g.input("vc", [1, nAttn * H, L, hd])
        onehot = g.transpose(g.input("onehot", [1, 1, 1, L]), [0, 1, 3, 2]); mask = g.input("mask", [1, 1, 1, L])
        cosI = g.input("cos", [1, 1, 1, hd]); sinI = g.input("sin", [1, 1, 1, hd])
        onesH = try constant(ones(H * hd), [1, H, hd, 1]); onesL = try constant(ones(H * L), [1, H, L, 1])
        var rot = [Float16](repeating: 0, count: H * hd * hd)
        for h in 0 ..< H { for i in 0 ..< hd / 2 { rot[h * hd * hd + (i + hd / 2) * hd + i] = -1; rot[h * hd * hd + i * hd + (i + hd / 2)] = 1 } }
        rotC = try constant(rot, [1, H, hd, hd])
    }
    let rowsA = 4 * D + 2 * FF
    var wA: V?, wB: V?, sm: V?, lut: V?
    switch mode {
    case .input:
        wA = g.input("wA", [1, n, rowsA, D]); wB = g.input("wB", [1, n, D, FF]); sm = g.input("sm", [1, n, fusedSmallRows, FF])
    case .pal2, .pal2conv: lut = g.blobFP16([-1, 0, 1, 0], [1, 1, 1, 1, 4, 1])
    case .fp16, .fp16conv: break
    }
    func norm(_ x: V, _ gamma: V, ones: V) -> V {
        let m = g.floor(g.rowMax(g.unary("abs", x)), 0x1p-14), xs = g.binary("real_div", x, m)
        let meanSquare = g.scale(g.matmul(g.mul(xs, xs), ones), 1 / Float(x.shape[3]))
        let r = g.rsqrt(g.add(meanSquare, g.divide(1e-5, by: g.mul(m, m))))
        return g.mul(g.mul(xs, r), gamma)
    }
    func small(_ li: Int, _ key: String, _ width: Int) -> V {
        if mode == .input {
            let row = fusedSmallKeys(attention: attention[li]).firstIndex(of: key)!
            return g.slice(sm!, begin: [0, li, row, 0], size: [1, 1, 1, width])
        }
        let data = layers![li], v = key.hasPrefix("s_") ? data.scale[String(key.dropFirst(2))]! : data.small[key]!
        return g.blobFP16(v, [1, 1, 1, width])
    }
    func weight(_ li: Int, _ name: String, out: Int, in k: Int) -> V {
        switch mode {
        case .input:
            if name == "w2" { return g.slice(wB!, begin: [0, li, 0, 0], size: [1, 1, D, FF]) }
            let names = fusedLinearNames(attention: attention[li]), idx = names.firstIndex(of: name)!
            let off = idx < 4 ? idx * D : 4 * D + (idx - 4) * FF
            return g.slice(wA!, begin: [0, li, off, 0], size: [1, 1, out, D])
        case .fp16, .fp16conv: return g.blobFP16(layers![li].trits[name]!.map { Float16($0) }, mode.conv ? [out, k, 1, 1] : [1, 1, out, k])
        case .pal2, .pal2conv: return g.lutToDense(indices: g.blob(pack2(layers![li].trits[name]!), element: .uint2, type: "uint2", shape: mode.conv ? [out, k, 1, 1] : [1, 1, out, k]), lut: lut!)
        }
    }
    func linear(_ li: Int, _ x: V, _ name: String, out: Int) -> V {
        let w = weight(li, name, out: out, in: x.shape[3])
        let y = mode.conv ? g.reshape(g.conv1x1(g.reshape(x, [1, x.shape[3], 1, 1]), w), [1, 1, 1, out]) : g.matmul(x, w, transposeY: true)
        return g.mul(y, small(li, "s_" + name, out))
    }

    var x = x0, nsPieces: [V] = [], knPieces: [V] = [], vnPieces: [V] = []
    var ci = 0, ai = 0
    for li in 0 ..< n {
        let xn = norm(x, small(li, "g_op", D), ones: onesD)
        let r: V
        if attention[li] {
            func heads(_ z: V) -> V { g.reshape(z, [1, H, 1, hd]) }
            func rope(_ z: V) -> V { g.add(g.mul(z, cosI!), g.mul(g.matmul(z, rotC!), sinI!)) }
            let q = rope(norm(heads(linear(li, xn, "q", out: H * hd)), g.reshape(small(li, "g_q", H * hd), [1, H, 1, hd]), ones: onesH!))
            let k = rope(norm(heads(linear(li, xn, "k", out: H * hd)), g.reshape(small(li, "g_k", H * hd), [1, H, 1, hd]), ones: onesH!))
            let v = heads(linear(li, xn, "v", out: H * hd))
            knPieces.append(k); vnPieces.append(v)
            let kf = g.add(g.slice(kc!, begin: [0, ai * H, 0, 0], size: [1, H, L, hd]), g.mul(onehot!, k))
            let vf = g.add(g.slice(vc!, begin: [0, ai * H, 0, 0], size: [1, H, L, hd]), g.mul(onehot!, v))
            let scores = g.add(g.scale(g.matmul(q, kf, transposeY: true), 1 / sqrt(Float(hd))), mask!)
            let e = g.unary("exp", g.binary("sub", scores, g.rowMax(scores))), den = g.matmul(e, onesL!)
            let o = g.scale(g.binary("real_div", g.matmul(e, g.scale(vf, 0x1p8)), den), 0x1p-8)
            r = linear(li, g.reshape(o, [1, 1, 1, H * hd]), "o", out: D)
            ai += 1
        } else {
            let lb = linear(li, xn, "b", out: D), lc = linear(li, xn, "c", out: D), lx = linear(li, xn, "x", out: D)
            let bx = g.mul(lb, lx)
            let s0 = g.slice(cst!, begin: [0, ci, 0, 0], size: [1, 1, 1, D]), s1 = g.slice(cst!, begin: [0, ci, 1, 0], size: [1, 1, 1, D])
            let k = (0 ..< 3).map { small(li, "k\($0)", D) }
            let conv = g.add(g.add(g.mul(s0, k[0]), g.mul(s1, k[1])), g.mul(bx, k[2]))
            nsPieces.append(g.concat([s1, bx], axis: 2))
            r = linear(li, g.mul(lc, conv), "o", out: D)
            ci += 1
        }
        let h = g.add(x, r)
        let hn = norm(h, small(li, "g_ffn", D), ones: onesD)
        let l1 = linear(li, hn, "w1", out: FF), l3 = linear(li, hn, "w3", out: FF)
        let silu = g.binary("real_div", l1, g.offset(g.unary("exp", g.scale(l1, -1)), 1))
        x = g.add(h, linear(li, g.mul(silu, l3), "w2", out: D))
    }
    var outputs = [g.output(x, "y")]
    if nConv > 0 { outputs.append(g.output(nsPieces.count == 1 ? nsPieces[0] : g.concat(nsPieces, axis: 1), "ns")) }
    if nAttn > 0 {
        outputs.append(g.output(knPieces.count == 1 ? knPieces[0] : g.concat(knPieces, axis: 1), "kn"))
        outputs.append(g.output(vnPieces.count == 1 ? vnPieces[0] : g.concat(vnPieces, axis: 1), "vn"))
    }
    return (try g.program(outputs: outputs), g.constants)
}
