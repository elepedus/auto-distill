// BLOBFILE constants in one in-memory program. kind: fp16 | pal2 (per-row 4-entry LUT, 2-bit indices) | pal2t (one shared LUT) | int8 (blockwise shift scale).
// Conv 1x1 form: x [1,K,1,1], w [N,K,1,1].
import ANE
import Foundation
import LFM2ANE

let header = "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"

@MainActor func blobProgram(m: Int, n: Int, kind: String, buildInfo: Bool) throws -> (Program, [String: ExternalInputBinding]) {
    var lines = ["string pt = const()[name = string(\"pt\"), val = string(\"valid\")];", "tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];",
                 "tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];", "tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];",
                 "int32 gr = const()[name = string(\"gr\"), val = int32(1)];", "bool tf = const()[name = string(\"tf\"), val = bool(false)];", "bool tt = const()[name = string(\"tt\"), val = bool(true)];"]
    var files: [String: Data] = [:]
    let sharedFile = ProcessInfo.processInfo.environment["SHARED"] != nil
    var chunks: [MILBlob.Chunk] = [], pos = 64
    var acc = ""
    var rng: UInt64 = 12345
    func rnd() -> UInt64 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; return rng >> 33 }
    let mm = ProcessInfo.processInfo.environment["MM"] != nil, W = Int(ProcessInfo.processInfo.environment["W"] ?? "1")!
    let wshape = mm ? "[1, 1, \(n), \(K)]" : "[\(n), \(K), 1, 1]"
    for i in 0 ..< m {
        func blob(_ name: String, _ type: String, _ shape: String, _ payload: Data, _ el: MILBlob.Element, pad: Int = 0) throws {
            if sharedFile {  // one file, chunk headers at 64-aligned offsets
                let offset = pos
                chunks.append(MILBlob.Chunk(payload: payload, element: el, paddingBits: pad))
                pos = MILBlob.nextOffset(after: offset, payloadCount: payload.count)
                lines.append("tensor<\(type), \(shape)> \(name) = const()[name = string(\"\(name)\"), val = tensor<\(type), \(shape)>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(offset))))];")
                return
            }
            files["weights/\(name).bin"] = try MILBlob.encode(payload: payload, element: el, paddingBits: pad)
            lines.append("tensor<\(type), \(shape)> \(name) = const()[name = string(\"\(name)\"), val = tensor<\(type), \(shape)>(BLOBFILE(path = string(\"@model_path/weights/\(name).bin\"), offset = uint64(64)))];")
        }
        let wname = "w\(i)"
        switch kind {
        case "fp16":
            var d = Data(count: n * K * 2)
            d.withUnsafeMutableBytes { let p = $0.bindMemory(to: Float16.self); for j in 0 ..< n * K { p[j] = Float16(Float(Int(rnd() % 3) - 1) * 0.05) } }
            try blob(wname, "fp16", wshape, d, .float16)
        case "pal2", "pal2t":
            var idx = [Int](repeating: 0, count: n * K)
            for j in 0 ..< n * K { idx[j] = Int(rnd() % 3) }
            try blob("ix\(i)", "uint2", wshape, Quantization.pack(idx, bits: 2), .uint2)
            let rows = kind == "pal2" ? n : 1
            var lut = Data()
            for r in 0 ..< rows { let s = Float16(0.05 * (1 + Float(r % 7) / 10)); for v in [-s, 0, s, 0] { lut.append(contentsOf: [UInt8(truncatingIfNeeded: v.bitPattern), UInt8(truncatingIfNeeded: v.bitPattern >> 8)]) } }
            try blob("lut\(i)", "fp16", "[\(rows), 1, 1, 1, 4, 1]", lut, .float16)
            lines.append("tensor<fp16, \(wshape)> \(wname) = constexpr_lut_to_dense(indices = ix\(i), lut = lut\(i))[name = string(\"\(wname)\")];")
        case "int8":
            var q = Data(count: n * K)
            q.withUnsafeMutableBytes { let p = $0.bindMemory(to: Int8.self); for j in 0 ..< n * K { p[j] = Int8(Int(rnd() % 3) - 1) } }
            try blob("q\(i)", "int8", wshape, q, .int8)
            var s = Data()
            for r in 0 ..< n { let v = Float16(0.05 * (1 + Float(r % 7) / 10)); s.append(contentsOf: [UInt8(truncatingIfNeeded: v.bitPattern), UInt8(truncatingIfNeeded: v.bitPattern >> 8)]) }
            try blob("s\(i)", "fp16", "[\(n), 1, 1, 1]", s, .float16)
            lines.append("tensor<fp16, \(wshape)> \(wname) = constexpr_blockwise_shift_scale(data = q\(i), scale = s\(i))[name = string(\"\(wname)\")];")
        default: fatalError()
        }
        if mm {
            lines.append("tensor<fp16, [1, 1, 1, \(n)]> y\(i) = matmul(transpose_x = tf, transpose_y = tt, x = x, y = \(wname))[name = string(\"y\(i)\")];")
        } else {
            lines.append("tensor<fp16, [1, \(n), 1, \(W)]> y\(i) = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = \(wname), x = x)[name = string(\"y\(i)\")];")
        }
        if i == 0 { acc = "y0" } else { lines.append("tensor<fp16, \(mm ? "[1, 1, 1, \(n)]" : "[1, \(n), 1, \(W)]")> a\(i) = add(x = \(acc), y = y\(i))[name = string(\"a\(i)\")];"); acc = "a\(i)" }
    }
    if sharedFile {
        files["weights/weight.bin"] = try MILBlob.encode(chunks: chunks).data
    }
    lines.append("tensor<fp16, \(mm ? "[1, 1, 1, \(n)]" : "[1, \(n), 1, \(W)]")> y = identity(x = \(acc))[name = string(\"y\")];")
    let text = (buildInfo ? header : "program(1.3)\n{\n") + "    func main<ios18>(tensor<fp16, \(mm ? "[1, 1, 1, \(K)]" : "[1, \(K), 1, \(W)]")> x) {\n" + lines.map { "        " + $0 }.joined(separator: "\n") + "\n    } -> (y);\n}\n"
    
    return (try Program(mil: text, weights: files), [:])
}

@MainActor func runBlobs(_ kinds: [String], buildInfos: [Bool] = [true, false], ms: [Int] = [1, 2, 4, 9, 16]) {
    let mm = ProcessInfo.processInfo.environment["MM"] != nil, W = Int(ProcessInfo.processInfo.environment["W"] ?? "1")!
    for kind in kinds {
        for m in ms {
            for bi in buildInfos {
                let n = Int(ProcessInfo.processInfo.environment["N"] ?? "2048")!
                do {
                    let t0 = now()
                    let (p, b) = try blobProgram(m: m, n: n, kind: kind, buildInfo: bi)
                    let model = try Model(program: p, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 4 << 30), inputBindings: b)
                    _ = try model.prepareUpdating(inputs: ["x": try Tensor(shape: mm ? TensorShape(batches: 1, channels: 1, height: 1, width: K) : TensorShape(batches: 1, channels: K, height: 1, width: W), values: [Float16](repeating: 0.1, count: K * W))])
                    let ct = now() - t0
                    let (med, wall) = try time(model)
                    let mb = Double(m * n * K * 2) / 1e6
                    print(String(format: "%@ buildInfo=%@ m=%2d N=%d  fp16-equiv %6.1f MB  %.3f ms (wall %.3f)  compile %.1fs", kind, bi ? "y" : "n", m, n, mb, med, wall, ct))
                } catch { print("\(kind) buildInfo=\(bi) m=\(m) N=\(n) FAILED: \(error)") }
            }
        }
    }
}
