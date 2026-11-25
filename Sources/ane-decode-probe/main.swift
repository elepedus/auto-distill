// Decode-floor probes for the ANE (T=1): dispatch floor, weight-read bandwidth, input-count limit, BLOBFILE constants.
//   ane-decode-probe floor      trivial matmul programs with m fp16 runtime-input weights of [N, 1024]
import ANE
import Foundation
import LFM2ANE
setvbuf(stdout, nil, _IOLBF, 0)

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
let K = 1024

func fill(_ b: SharedBuffer, seed: UInt64) {
    let p = b.pointer.assumingMemoryBound(to: Float16.self)
    var s = seed &* 6364136223846793005 &+ 1
    for i in 0 ..< b.bytes / 2 { s = s &* 6364136223846793005 &+ 1442695040888963407; p[i] = Float16(Float(Int64(bitPattern: s) >> 40) / Float(1 << 23) * 0.05) }
}

/// m matmuls x[1,1,1,K] @ w_i[1,1,N,K]^T summed into y [1,1,1,N].
func floorProgram(m: Int, n: Int) throws -> (Program, [String: ExternalInputBinding]) {
    var sig = ["tensor<fp16, [1, 1, 1, \(K)]> x"], lines = ["bool tf = const()[name = string(\"tf\"), val = bool(false)];", "bool tt = const()[name = string(\"tt\"), val = bool(true)];"]
    var bind: [String: ExternalInputBinding] = [:]
    let x = SharedBuffer([1, 1, 1, K]); fill(x, seed: 1); bind["x"] = try x.binding()
    var acc = ""
    for i in 0 ..< m {
        sig.append("tensor<fp16, [1, 1, \(n), \(K)]> w\(i)")
        let w = SharedBuffer([1, 1, n, K]); fill(w, seed: UInt64(i + 2)); bind["w\(i)"] = try w.binding()
        lines.append("tensor<fp16, [1, 1, 1, \(n)]> y\(i) = matmul(transpose_x = tf, transpose_y = tt, x = x, y = w\(i))[name = string(\"y\(i)\")];")
        if i == 0 { acc = "y0" } else { lines.append("tensor<fp16, [1, 1, 1, \(n)]> a\(i) = add(x = \(acc), y = y\(i))[name = string(\"a\(i)\")];"); acc = "a\(i)" }
    }
    lines.append("tensor<fp16, [1, 1, 1, \(n)]> y = identity(x = \(acc))[name = string(\"y\")];")
    let text = "program(1.3)\n{\n    func main<ios18>(\(sig.joined(separator: ", "))) {\n" + lines.map { "        " + $0 }.joined(separator: "\n") + "\n    } -> (y);\n}\n"
    return (try Program(mil: text), bind)
}

func time(_ model: Model, reps: Int = 200) throws -> (median: Double, wall: Double) {
    for _ in 0 ..< 20 { _ = try model.evaluatePrepared(readOutputs: false, poisonOutputs: false) }
    var ts: [Double] = []
    let t0 = now()
    for _ in 0 ..< reps { ts.append(try model.evaluatePrepared(readOutputs: false, poisonOutputs: false).elapsed) }
    ts.sort()
    return (ts[reps / 2] * 1000, (now() - t0) / Double(reps) * 1000)
}

let mode = CommandLine.arguments.dropFirst().first ?? "floor"
if mode == "blobs" { runBlobs(CommandLine.arguments.count > 2 ? [CommandLine.arguments[2]] : ["fp16", "pal2", "pal2t", "int8"]) }
if mode == "floor" {
    // 1 MB unit = 512 rows of K=1024 fp16
    for (m, n) in [(1, 16), (1, 512), (1, 2048), (4, 2048), (9, 2048), (16, 2048), (2, 512), (4, 512), (8, 512), (16, 512), (30, 512), (31, 512), (32, 512), (33, 512), (1, 8192), (2, 8192), (4, 8192), (8, 8192)] {
        do {
            let (p, b) = try floorProgram(m: m, n: n)
            let model = try Model(program: p, configuration: .init(maximumSurfaceBytes: 512 << 20, maximumTotalBytes: 4 << 30), inputBindings: b)
            _ = try model.prepareUpdating(inputs: [:])
            let (med, wall) = try time(model)
            let mb = Double(m * n * K * 2) / 1e6
            print(String(format: "m=%2d N=%5d  %7.1f MB  %.3f ms  wall %.3f  %.1f GB/s", m, n, mb, med, wall, mb / 1e3 / (med / 1e3)))
        } catch { print("m=\(m) N=\(n) FAILED: \(error)") }
    }
}
