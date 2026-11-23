// Is a ternary matmul on the ANE exact up to one final fp16 rounding?
// One program computes y = x·W (forward) and dx = g·Wᵀ (backward) with W in {-1,0,1} fed at runtime.
// Reference: the exact sum (Double is exact here), rounded once to fp16. Compared bit for bit.
import ANE
import Accelerate
import Foundation

struct Rng {
    var s: UInt64
    mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
    mutating func int(_ lo: Int, _ hi: Int) -> Int { lo + Int(next() % UInt64(hi - lo + 1)) }
    mutating func normal() -> Double {  // Box-Muller
        let u = (Double(next() >> 11) + 0.5) / 0x1p53, v = Double(next() >> 11) / 0x1p53
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
}

// Power-of-two prescale so the worst case 127·K stays under 16384, half the measured ~32752 matmul cliff.
func shift(_ k: Int) -> Int { max(0, Int(ceil(log2(127 * Double(k) / 16384)))) }

func exact(_ a: [Double], _ b: [Double], m: Int, k: Int, n: Int, transB: Bool) -> [Double] {
    var c = [Double](repeating: 0, count: m * n)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, transB ? CblasTrans : CblasNoTrans, Int32(m), Int32(n), Int32(k),
                1, a, Int32(k), b, Int32(transB ? k : n), 0, &c, Int32(n))
    return c
}

// Round once to fp16 under a given tie rule (only exact midpoints differ between rules).
enum Tie: String, CaseIterable { case even, awayFromZero, towardPlusInf, towardZero }
func round(_ d: Double, _ tie: Tie) -> Float16 {
    let f = Float16(d)  // nearest, ties to even
    guard f.isFinite, Double(f) != d else { return f }
    let (lo, hi) = Double(f) < d ? (f, f.nextUp) : (f.nextDown, f)
    guard d == (Double(lo) + Double(hi)) / 2 else { return f }
    switch tie {
    case .even: return f
    case .awayFromZero: return abs(lo) > abs(hi) ? lo : hi
    case .towardPlusInf: return hi
    case .towardZero: return abs(lo) < abs(hi) ? lo : hi
    }
}

func compare(_ got: [Float16], _ exact: [Double]) -> String {
    let finite = zip(got, exact).filter { $0.0.isFinite }
    let ties = exact.filter { d in let f = Float16(d); return f.isFinite && Double(f) != d && round(d, .awayFromZero) != round(d, .towardZero) }.count
    let rules = Tie.allCases.map { tie in
        String(format: "%@ %.3f%%", tie.rawValue, 100 * Double(finite.filter { $0.0 == round($0.1, tie) }.count) / Double(exact.count))
    }
    let worst = finite.map { abs(Double($0.0) - $0.1) / Double(Float16($0.1).ulp) }.max() ?? 0
    return String(format: "ties %5.2f%%  nonfinite %d  max %.2f ulp | ", 100 * Double(ties) / Double(exact.count),
                  got.count - finite.count, worst) + rules.joined(separator: "  ")
}

func run(_ label: String, m: Int, k: Int, n: Int, prescale: Bool = true,
         act: (inout Rng) -> Double, trit: (inout Rng) -> Double) throws {
    var rng = Rng(s: 0x9E3779B97F4A7C15)
    let sx = prescale ? Double(1 << shift(k)) : 1, sg = prescale ? Double(1 << shift(n)) : 1
    let fp16 = { (d: Double) in Double(Float16(d)) }  // the reference sees exactly what the ANE sees
    let xv = (0..<m * k).map { _ in fp16(act(&rng) / sx) }
    let gv = (0..<m * n).map { _ in fp16(act(&rng) / sg) }
    let wv = (0..<k * n).map { _ in trit(&rng) }

    let mil = MIL()
    let x = try mil.input("x", shape: TensorShape(channels: 1, height: m, width: k))
    let w = try mil.input("w", shape: TensorShape(channels: 1, height: k, width: n))
    let g = try mil.input("g", shape: TensorShape(channels: 1, height: m, width: n))
    let y = try mil.matmul(x, w), dx = try mil.matmul(g, w, transposeY: true)
    let model = try Model(program: mil.finish(outputs: [y, dx]))
    defer { try? model.close() }

    let half = { (v: [Double]) in v.map { Float16($0) } }
    let obs = try model.run(inputs: [
        "x": Tensor(shape: TensorShape(channels: 1, height: m, width: k), values: half(xv)),
        "w": Tensor(shape: TensorShape(channels: 1, height: k, width: n), values: half(wv)),
        "g": Tensor(shape: TensorShape(channels: 1, height: m, width: n), values: half(gv)),
    ])
    func output(_ v: Value) throws -> [Float16] {
        let name = model.outputs.map(\.name).first { $0 == v.name || $0.hasPrefix(v.name + "@") }!
        return try obs.outputs[name]!.float16Values()
    }
    for (dir, got, want) in [("fwd", try output(y), exact(xv, wv, m: m, k: k, n: n, transB: false)),
                             ("bwd", try output(dx), exact(gv, wv, m: m, k: n, n: k, transB: true))] {
        print(String(format: "%-20@ %@ K=%4d  ", label, dir, dir == "fwd" ? k : n) + compare(got, want))
    }
}

let trit = { (r: inout Rng) in Double(r.int(-1, 1)) }
let code = { (r: inout Rng) in Double(r.int(-127, 127)) }
for (k, n) in [(1024, 4608), (4608, 1024)] {
    try run("int8 codes × trits", m: 256, k: k, n: n, act: code, trit: trit)
    try run("fp16 normal × trits", m: 256, k: k, n: n, act: { r in (r.normal() * 64).rounded() / 64 * 127 / 4 }, trit: trit)
    try run("worst 127×+1", m: 64, k: k, n: n, act: { _ in 127 }, trit: { _ in 1 })
    try run("worst, no prescale", m: 64, k: k, n: n, prescale: false, act: { _ in 127 }, trit: { _ in 1 })
}
