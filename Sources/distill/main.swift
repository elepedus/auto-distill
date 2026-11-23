// Dataset-free distillation. The student writes text, the teacher scores every token, and the
// student learns the teacher's full next-token distribution. Nothing is stored.
//
//   distill train TEACHER STUDENT OUT [--steps 300] [--batch 8] [--seq 256] [--lr 1e-5]
//   distill eval TEXTFILE TEACHER MODEL...
//
// Models are local directories or Hugging Face ids already in the local cache.
// A quantised student trains only its scales and biases; a full-precision student trains every weight.
import Foundation
import LMLoad
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import Tokenizers

let load = loadLLM, directory = modelDirectory

func option(_ name: String, _ fallback: Double) -> Double {
    guard let i = CommandLine.arguments.firstIndex(of: "--\(name)"), i + 1 < CommandLine.arguments.count,
          let value = Double(CommandLine.arguments[i + 1]) else { return fallback }
    return value
}

func logProbs(_ model: any LLMModel, _ x: MLXArray) -> MLXArray {
    logSoftmax(model(x, cache: nil).asType(.float32), axis: -1)
}

func kl(_ teacher: MLXArray, _ student: MLXArray) -> MLXArray {
    (exp(teacher) * (teacher - student)).sum(axis: -1).mean()
}

func train(_ args: [String]) async throws {
    let steps = Int(option("steps", 300)), batch = Int(option("batch", 8)), seq = Int(option("seq", 256))
    let (teacher, _) = try await load(args[0])
    let (student, studentDir) = try await load(args[1])
    let out = URL(filePath: args[2])

    student.freeze()
    if student.leafModules().flattened().contains(where: { $0.1 is QuantizedLinear }) {
        student.unfreeze(keys: ["scales", "biases"])
    } else {
        student.unfreeze()  // ponytail: full fine-tune, switch to LoRA when fp32 Adam state stops fitting
    }
    let original = student.trainableParameters()
    student.update(parameters: original.mapValues { $0.asType(.float32) })  // bf16 swallows small updates
    let names = Set(original.flattened().map { $0.0.split(separator: ".").last! })
    print("training \(original.flattened().map(\.1.size).reduce(0, +)) parameters: \(names.sorted().joined(separator: ", "))")

    let vocab = teacher(MLXArray([Int32(0)])[.newAxis], cache: nil).dim(-1)
    func sample() throws -> MLXArray {
        // random first token, then free-running sampling (as in LLM-QAT): the model supplies its own data
        var x = MLXRandom.randInt(0 ..< vocab, [batch, 1]).asType(.int32)
        var tokens = [x]
        let cache = try student.newCache(parameters: nil)
        for _ in 1 ..< seq {
            x = categorical(student(x, cache: cache)[0..., -1, 0...])[0..., .newAxis].asType(.int32)
            eval(x)
            tokens.append(x)
        }
        return concatenated(tokens, axis: 1)
    }

    let optimizer = Adam(learningRate: Float(option("lr", 1e-5)))  // 1e-4 wrecks 3-bit scales within 3 steps
    let step = valueAndGrad(model: student) { model, arrays in
        [kl(arrays[1], logProbs(model as! any LLMModel, arrays[0]))]
    }
    for i in 0 ..< steps {
        let start = Date()
        let x = try sample()  // on-policy: the teacher only prefills, never decodes
        let (loss, grads) = step(student, [x, logProbs(teacher, x)])
        optimizer.update(model: student, gradients: grads)
        eval(student, optimizer, loss)
        let seconds = Date().timeIntervalSince(start)
        print(String(format: "step %d kl %.4f %.2fs %.0f tok/s", i, loss[0].item(Float.self), seconds,
                     Double(batch * seq) / seconds))
    }
    print(String(format: "peak memory %.1f GB", Double(Memory.peakMemory) / 1e9))

    let trained = student.trainableParameters().flattened()
    let dtypes = Dictionary(uniqueKeysWithValues: original.flattened().map { ($0.0, $0.1.dtype) })
    student.update(parameters: .unflattened(trained.map { ($0.0, $0.1.asType(dtypes[$0.0]!)) }))
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    for file in try FileManager.default.contentsOfDirectory(atPath: studentDir.path)
    where !file.hasSuffix(".safetensors") && !file.hasSuffix(".safetensors.index.json") {
        try? FileManager.default.removeItem(at: out.appending(path: file))
        try FileManager.default.copyItem(at: studentDir.appending(path: file), to: out.appending(path: file))
    }
    try save(arrays: Dictionary(uniqueKeysWithValues: student.parameters().flattened()),
             url: out.appending(path: "model.safetensors"))
}

// Perplexity and KL to the teacher over 64 windows of 512 tokens (BOS + 511), as in the README table.
func evaluate(_ args: [String]) async throws {
    let teacherDir = try directory(args[1])
    let tokenizer = try await AutoTokenizer.from(modelFolder: teacherDir)
    let text = try String(contentsOf: URL(filePath: args[0]), encoding: .utf8)
    let ids = tokenizer.encode(text: text, addSpecialTokens: false).prefix(64 * 511).map(Int32.init)
    let bos = MLXArray.full([64, 1], values: MLXArray(Int32(tokenizer.bosTokenId ?? 1)))
    let x = concatenated([bos, MLXArray(ids, [64, 511])], axis: 1)
    func windows(_ model: any LLMModel) -> MLXArray {
        concatenated(stride(from: 0, to: 64, by: 8).map { i in let lp = logProbs(model, x[i ..< i + 8]); eval(lp); return lp })
    }
    let (teacher, _) = try await load(args[1])
    let t = windows(teacher)
    for name in args[1...] {
        let (model, _) = try await load(name)
        let s = name == args[1] ? t : windows(model)
        let nll = -takeAlong(s[0..., ..<(-1)], x[0..., 1...].expandedDimensions(axis: -1), axis: -1).mean()
        print(String(format: "%-40@ ppl %8.2f  kl_vs_teacher %.4f", name, exp(nll).item(Float.self), kl(t, s).item(Float.self)))
    }
}

switch CommandLine.arguments.dropFirst().first {
case "train": try await train(positional())
case "eval": try await evaluate(positional())
default: print("usage: distill train TEACHER STUDENT OUT [--steps N --batch N --seq N --lr X] | distill eval TEXTFILE TEACHER MODEL...")
}

// Positional arguments after the mode, skipping "--name value" pairs.
func positional() -> [String] {
    var out = [String](), rest = CommandLine.arguments.dropFirst(2)[...]
    while let a = rest.popFirst() { if a.hasPrefix("--") { _ = rest.popFirst() } else { out.append(a) } }
    return out
}
