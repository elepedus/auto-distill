import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// A local model directory, or the cached snapshot of a Hugging Face id ("org/name").
public func modelDirectory(_ model: String) throws -> URL {
    if FileManager.default.fileExists(atPath: model) { return URL(filePath: model) }
    let snapshots = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: ".cache/huggingface/hub/models--\(model.replacing("/", with: "--"))/snapshots")
    guard let snapshot = try FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil).first
    else { throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: snapshots.path]) }
    return snapshot
}

/// Builds the model from its config and safetensors. No tokenizer needed.
public func loadLLM(_ model: String) async throws -> (any LLMModel, URL) {
    let dir = try modelDirectory(model)
    let config = try Data(contentsOf: dir.appending(path: "config.json"))
    let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: config)
    guard let lm = try await LLMTypeRegistry.shared.createModel(configuration: config, modelType: base.modelType) as? any LLMModel
    else { throw CocoaError(.featureUnsupported) }
    try await loadWeights(modelDirectory: dir, model: lm, perLayerQuantization: base.perLayerQuantization)
    return (lm, dir)
}
