// swift-tools-version: 6.2
import PackageDescription

let mlx: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXOptimizers", package: "mlx-swift"),
    .product(name: "MLXLLM", package: "mlx-swift-lm"),
    .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
    .product(name: "Tokenizers", package: "swift-transformers"),
]

let package = Package(
    name: "auto-distill",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "/Users/Shared/_Projects/agx/swift-ane"),  // local-only for now
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.33.0")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.33.0")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(name: "LMLoad", dependencies: mlx),
        .executableTarget(name: "distill", dependencies: mlx + ["LMLoad"]),
        .target(name: "LFM2ANE", dependencies: mlx + [.product(name: "ANE", package: "swift-ane")]),
        .executableTarget(name: "lfm2-ane", dependencies: mlx + ["LMLoad", "LFM2ANE", .product(name: "ANE", package: "swift-ane")]),
        .executableTarget(name: "ane-train", dependencies: mlx + ["LMLoad", "LFM2ANE", .product(name: "ANE", package: "swift-ane")]),
        .executableTarget(
            name: "ternary-probe",
            dependencies: [.product(name: "ANE", package: "swift-ane")],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
    ]
)
