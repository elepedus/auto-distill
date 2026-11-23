// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "auto-distill-ane",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "/Users/Shared/_Projects/agx/swift-ane")],
    targets: [
        .executableTarget(
            name: "ternary-probe",
            dependencies: [.product(name: "ANE", package: "swift-ane")],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
    ]
)
