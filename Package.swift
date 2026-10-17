// swift-tools-version:6.2
import PackageDescription

let release: [SwiftSetting] = [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]

let package = Package(
    name: "instantcua",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "instantcua", targets: ["instantcua"]),
        .executable(name: "icua", targets: ["icua"]),
    ],
    targets: [
        .target(name: "Tokenizer", swiftSettings: release),
        .target(name: "CShim"),
        .target(
            name: "Engine",
            dependencies: ["CShim", "Tokenizer"],
            swiftSettings: release,
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreGraphics"),
            ]
        ),
        .target(
            name: "Actuator",
            dependencies: ["CShim"],
            swiftSettings: release,
            linkerSettings: [
                .linkedFramework("ApplicationServices"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("AppKit"),
                .linkedFramework("ScreenCaptureKit"),
            ]
        ),
        .executableTarget(
            name: "instantcua",
            dependencies: ["Engine", "Tokenizer", "Actuator"],
            swiftSettings: release
        ),
        .executableTarget(
            name: "icua",
            cSettings: [.unsafeFlags(["-O2"])]
        ),
    ],
    swiftLanguageModes: [.v5]
)
