import Actuator
import CShim
import CoreGraphics
import Engine
import Foundation
import Metal
import Tokenizer

func writeBin(path: String, floats: UnsafePointer<Float>, count: Int) throws {
    let data = Data(buffer: UnsafeBufferPointer(start: floats, count: count))
    try data.write(to: URL(fileURLWithPath: path))
}

func loadSystemPrompt() -> String {
    if let s = try? String(contentsOfFile: "bench/system.txt", encoding: .utf8) {
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    return """
    You are a macOS computer-use agent. Inspect the screenshot and output ONLY a single action call with normalized integer coordinates in [0, 1000] x [0, 1000] (top-left is (0, 0), bottom-right is (1000, 1000)):
    click(x, y)
    double_click(x, y)
    right_click(x, y)
    move(x, y)
    drag(x1, y1, x2, y2)
    type("text")
    key("combo")
    scroll(dx, dy)
    wait(ms)
    done()
    Do not output any explanation or markdown—only the single function call.
    """
}

struct BenchTask: Codable {
    let id: String
    let image: String
    let prompt: String
    let expectedAction: String
    let targetBox1000: [Int]
}

@main
struct InstantCUACLI {
    static func main() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let cmd = args.first else {
            printUsage()
            return
        }

        switch cmd {
        case "compile-tokenizer":
            guard args.count >= 3 else {
                throw EngineError("usage: instantcua compile-tokenizer <tokenizer.json> <output.tkn>")
            }
            try Tokenizer.compile(tokenizerJSON: args[1], output: args[2])
            print("Compiled tokenizer -> \(args[2])")

        case "convert":
            guard args.count >= 3 else {
                throw EngineError("usage: instantcua convert <hf_dir> <output.icua> [--format f16|q8|q4]")
            }
            let hfDir = args[1]
            let outPath = args[2]
            var fmt: WeightFormat = .f16
            if let idx = args.firstIndex(of: "--format"), idx + 1 < args.count,
               let f = WeightFormat(name: args[idx + 1]) {
                fmt = f
            }
            try Converter.convert(hfDir: hfDir, output: outPath, format: fmt)

        case "golden":
            guard args.count >= 4 else {
                throw EngineError("usage: instantcua golden <model.icua> <tokenizer.tkn> <out_dir>")
            }
            try runGolden(modelPath: args[1], tokPath: args[2], outDir: args[3])

        case "bench":
            guard args.count >= 3 else {
                throw EngineError("usage: instantcua bench <model.icua> <tokenizer.tkn>")
            }
            try runBench(modelPath: args[1], tokPath: args[2])

        case "run":
            guard args.count >= 4 else {
                throw EngineError("usage: instantcua run <model.icua> <tokenizer.tkn> <instruction> [--image <path>] [--dry-run]")
            }
            let modelPath = args[1]
            let tokPath = args[2]
            let instruction = args[3]
            var imagePath: String? = nil
            var dryRun = false
            var i = 4
            while i < args.count {
                if args[i] == "--image", i + 1 < args.count {
                    imagePath = args[i + 1]
                    i += 2
                } else if args[i] == "--dry-run" {
                    dryRun = true
                    i += 1
                } else {
                    i += 1
                }
            }
            try runSingle(modelPath: modelPath, tokPath: tokPath, instruction: instruction, imagePath: imagePath, dryRun: dryRun)

        default:
            printUsage()
        }
    }

    static func printUsage() {
        print("""
        instantcua — Native Metal 4 macOS Computer-Use Agent (Qwen3-VL-2B-Instruct)
        Commands:
          compile-tokenizer <tokenizer.json> <output.tkn>
          convert <hf_dir> <output.icua> [--format f16|q8|q4]
          golden <model.icua> <tokenizer.tkn> <out_dir>
          bench <model.icua> <tokenizer.tkn>
          run <model.icua> <tokenizer.tkn> <instruction> [--image <path>] [--dry-run]
        """)
    }

    static func runGolden(modelPath: String, tokPath: String, outDir: String) throws {
        try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        let engine = try Engine(modelPath: modelPath)
        let tok = try Tokenizer(path: tokPath)
        let sysPrompt = loadSystemPrompt()
        let tasksData = try Data(contentsOf: URL(fileURLWithPath: "bench/tasks.json"))
        let tasks = try JSONDecoder().decode([BenchTask].self, from: tasksData)

        var summary: [[String: Any]] = []
        var hits = 0

        for (idx, task) in tasks.enumerated() {
            let rgb = try ScreenCapture.loadRGB(path: task.image, targetW: 640, targetH: 384)
            let promptTokens = tok.encodeVisionChat(system: sysPrompt, user: task.prompt, numImageTokens: 240).tokens
            let outTokens = rgb.withUnsafeBytes { raw in
                engine.generateVision(rgbBytes: raw.baseAddress!, width: 640, height: 384,
                                      tokens: promptTokens, maxNewTokens: 32, writeFirstLogits: idx == 0)
            }
            let text = tok.decode(outTokens)
            let t = engine.lastTiming

            if idx == 0 {
                let poolerPtr = engine.poolerBuf.contents().assumingMemoryBound(to: Float.self)
                try writeBin(path: "\(outDir)/task0_pooler.bin", floats: poolerPtr, count: 240 * 2048)
                for d in 0..<3 {
                    let dsPtr = engine.deepstackBufs[d].contents().assumingMemoryBound(to: Float.self)
                    try writeBin(path: "\(outDir)/task0_deepstack_\(d).bin", floats: dsPtr, count: 240 * 2048)
                }
                let logPtr = engine.logits.contents().assumingMemoryBound(to: Float.self)
                try writeBin(path: "\(outDir)/task0_first_logits.bin", floats: logPtr, count: Dims.vocab)
            }

            var hit = false
            var px = -1, py = -1
            if let action = ActionParser.parse(text), let c = action.coords {
                px = c.x; py = c.y
                let b = task.targetBox1000
                if px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3] {
                    hit = true
                    hits += 1
                }
            }

            let trimmed = text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            print(String(format: "[%02d/%02d] %-24s -> %-22s box=[%d,%d,%d,%d] hit=%s (vis=%.1fms pre=%.1fms dec=%.1fms total=%.1fms)",
                         idx + 1, tasks.count,
                         (task.id as NSString).utf8String!,
                         (trimmed as NSString).utf8String!,
                         task.targetBox1000[0], task.targetBox1000[1], task.targetBox1000[2], task.targetBox1000[3],
                         (hit ? "YES" : "NO" as NSString).utf8String!,
                         t.visionMs, t.prefillMs, t.decodeMs, t.totalMs))

            summary.append([
                "id": task.id,
                "prompt": task.prompt,
                "text": text,
                "tokens": outTokens.map { Int($0) },
                "pred_xy": [px, py],
                "targetBox1000": task.targetBox1000,
                "hit": hit,
                "vision_ms": t.visionMs,
                "prefill_ms": t.prefillMs,
                "decode_ms": t.decodeMs,
                "total_ms": t.totalMs,
            ])
        }

        let jsonData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: "\(outDir)/summary.json"))
        print("Summary: \(hits)/\(tasks.count) GUI benchmark tasks hit target bounding box.")
    }

    static func runBench(modelPath: String, tokPath: String) throws {
        let engine = try Engine(modelPath: modelPath)
        let tok = try Tokenizer(path: tokPath)
        let sysPrompt = loadSystemPrompt()
        let tasksData = try Data(contentsOf: URL(fileURLWithPath: "bench/tasks.json"))
        let tasks = try JSONDecoder().decode([BenchTask].self, from: tasksData)

        // Warmup pass
        do {
            let rgb = try ScreenCapture.loadRGB(path: tasks[0].image, targetW: 640, targetH: 384)
            let p = tok.encodeVisionChat(system: sysPrompt, user: tasks[0].prompt, numImageTokens: 240).tokens
            _ = rgb.withUnsafeBytes { raw in
                engine.generateVision(rgbBytes: raw.baseAddress!, width: 640, height: 384, tokens: p, maxNewTokens: 16)
            }
        }

        var visTimes: [Double] = []
        var preTimes: [Double] = []
        var decTimes: [Double] = []
        var totTimes: [Double] = []
        var decToks = 0
        var hits = 0

        for task in tasks {
            let rgb = try ScreenCapture.loadRGB(path: task.image, targetW: 640, targetH: 384)
            let promptTokens = tok.encodeVisionChat(system: sysPrompt, user: task.prompt, numImageTokens: 240).tokens
            let outTokens = rgb.withUnsafeBytes { raw in
                engine.generateVision(rgbBytes: raw.baseAddress!, width: 640, height: 384, tokens: promptTokens, maxNewTokens: 32)
            }
            let text = tok.decode(outTokens)
            let t = engine.lastTiming
            visTimes.append(t.visionMs)
            preTimes.append(t.prefillMs)
            decTimes.append(t.decodeMs)
            totTimes.append(t.totalMs)
            decToks += max(0, t.generatedTokens - 1)

            if let action = ActionParser.parse(text), let c = action.coords {
                let b = task.targetBox1000
                if c.x >= b[0] && c.x <= b[2] && c.y >= b[1] && c.y <= b[3] {
                    hits += 1
                }
            }
        }

        func median(_ a: [Double]) -> Double {
            let s = a.sorted()
            return s[s.count / 2]
        }
        let sumDecMs = decTimes.reduce(0, +)
        let tokPerSec = sumDecMs > 0 ? (Double(decToks) / (sumDecMs / 1000.0)) : 0

        print(String(format: "Model: %s (%s) | Accuracy: %d/%d | Median Vision: %.2f ms | Median Prefill: %.2f ms | Decode: %.1f tok/s | Median Total Action: %.2f ms",
                     ((modelPath as NSString).lastPathComponent as NSString).utf8String!,
                     (engine.format.description as NSString).utf8String!,
                     hits, tasks.count,
                     median(visTimes), median(preTimes), tokPerSec, median(totTimes)))
    }

    static func runSingle(modelPath: String, tokPath: String, instruction: String, imagePath: String?, dryRun: Bool) throws {
        let engine = try Engine(modelPath: modelPath)
        let tok = try Tokenizer(path: tokPath)
        let sysPrompt = loadSystemPrompt()

        let rgb: [UInt8]
        let screenBounds: CGRect
        if let path = imagePath {
            rgb = try ScreenCapture.loadRGB(path: path, targetW: 640, targetH: 384)
            screenBounds = CGDisplayBounds(CGMainDisplayID())
        } else {
            let (captured, bounds) = try ScreenCapture.captureDisplayRGB(targetW: 640, targetH: 384)
            rgb = captured
            screenBounds = bounds
        }

        let promptTokens = tok.encodeVisionChat(system: sysPrompt, user: instruction, numImageTokens: 240).tokens
        let outTokens = rgb.withUnsafeBytes { raw in
            engine.generateVision(rgbBytes: raw.baseAddress!, width: 640, height: 384, tokens: promptTokens, maxNewTokens: 32)
        }
        let text = tok.decode(outTokens).trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        let t = engine.lastTiming
        print(String(format: "%s  [vis=%.1fms pre=%.1fms dec=%.1fms total=%.1fms]",
                     (text as NSString).utf8String!, t.visionMs, t.prefillMs, t.decodeMs, t.totalMs))

        if let action = ActionParser.parse(text) {
            Actuator.perform(action, in: screenBounds, dryRun: dryRun || imagePath != nil)
        }
    }
}
