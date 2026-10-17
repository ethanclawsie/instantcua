import Foundation

// The .icua model file: a 16 KiB header (magic "ICU1", version, format, tensor table of
// (offset, size) pairs) followed by 256-byte aligned tensors, mmapped directly into one MTLBuffer.

public enum WeightFormat: UInt32, CustomStringConvertible {
    case f16 = 0
    case q8 = 1
    case q4 = 2

    public var description: String {
        switch self {
        case .f16: return "f16"
        case .q8: return "q8"
        case .q4: return "q4"
        }
    }

    public init?(name: String) {
        switch name {
        case "f16": self = .f16
        case "q8": self = .q8
        case "q4": self = .q4
        default: return nil
        }
    }
}

public enum Dims {
    // Vision Encoder (Qwen3VLVisionModel)
    public static let vitLayers = 24
    public static let vitHidden = 1024
    public static let vitFF = 4096
    public static let vitHeads = 16
    public static let vitHeadDim = 64
    public static let vitQkvRows = 3072
    public static let vitPatchSize = 16
    public static let vitMergeSize = 2
    public static let vitPatchDim = 768 // 3 * 16 * 16 (temporal_patch_size=2 folded into patch_embed)
    public static let vitPosGrid = 48   // 48 x 48 = 2304
    public static let vitPosCount = 2304
    public static let deepstackLayers = [5, 11, 17]
    public static let numDeepstack = 3
    public static let mergerHidden = 4096 // 1024 * 4
    public static let vitRopeTheta = 10_000.0

    // Language Backbone (Qwen3VLTextModel)
    public static let layers = 28
    public static let hidden = 2048
    public static let ff = 6144
    public static let heads = 16
    public static let kvHeads = 8
    public static let headDim = 128
    public static let qRows = 2048  // 16 * 128
    public static let kvRows = 1024 // 8 * 128
    public static let qkvRows = 4096 // 2048 + 1024 + 1024
    public static let guRows = 12288 // 6144 * 2
    public static let vocab = 151936
    public static let ropeTheta = 5_000_000.0
}

public enum TensorID {
    // Global Vision tensors (stored in f16/f32)
    public static let vitPatchW = 0      // [1024, 768] f16
    public static let vitPatchB = 1      // [1024] f32
    public static let vitPosEmb = 2      // [2304, 1024] f32

    // Per-ViT-layer tensors (24 layers * 12 slots = 288)
    public static let vitBase = 3
    public static let vitPerLayer = 12
    public static let vNorm1W = 0, vNorm1B = 1
    public static let vQkvW = 2, vQkvB = 3
    public static let vProjW = 4, vProjB = 5
    public static let vNorm2W = 6, vNorm2B = 7
    public static let vFc1W = 8, vFc1B = 9
    public static let vFc2W = 10, vFc2B = 11
    @inline(__always) public static func vitLayer(_ l: Int, _ k: Int) -> Int {
        vitBase + l * vitPerLayer + k
    }

    // Mergers: 3 DeepStack mergers (indices 0, 1, 2) + 1 final merger (index 3)
    // Each merger has 6 slots: normW, normB, fc1W, fc1B, fc2W, fc2B
    public static let mergerBase = vitBase + Dims.vitLayers * vitPerLayer
    public static let mergerPerUnit = 6
    public static let mNormW = 0, mNormB = 1, mFc1W = 2, mFc1B = 3, mFc2W = 4, mFc2B = 5
    @inline(__always) public static func merger(_ idx: Int, _ k: Int) -> Int {
        mergerBase + idx * mergerPerUnit + k
    }

    // Language Model global tensors
    public static let llmBase = mergerBase + (Dims.numDeepstack + 1) * mergerPerUnit
    public static let embed = llmBase + 0
    public static let embedSB = llmBase + 1
    public static let finalNorm = llmBase + 2

    // Per-LLM-layer tensors (28 layers * 14 slots)
    public static let llmLayerBase = llmBase + 3
    public static let llmPerLayer = 14
    public static let attnNorm = 0
    public static let wqkv = 1, wqkvSB = 2
    public static let qNorm = 3, kNorm = 4
    public static let wo = 5, woSB = 6
    public static let mlpNorm = 7
    public static let wgu = 8, wguSB = 9
    public static let wd = 10, wdSB = 11
    @inline(__always) public static func llmLayer(_ l: Int, _ k: Int) -> Int {
        llmLayerBase + l * llmPerLayer + k
    }

    public static let count = llmLayerBase + Dims.layers * llmPerLayer
}

let modelMagic: UInt32 = 0x3155_4349 // "ICU1"
let headerSize = 16384
let pageSize = 16384

public final class ModelFile {
    public let path: String
    public let base: UnsafeMutableRawPointer
    public let length: Int
    public let format: WeightFormat
    private let table: UnsafePointer<UInt64>

    public init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw EngineError("cannot open \(path)") }
        defer { close(fd) }
        var sb = stat()
        guard fstat(fd, &sb) == 0 else { throw EngineError("cannot stat \(path)") }
        let size = Int(sb.st_size)
        guard size >= headerSize, size % pageSize == 0 else { throw EngineError("\(path): bad size \(size)") }
        guard let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else {
            throw EngineError("mmap failed for \(path)")
        }
        base = p
        length = size
        let h = UnsafeRawPointer(p)
        guard h.load(as: UInt32.self) == modelMagic else { throw EngineError("\(path): not an instantcua model") }
        guard h.load(fromByteOffset: 4, as: UInt32.self) == 1 else { throw EngineError("\(path): unsupported version") }
        guard let f = WeightFormat(rawValue: h.load(fromByteOffset: 8, as: UInt32.self)) else {
            throw EngineError("\(path): unknown weight format")
        }
        format = f
        guard h.load(fromByteOffset: 12, as: UInt32.self) == UInt32(TensorID.count) else {
            throw EngineError("\(path): tensor table mismatch")
        }
        table = h.advanced(by: 16).assumingMemoryBound(to: UInt64.self)
    }

    deinit { munmap(base, length) }

    @inline(__always) public func offset(_ id: Int) -> Int { Int(table[2 * id]) }
    @inline(__always) public func size(_ id: Int) -> Int { Int(table[2 * id + 1]) }
    public func pointer(_ id: Int) -> UnsafeRawPointer { UnsafeRawPointer(base).advanced(by: offset(id)) }
}

public struct EngineError: Error, CustomStringConvertible {
    public let description: String
    public init(_ d: String) { description = d }
}

final class ModelWriter {
    private var blobs: [(Int, Data)] = []

    func add(_ id: Int, _ data: Data) { blobs.append((id, data)) }

    func write(to path: String, format: WeightFormat) throws {
        var table = [UInt64](repeating: 0, count: 2 * TensorID.count)
        var cursor = headerSize
        var layout: [(Int, Data)] = []
        for (id, d) in blobs.sorted(by: { $0.0 < $1.0 }) {
            cursor = (cursor + 255) & ~255
            table[2 * id] = UInt64(cursor)
            table[2 * id + 1] = UInt64(d.count)
            layout.append((cursor, d))
            cursor += d.count
        }
        let total = (cursor + pageSize - 1) / pageSize * pageSize
        var out = Data(count: total)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let p = raw.baseAddress!
            p.storeBytes(of: modelMagic, as: UInt32.self)
            p.storeBytes(of: UInt32(1), toByteOffset: 4, as: UInt32.self)
            p.storeBytes(of: format.rawValue, toByteOffset: 8, as: UInt32.self)
            p.storeBytes(of: UInt32(TensorID.count), toByteOffset: 12, as: UInt32.self)
            for (i, v) in table.enumerated() { p.storeBytes(of: v, toByteOffset: 16 + 8 * i, as: UInt64.self) }
            for (off, d) in layout {
                if !d.isEmpty {
                    d.withUnsafeBytes { src in p.advanced(by: off).copyMemory(from: src.baseAddress!, byteCount: d.count) }
                }
            }
        }
        try out.write(to: URL(fileURLWithPath: path))
    }
}
