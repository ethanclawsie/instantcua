import Foundation

// Converts the Hugging Face bf16 safetensors checkpoint of Qwen3-VL-2B-Instruct into an .icua file.
// Vision Encoder weights (ViT + DeepStack + Merger) are stored in f16 (norms/biases in f32) for
// zero-loss visual grounding and in-place Neural Accelerator GEMMs; Language Backbone weights are
// stored in f16, q8 (symmetric group-32), or q4 (affine group-64, MSE-optimal).

final class SafeTensors {
    struct Entry { let dtype: String; let shape: [Int]; let start: Int; let end: Int }
    let base: UnsafeMutableRawPointer
    let length: Int
    let dataStart: Int
    let entries: [String: Entry]

    init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw EngineError("cannot open \(path)") }
        defer { close(fd) }
        var sb = stat()
        fstat(fd, &sb)
        length = Int(sb.st_size)
        guard let p = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else {
            throw EngineError("mmap \(path)")
        }
        base = p
        let raw = UnsafeRawPointer(p)
        let n = Int(raw.loadUnaligned(as: UInt64.self))
        dataStart = 8 + n
        let json = try JSONSerialization.jsonObject(with: Data(bytes: raw.advanced(by: 8), count: n)) as! [String: Any]
        var e: [String: Entry] = [:]
        for (k, v) in json where k != "__metadata__" {
            let d = v as! [String: Any]
            let off = d["data_offsets"] as! [Int]
            e[k] = Entry(dtype: d["dtype"] as! String, shape: d["shape"] as! [Int], start: off[0], end: off[1])
        }
        entries = e
    }

    deinit {
        munmap(base, length)
    }

    func floats(_ name: String) throws -> (shape: [Int], data: [Float]) {
        guard let e = entries[name] else { throw EngineError("missing tensor \(name)") }
        let src = UnsafeRawPointer(base).advanced(by: dataStart + e.start)
        switch e.dtype {
        case "BF16":
            let count = (e.end - e.start) / 2
            var out = [Float](repeating: 0, count: count)
            out.withUnsafeMutableBufferPointer { dst in
                for i in 0..<count {
                    let b = src.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)
                    dst[i] = Float(bitPattern: UInt32(b) << 16)
                }
            }
            return (e.shape, out)
        case "F16":
            let count = (e.end - e.start) / 2
            let buf = UnsafeBufferPointer(start: src.assumingMemoryBound(to: Float16.self), count: count)
            return (e.shape, buf.map { Float($0) })
        case "F32":
            let count = (e.end - e.start) / 4
            let buf = UnsafeBufferPointer(start: src.assumingMemoryBound(to: Float.self), count: count)
            return (e.shape, Array(buf))
        default:
            throw EngineError("\(name): unsupported dtype \(e.dtype)")
        }
    }
}

struct Q4Mat {
    var packed: [UInt8]
    var sb: [Float16]
    let rows: Int
    let cols: Int

    static func + (a: Q4Mat, b: Q4Mat) -> Q4Mat {
        Q4Mat(packed: a.packed + b.packed, sb: a.sb + b.sb, rows: a.rows + b.rows, cols: a.cols)
    }
}

public enum Converter {
    public static func convert(hfDir: String, output: String, format: WeightFormat, log: (String) -> Void = { print($0) }) throws {
        let t0 = Date()
        let st = try SafeTensors(path: hfDir + "/model.safetensors")
        let w = ModelWriter()

        func f32(_ name: String) throws -> Data {
            let v = try st.floats(name).data
            return v.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        func f16(_ name: String) throws -> Data {
            let v = try st.floats(name).data
            return f16Data(v)
        }

        // 1. Global Vision tensors: fold temporal_patch_size=2 in patch_embed.proj.weight [1024, 3, 2, 16, 16] -> [1024, 768]
        do {
            let (_, pw) = try st.floats("model.visual.patch_embed.proj.weight")
            var folded = [Float](repeating: 0, count: Dims.vitHidden * Dims.vitPatchDim)
            let patchArea = 16 * 16 // 256
            for o in 0..<Dims.vitHidden {
                for c in 0..<3 {
                    let srcBase0 = ((o * 3 + c) * 2 + 0) * patchArea
                    let srcBase1 = ((o * 3 + c) * 2 + 1) * patchArea
                    let dstBase = (o * 3 + c) * patchArea
                    for p in 0..<patchArea {
                        folded[dstBase + p] = pw[srcBase0 + p] + pw[srcBase1 + p]
                    }
                }
            }
            w.add(TensorID.vitPatchW, f16Data(folded))
            w.add(TensorID.vitPatchB, try f32("model.visual.patch_embed.proj.bias"))
            w.add(TensorID.vitPosEmb, try f32("model.visual.pos_embed.weight"))
        }

        // 2. 24 Vision Transformer blocks
        for l in 0..<Dims.vitLayers {
            let p = "model.visual.blocks.\(l)."
            w.add(TensorID.vitLayer(l, TensorID.vNorm1W), try f32(p + "norm1.weight"))
            w.add(TensorID.vitLayer(l, TensorID.vNorm1B), try f32(p + "norm1.bias"))
            w.add(TensorID.vitLayer(l, TensorID.vQkvW), try f16(p + "attn.qkv.weight"))
            w.add(TensorID.vitLayer(l, TensorID.vQkvB), try f32(p + "attn.qkv.bias"))
            w.add(TensorID.vitLayer(l, TensorID.vProjW), try f16(p + "attn.proj.weight"))
            w.add(TensorID.vitLayer(l, TensorID.vProjB), try f32(p + "attn.proj.bias"))
            w.add(TensorID.vitLayer(l, TensorID.vNorm2W), try f32(p + "norm2.weight"))
            w.add(TensorID.vitLayer(l, TensorID.vNorm2B), try f32(p + "norm2.bias"))
            w.add(TensorID.vitLayer(l, TensorID.vFc1W), try f16(p + "mlp.linear_fc1.weight"))
            w.add(TensorID.vitLayer(l, TensorID.vFc1B), try f32(p + "mlp.linear_fc1.bias"))
            w.add(TensorID.vitLayer(l, TensorID.vFc2W), try f16(p + "mlp.linear_fc2.weight"))
            w.add(TensorID.vitLayer(l, TensorID.vFc2B), try f32(p + "mlp.linear_fc2.bias"))
        }

        // 3. 3 DeepStack mergers + 1 final merger
        for d in 0..<Dims.numDeepstack {
            let p = "model.visual.deepstack_merger_list.\(d)."
            w.add(TensorID.merger(d, TensorID.mNormW), try f32(p + "norm.weight"))
            w.add(TensorID.merger(d, TensorID.mNormB), try f32(p + "norm.bias"))
            w.add(TensorID.merger(d, TensorID.mFc1W), try f16(p + "linear_fc1.weight"))
            w.add(TensorID.merger(d, TensorID.mFc1B), try f32(p + "linear_fc1.bias"))
            w.add(TensorID.merger(d, TensorID.mFc2W), try f16(p + "linear_fc2.weight"))
            w.add(TensorID.merger(d, TensorID.mFc2B), try f32(p + "linear_fc2.bias"))
        }
        do {
            let p = "model.visual.merger."
            let idx = Dims.numDeepstack
            w.add(TensorID.merger(idx, TensorID.mNormW), try f32(p + "norm.weight"))
            w.add(TensorID.merger(idx, TensorID.mNormB), try f32(p + "norm.bias"))
            w.add(TensorID.merger(idx, TensorID.mFc1W), try f16(p + "linear_fc1.weight"))
            w.add(TensorID.merger(idx, TensorID.mFc1B), try f32(p + "linear_fc1.bias"))
            w.add(TensorID.merger(idx, TensorID.mFc2W), try f16(p + "linear_fc2.weight"))
            w.add(TensorID.merger(idx, TensorID.mFc2B), try f32(p + "linear_fc2.bias"))
        }
        log(String(format: "vision encoder & mergers done (%.1fs)", Date().timeIntervalSince(t0)))

        // 4. Language Backbone (embed_tokens + 28 decoder layers + final norm)
        func emitMat(_ id: Int, _ sbID: Int, _ floats: [Float], rows: Int, cols: Int) {
            switch format {
            case .f16:
                w.add(id, f16Data(floats))
                w.add(sbID, Data())
            case .q8:
                let (codes, scales) = quantizeQ8(floats, rows: rows, cols: cols)
                w.add(id, codes.withUnsafeBufferPointer { Data(buffer: $0) })
                w.add(sbID, scales.withUnsafeBufferPointer { Data(buffer: $0) })
            case .q4:
                let (packed, sb) = quantizeQ4Raw(floats, rows: rows, cols: cols)
                w.add(id, packed.withUnsafeBufferPointer { Data(buffer: $0) })
                w.add(sbID, sb.withUnsafeBufferPointer { Data(buffer: $0) })
            }
        }

        let H = Dims.hidden, F = Dims.ff, V = Dims.vocab
        emitMat(TensorID.embed, TensorID.embedSB, try st.floats("model.language_model.embed_tokens.weight").data, rows: V, cols: H)
        w.add(TensorID.finalNorm, try f32("model.language_model.norm.weight"))
        log(String(format: "language embeddings done (%.1fs)", Date().timeIntervalSince(t0)))

        for l in 0..<Dims.layers {
            let p = "model.language_model.layers.\(l)."
            w.add(TensorID.llmLayer(l, TensorID.attnNorm), try f32(p + "input_layernorm.weight"))
            w.add(TensorID.llmLayer(l, TensorID.qNorm), try f32(p + "self_attn.q_norm.weight"))
            w.add(TensorID.llmLayer(l, TensorID.kNorm), try f32(p + "self_attn.k_norm.weight"))
            w.add(TensorID.llmLayer(l, TensorID.mlpNorm), try f32(p + "post_attention_layernorm.weight"))

            let qkv = try st.floats(p + "self_attn.q_proj.weight").data
                + st.floats(p + "self_attn.k_proj.weight").data
                + st.floats(p + "self_attn.v_proj.weight").data
            emitMat(TensorID.llmLayer(l, TensorID.wqkv), TensorID.llmLayer(l, TensorID.wqkvSB), qkv, rows: Dims.qkvRows, cols: H)

            let wo = try st.floats(p + "self_attn.o_proj.weight").data
            emitMat(TensorID.llmLayer(l, TensorID.wo), TensorID.llmLayer(l, TensorID.woSB), wo, rows: H, cols: H)

            let gu = try st.floats(p + "mlp.gate_proj.weight").data + st.floats(p + "mlp.up_proj.weight").data
            emitMat(TensorID.llmLayer(l, TensorID.wgu), TensorID.llmLayer(l, TensorID.wguSB), gu, rows: Dims.guRows, cols: H)

            let wd = try st.floats(p + "mlp.down_proj.weight").data
            emitMat(TensorID.llmLayer(l, TensorID.wd), TensorID.llmLayer(l, TensorID.wdSB), wd, rows: H, cols: F)

            if (l + 1) % 7 == 0 || l == Dims.layers - 1 {
                log(String(format: "llm layer %d/%d done (%.1fs)", l + 1, Dims.layers, Date().timeIntervalSince(t0)))
            }
        }

        try w.write(to: output, format: format)
        log(String(format: "wrote %@ (%.1fs)", output, Date().timeIntervalSince(t0)))
    }

    static func f16Data(_ v: [Float]) -> Data {
        var h = [Float16](repeating: 0, count: v.count)
        for i in 0..<v.count { h[i] = Float16(v[i]) }
        return h.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// Symmetric 8-bit, group 32 along K, f16 scale: w ~ s * q, q in -127...127.
    static func quantizeQ8(_ m: [Float], rows: Int, cols: Int) -> ([Int8], [Float16]) {
        let groups = cols / 32
        var codes = [Int8](repeating: 0, count: rows * cols)
        var scales = [Float16](repeating: 0, count: rows * groups)
        m.withUnsafeBufferPointer { src in
            codes.withUnsafeMutableBufferPointer { cp in
                scales.withUnsafeMutableBufferPointer { sp in
                    DispatchQueue.concurrentPerform(iterations: (rows + 63) / 64) { ci in
                        for r in (ci * 64)..<min(rows, (ci + 1) * 64) {
                            for g in 0..<groups {
                                let base = r * cols + g * 32
                                var amax: Float = 0
                                for i in 0..<32 { amax = max(amax, abs(src[base + i])) }
                                var sh = Float16(amax / 127)
                                if Float(sh) * 127 < amax { sh = Float16(bitPattern: sh.bitPattern &+ 1) }
                                let s = Float(sh)
                                sp[r * groups + g] = sh
                                let inv: Float = s > 0 ? 1 / s : 0
                                for i in 0..<32 { cp[base + i] = Int8(max(-127, min(127, (src[base + i] * inv).rounded()))) }
                            }
                        }
                    }
                }
            }
        }
        return (codes, scales)
    }

    static let clips: [Float] = [0, 0.03, 0.06, 0.1, 0.15]

    /// Affine 4-bit, group 64, f16 (scale, bias) per group: w ~ scale * q + bias, q in 0...15.
    static func quantizeQ4Raw(_ m: [Float], rows: Int, cols: Int) -> ([UInt8], [Float16]) {
        let groups = cols / 64
        var packed = [UInt8](repeating: 0, count: rows * cols / 2)
        var sb = [Float16](repeating: 0, count: rows * groups * 2)
        m.withUnsafeBufferPointer { src in
            packed.withUnsafeMutableBufferPointer { pk in
                sb.withUnsafeMutableBufferPointer { sbp in
                    let chunk = 64
                    DispatchQueue.concurrentPerform(iterations: (rows + chunk - 1) / chunk) { ci in
                        var q = [UInt8](repeating: 0, count: 64)
                        for r in (ci * chunk)..<min(rows, (ci + 1) * chunk) {
                            for g in 0..<groups {
                                let base = r * cols + g * 64
                                let (s, b) = quantGroup(src.baseAddress! + base, &q)
                                sbp[(r * groups + g) * 2] = s
                                sbp[(r * groups + g) * 2 + 1] = b
                                for j in 0..<32 { pk[base / 2 + j] = q[2 * j] | (q[2 * j + 1] << 4) }
                            }
                        }
                    }
                }
            }
        }
        return (packed, sb)
    }

    @inline(__always)
    static func quantGroup(_ w: UnsafePointer<Float>, _ qOut: inout [UInt8]) -> (Float16, Float16) {
        var mn = w[0], mx = w[0]
        for i in 1..<64 { mn = min(mn, w[i]); mx = max(mx, w[i]) }
        if mx - mn < 1e-12 {
            for i in 0..<64 { qOut[i] = 0 }
            return (0, Float16(mn))
        }
        var bestErr = Float.infinity
        var bestS: Float16 = 0, bestB: Float16 = 0
        var q = [UInt8](repeating: 0, count: 64)
        let range = mx - mn
        for a in clips {
            for c in clips {
                let lo = mn + a * range, hi = mx - c * range
                if hi <= lo { continue }
                var s = (hi - lo) / 15, b = lo
                for iter in 0..<2 {
                    let inv = 1 / s
                    var sq: Float = 0, sw: Float = 0, sqq: Float = 0, sqw: Float = 0
                    for i in 0..<64 {
                        let v = min(15, max(0, ((w[i] - b) * inv).rounded()))
                        q[i] = UInt8(v)
                        sq += v; sw += w[i]; sqq += v * v; sqw += v * w[i]
                    }
                    if iter == 1 { break }
                    let den = 64 * sqq - sq * sq
                    if den <= 0 { break }
                    let s2 = (64 * sqw - sq * sw) / den
                    if s2 <= 0 { break }
                    s = s2
                    b = (sw - s2 * sq) / 64
                }
                let sh = Float16(s), bh = Float16(b)
                let sf = Float(sh), bf = Float(bh)
                if sf <= 0 { continue }
                let inv = 1 / sf
                var err: Float = 0
                for i in 0..<64 {
                    let v = min(15, max(0, ((w[i] - bf) * inv).rounded()))
                    q[i] = UInt8(v)
                    let d = sf * v + bf - w[i]
                    err += d * d
                }
                if err < bestErr { bestErr = err; bestS = sh; bestB = bh; qOut = q }
            }
        }
        return (bestS, bestB)
    }
}
