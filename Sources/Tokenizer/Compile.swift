// tokenizer.json -> compact binary file (see H in Tokenizer.swift for the header layout).
//
// Sections (all 64-byte aligned):
//   tokOff     UInt32[nTotal+1]   byte offsets of each token's raw bytes in tokBytes
//   tokBytes   UInt8[]            raw token bytes (byte-level alphabet decoded; added tokens' content)
//   byteTok    Int32[256]         token id of each single byte
//   byteProp   UInt16[256]        Unicode property of ASCII bytes (7 = non-ASCII lead/continuation)
//   merge      UInt64[2^mergeBits]  open addressing: (left<<18|right)<<18 | rank, empty = ~0
//   tokTab     UInt64[2^tokenBits]  bytes -> id for every token t with BPE(bytes(t)) == [t]
//   specials   UInt32[3*n] (id, offset, len) + specBytes  added tokens, matched atomically
//   uni1/uni2  two-stage Unicode property table (class, NFC flag, contraction bits, ccc)
//   decomp*    sorted canonical decompositions (crate data), comp*  sorted primary compositions
import Foundation

extension Tokenizer {
    static let qwen2SplitRegex =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// Compile a HuggingFace tokenizer.json (Qwen2 pipeline) into the binary format mmapped by `init(path:)`.
    public static func compile(tokenizerJSON: String, output: String) throws {
        guard let data = FileManager.default.contents(atPath: tokenizerJSON) else {
            throw TokenizerError.io("cannot read \(tokenizerJSON)")
        }
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw TokenizerError.format("\(tokenizerJSON) is not a JSON object")
        }
        try checkPipeline(root)
        let model = root["model"] as! [String: Any]
        guard let vocab = model["vocab"] as? [String: Any], let merges = model["merges"] as? [Any] else {
            throw TokenizerError.format("model.vocab / model.merges missing")
        }
        let u2b = byteLevelDecodeMap()

        // ---- vocab and added tokens
        var idOf = [String: Int](minimumCapacity: vocab.count)
        var bytesOf = [Int: [UInt8]](minimumCapacity: vocab.count)
        var maxId = -1
        for (k, v) in vocab {
            guard let id = (v as? NSNumber)?.intValue, id >= 0 else { throw TokenizerError.format("bad vocab id for \(k)") }
            var b = [UInt8]()
            for u in k.unicodeScalars {
                guard let x = u2b[u.value] else { throw TokenizerError.unsupported("vocab token \(k) is not byte-level") }
                b.append(x)
            }
            idOf[k] = id
            bytesOf[id] = b
            maxId = max(maxId, id)
        }
        let nVocab = maxId + 1
        guard let added = root["added_tokens"] as? [[String: Any]] else { throw TokenizerError.format("added_tokens missing") }
        var specials = [(id: Int, match: [UInt8])]()
        var specialId = [String: Int]()
        for t in added {
            guard let id = (t["id"] as? NSNumber)?.intValue, let content = t["content"] as? String else {
                throw TokenizerError.format("bad added token")
            }
            for f in ["single_word", "lstrip", "rstrip", "normalized"] where (t[f] as? Bool) == true {
                throw TokenizerError.unsupported("added token \(content) has \(f)=true")
            }
            // HF ByteLevel decoder: map chars through the byte-level alphabet if all are in it, else raw UTF-8.
            var dec = [UInt8]()
            var allMapped = true
            for u in content.unicodeScalars {
                if let x = u2b[u.value] { dec.append(x) } else { allMapped = false; break }
            }
            if bytesOf[id] == nil { bytesOf[id] = allMapped ? dec : Array(content.utf8) }
            specials.append((id, Array(content.utf8)))
            specialId[content] = id
            maxId = max(maxId, id)
        }
        let nTotal = maxId + 1
        guard nTotal < (1 << 18) else { throw TokenizerError.unsupported("more than 2^18 tokens") }
        guard let imStart = specialId["<|im_start|>"], let imEnd = specialId["<|im_end|>"],
              let endOfText = specialId["<|endoftext|>"] else {
            throw TokenizerError.unsupported("missing <|im_start|> / <|im_end|> / <|endoftext|>")
        }
        var byteTok = [Int32](repeating: -1, count: 256)
        for (k, id) in idOf where k.unicodeScalars.count == 1 {
            let b = u2b[k.unicodeScalars.first!.value]!
            byteTok[Int(b)] = Int32(id)
        }
        guard !byteTok.contains(-1) else { throw TokenizerError.unsupported("vocab lacks some single-byte tokens") }

        // ---- merges: rank r merges (a, b) into the token with id r + 256
        var mergeList = [(UInt64, UInt32)]()
        mergeList.reserveCapacity(merges.count)
        for (rank, m) in merges.enumerated() {
            let a: String, b: String
            if let s = m as? String {
                let parts = s.split(separator: " ", omittingEmptySubsequences: false)
                guard parts.count == 2 else { throw TokenizerError.format("bad merge \(s)") }
                a = String(parts[0]); b = String(parts[1])
            } else if let arr = m as? [String], arr.count == 2 {
                a = arr[0]; b = arr[1]
            } else {
                throw TokenizerError.format("bad merge entry \(rank)")
            }
            guard let ia = idOf[a], let ib = idOf[b], let inew = idOf[a + b] else {
                throw TokenizerError.format("merge \(rank) refers to unknown tokens")
            }
            guard inew == rank + 256 else {
                throw TokenizerError.unsupported("merge \(rank) (\(a) \(b)) produces id \(inew), expected rank+256")
            }
            mergeList.append((UInt64(ia) << 18 | UInt64(ib), UInt32(rank)))
        }

        // ---- Unicode tables
        let uni = try UnicodeTables.load()

        // ---- token bytes
        var tokOff = [UInt32](repeating: 0, count: nTotal + 1)
        var tokBytes = [UInt8]()
        var maxTokenLen = 0
        for id in 0..<nTotal {
            tokOff[id] = UInt32(tokBytes.count)
            if let b = bytesOf[id] {
                tokBytes += b
                if id < nVocab { maxTokenLen = max(maxTokenLen, b.count) }
            }
        }
        tokOff[nTotal] = UInt32(tokBytes.count)
        maxTokenLen = min(maxTokenLen, 255)

        // ---- specials table
        var specTab = [UInt32]()
        var specBytes = [UInt8]()
        for s in specials {
            specTab += [UInt32(s.id), UInt32(specBytes.count), UInt32(s.match.count)]
            specBytes += s.match
        }
        let firsts = Set(specials.compactMap { $0.match.first })
        let specFirstByte: UInt32 = firsts.count == 1 ? UInt32(firsts.first!) : 0xFFFF_FFFF

        // ---- layout
        func bitsFor(_ n: Int) -> Int {
            var b = 4
            while Double(n) / Double(1 << b) > 0.6 { b += 1 }
            return b
        }
        let mergeBits = bitsFor(mergeList.count)
        let tokenBits = bitsFor(nVocab)
        var hdr = [UInt32](repeating: 0, count: H.words)
        var size = H.size
        func section(_ field: Int, _ bytes: Int) -> Int {
            size = (size + 63) & ~63
            let off = size
            hdr[field] = UInt32(off)
            size += bytes
            return off
        }
        let oTokOff = section(H.oTokOff, tokOff.count * 4)
        let oTokBytes = section(H.oTokBytes, tokBytes.count)
        let oByteTok = section(H.oByteTok, 256 * 4)
        let oByteProp = section(H.oByteProp, 256 * 2)
        let oMerge = section(H.oMerge, (1 << mergeBits) * 8)
        let oTokTab = section(H.oTokTab, (1 << tokenBits) * 8)
        let oSpec = section(H.oSpecials, max(specTab.count, 1) * 4)
        let oSpecBytes = section(H.oSpecBytes, max(specBytes.count, 1))
        let oUni1 = section(H.oUni1, uni.stage1.count * 2)
        let oUni2 = section(H.oUni2, uni.stage2.count * 2)
        let oDK = section(H.oDecompKeys, max(uni.decompKeys.count, 1) * 4)
        let oDV = section(H.oDecompVals, max(uni.decompVals.count, 1) * 4)
        let oDD = section(H.oDecompData, max(uni.decompData.count, 1) * 4)
        let oCK = section(H.oCompKeys, max(uni.compKeys.count, 1) * 8)
        let oCV = section(H.oCompVals, max(uni.compVals.count, 1) * 4)
        size = (size + 63) & ~63

        hdr[H.fMagic] = H.magic
        hdr[H.fVersion] = H.version
        hdr[H.fFileSize] = UInt32(size)
        hdr[H.fNVocab] = UInt32(nVocab)
        hdr[H.fNTotal] = UInt32(nTotal)
        hdr[H.fNMerges] = UInt32(mergeList.count)
        hdr[H.fMaxTokenLen] = UInt32(maxTokenLen)
        hdr[H.fImStart] = UInt32(imStart)
        hdr[H.fImEnd] = UInt32(imEnd)
        hdr[H.fEndOfText] = UInt32(endOfText)
        hdr[H.fMergeBits] = UInt32(mergeBits)
        hdr[H.fTokenBits] = UInt32(tokenBits)
        hdr[H.fNSpecials] = UInt32(specials.count)
        hdr[H.fNUniBlocks] = UInt32(uni.stage2.count / 256)
        hdr[H.fNDecomp] = UInt32(uni.decompKeys.count)
        hdr[H.fNDecompData] = UInt32(uni.decompData.count)
        hdr[H.fNComp] = UInt32(uni.compKeys.count)
        hdr[H.fSpecFirstByte] = specFirstByte

        let buf = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 64)
        defer { buf.deallocate() }
        buf.initializeMemory(as: UInt8.self, repeating: 0, count: size)
        func put<T>(_ a: [T], _ off: Int) {
            a.withUnsafeBytes { buf.advanced(by: off).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        put(hdr, 0)
        put(tokOff, oTokOff)
        put(tokBytes, oTokBytes)
        put(byteTok, oByteTok)
        put(uni.byteProp, oByteProp)
        put(specTab, oSpec)
        put(specBytes, oSpecBytes)
        put(uni.stage1, oUni1)
        put(uni.stage2, oUni2)
        put(uni.decompKeys, oDK)
        put(uni.decompVals, oDV)
        put(uni.decompData, oDD)
        put(uni.compKeys, oCK)
        put(uni.compVals, oCV)

        // merge table
        let mergeTab = buf.advanced(by: oMerge).bindMemory(to: UInt64.self, capacity: 1 << mergeBits)
        let mergeMask = (1 << mergeBits) - 1
        for i in 0..<(1 << mergeBits) { mergeTab[i] = .max }
        for (key, rank) in mergeList {
            var slot = Int(truncatingIfNeeded: (key &* mergeHashMul) >> UInt64(64 - mergeBits))
            while mergeTab[slot] != .max {
                guard mergeTab[slot] >> 18 != key else { throw TokenizerError.format("duplicate merge") }
                slot = (slot + 1) & mergeMask
            }
            mergeTab[slot] = key << 18 | UInt64(rank)
        }

        // core over the buffer (token table still empty) to verify which tokens BPE reproduces
        let raw = UnsafeRawPointer(buf)
        func p<T>(_ off: Int, _: T.Type) -> UnsafePointer<T> { (raw + off).assumingMemoryBound(to: T.self) }
        let core = Core(
            tokOff: p(oTokOff, UInt32.self), tokBytes: p(oTokBytes, UInt8.self), byteTok: p(oByteTok, Int32.self),
            byteProp: p(oByteProp, UInt16.self), mergeTab: p(oMerge, UInt64.self),
            mergeShift: UInt64(64 - mergeBits), mergeMask: mergeMask, tokTab: p(oTokTab, UInt64.self),
            tokShift: UInt64(64 - tokenBits), tokMask: (1 << tokenBits) - 1, maxTokenLen: maxTokenLen,
            uni1: p(oUni1, UInt16.self), uni2: p(oUni2, UInt16.self), decompKeys: p(oDK, UInt32.self),
            decompVals: p(oDV, UInt32.self), decompData: p(oDD, UInt32.self), nDecomp: uni.decompKeys.count,
            compKeys: p(oCK, UInt64.self), compVals: p(oCV, UInt32.self), nComp: uni.compKeys.count)
        let tokTab = buf.advanced(by: oTokTab).bindMemory(to: UInt64.self, capacity: 1 << tokenBits)
        let tokMask = (1 << tokenBits) - 1
        var out = [Int32](repeating: 0, count: 256)
        var unreachable = 0
        let tb = p(oTokBytes, UInt8.self)
        for id in 0..<nVocab {
            let off = Int(tokOff[id]), len = Int(tokOff[id + 1]) - off
            if len < 2 || len > maxTokenLen || off >= (1 << 20) { continue }
            var cnt = 0
            out.withUnsafeMutableBufferPointer { core.bpe(tb + off, len, $0.baseAddress!, &cnt) }
            if cnt != 1 || out[0] != Int32(id) {
                unreachable += 1
                continue
            }
            let h = hashBytes(tb + off, len)
            var slot = Int(truncatingIfNeeded: h >> UInt64(64 - tokenBits))
            while tokTab[slot] != 0 { slot = (slot + 1) & tokMask }
            tokTab[slot] = (h & 0x3FFFF) << slotFpShift | UInt64(off) << slotOffShift | UInt64(len) << slotLenShift
                | UInt64(id)
        }
        _ = unreachable

        let d = Data(bytes: buf, count: size)
        do { try d.write(to: URL(fileURLWithPath: output), options: .atomic) } catch {
            throw TokenizerError.io("cannot write \(output): \(error)")
        }
    }

    static func checkPipeline(_ root: [String: Any]) throws {
        func no(_ s: String) -> TokenizerError { .unsupported(s) }
        guard let norm = root["normalizer"] as? [String: Any], norm["type"] as? String == "NFC" else {
            throw no("normalizer must be NFC")
        }
        guard let pre = root["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence",
              let list = pre["pretokenizers"] as? [[String: Any]], list.count == 2 else {
            throw no("pre_tokenizer must be Sequence[Split, ByteLevel]")
        }
        let split = list[0], bl = list[1]
        guard split["type"] as? String == "Split",
              (split["pattern"] as? [String: Any])?["Regex"] as? String == qwen2SplitRegex,
              split["behavior"] as? String == "Isolated", (split["invert"] as? Bool ?? false) == false else {
            throw no("Split must be the Qwen2 regex with behavior Isolated")
        }
        guard bl["type"] as? String == "ByteLevel", (bl["add_prefix_space"] as? Bool ?? false) == false,
              (bl["use_regex"] as? Bool ?? true) == false else {
            throw no("ByteLevel pre-tokenizer must have add_prefix_space=false, use_regex=false")
        }
        guard let model = root["model"] as? [String: Any], model["type"] as? String == "BPE" else {
            throw no("model must be BPE")
        }
        if let d = model["dropout"], !(d is NSNull) { throw no("BPE dropout") }
        if model["byte_fallback"] as? Bool == true { throw no("byte_fallback") }
        if let s = model["continuing_subword_prefix"] as? String, !s.isEmpty { throw no("continuing_subword_prefix") }
        if let s = model["end_of_word_suffix"] as? String, !s.isEmpty { throw no("end_of_word_suffix") }
        if model["ignore_merges"] as? Bool == true { throw no("ignore_merges") }
        if let dec = root["decoder"] as? [String: Any], dec["type"] as? String != "ByteLevel" {
            throw no("decoder must be ByteLevel")
        }
    }

    /// GPT-2 bytes_to_unicode, inverted: byte-level alphabet char -> byte.
    static func byteLevelDecodeMap() -> [UInt32: UInt8] {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var m = [UInt32: UInt8]()
        for (b, c) in zip(bs, cs) { m[UInt32(c)] = UInt8(b) }
        return m
    }
}

/// Unicode tables decoded from the generated UnicodeData blob (tools/tokenizer_unicode.py).
struct UnicodeTables {
    var byteProp = [UInt16](repeating: 0, count: 256)
    var stage1 = [UInt16](repeating: 0, count: 0x1100)
    var stage2 = [UInt16]()
    var decompKeys = [UInt32](), decompVals = [UInt32](), decompData = [UInt32]()
    var compKeys = [UInt64](), compVals = [UInt32]()

    static func load() throws -> UnicodeTables {
        guard let blob = Data(base64Encoded: UnicodeData.base64, options: .ignoreUnknownCharacters) else {
            throw TokenizerError.format("bad UnicodeData blob")
        }
        var t = UnicodeTables()
        try blob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var pos = 0
            func u8() -> UInt8 { defer { pos += 1 }; return raw[pos] }
            func u16() -> UInt16 { defer { pos += 2 }; return raw.loadUnaligned(fromByteOffset: pos, as: UInt16.self) }
            func u32() -> UInt32 { defer { pos += 4 }; return raw.loadUnaligned(fromByteOffset: pos, as: UInt32.self) }
            guard u32() == 0x3142_5455 else { throw TokenizerError.format("bad UnicodeData magic") }  // "UTB1"
            let nRuns = Int(u32())
            var runs = [(Int, UInt16)]()
            for _ in 0..<nRuns {
                let c = Int(u32())
                runs.append((c, u16()))
            }
            var prop = [UInt16](repeating: 2, count: 0x110000)
            for (k, r) in runs.enumerated() {
                let end = k + 1 < runs.count ? runs[k + 1].0 : 0x110000
                for c in r.0..<end { prop[c] = r.1 }
            }
            for b in 0..<256 { t.byteProp[b] = b < 0x80 ? prop[b] : cNonASCII }
            var index = [[UInt16]: Int]()
            for hi in 0..<0x1100 {
                let blk = Array(prop[(hi << 8)..<((hi + 1) << 8)])
                if let i = index[blk] {
                    t.stage1[hi] = UInt16(i)
                } else {
                    let i = t.stage2.count / 256
                    index[blk] = i
                    t.stage2 += blk
                    t.stage1[hi] = UInt16(i)
                }
            }
            let nDec = Int(u32())
            for _ in 0..<nDec {
                let c = u32()
                let len = Int(u8())
                t.decompKeys.append(c)
                t.decompVals.append(UInt32(t.decompData.count) << 8 | UInt32(len))
                for _ in 0..<len { t.decompData.append(u32()) }
            }
            let nComp = Int(u32())
            var comps = [(UInt64, UInt32)]()
            for _ in 0..<nComp {
                let a = u32(), b = u32(), r = u32()
                comps.append((UInt64(a) << 21 | UInt64(b), r))
            }
            comps.sort { $0.0 < $1.0 }
            t.compKeys = comps.map { $0.0 }
            t.compVals = comps.map { $0.1 }
        }
        return t
    }
}
