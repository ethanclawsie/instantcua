// Qwen2.5 byte-level BPE tokenizer, bit-exact with HuggingFace `tokenizers` (Oniguruma split regex,
// unicode-normalization-alignments NFC). `Tokenizer.compile` turns tokenizer.json into a compact
// binary file; `init(path:)` mmaps it (no parsing, no dictionary building).
import Darwin

public enum TokenizerError: Error, CustomStringConvertible {
    case io(String)
    case format(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .io(let s): return "tokenizer I/O error: \(s)"
        case .format(let s): return "tokenizer format error: \(s)"
        case .unsupported(let s): return "unsupported tokenizer: \(s)"
        }
    }
}

// Compiled file header: 64 little-endian UInt32 words.
enum H {
    static let magic: UInt32 = 0x314B_5451  // "QTK1"
    static let version: UInt32 = 1
    static let words = 64
    static let size = words * 4
    // field indices
    static let fMagic = 0, fVersion = 1, fFileSize = 2, fNVocab = 3, fNTotal = 4, fNMerges = 5
    static let fMaxTokenLen = 6, fImStart = 7, fImEnd = 8, fEndOfText = 9, fMergeBits = 10
    static let fTokenBits = 11, fNSpecials = 12, fNUniBlocks = 13, fNDecomp = 14, fNDecompData = 15
    static let fNComp = 16, fSpecFirstByte = 17
    static let oTokOff = 20, oTokBytes = 21, oByteTok = 22, oByteProp = 23, oMerge = 24, oTokTab = 25
    static let oSpecials = 26, oSpecBytes = 27, oUni1 = 28, oUni2 = 29, oDecompKeys = 30
    static let oDecompVals = 31, oDecompData = 32, oCompKeys = 33, oCompVals = 34
}

public final class Tokenizer: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let mappedSize: Int
    let core: Core
    let nTotal: Int
    let specTab: UnsafePointer<UInt32>   // per special: id, offset into specBytes, length
    let specBytes: UnsafePointer<UInt8>
    let nSpecials: Int
    let specFirstByte: Int32             // shared first byte of all specials, or -1
    // Chat template with {system} and {user} cut out: bytes [0, cut0) {system} [cut0, cut1) {user} [cut1, len)
    let chatBytes: UnsafeMutablePointer<UInt8>
    let chatCut: (Int, Int)
    let chatLen: Int

    public let imStart: Int32
    public let imEnd: Int32
    public let endOfText: Int32
    public let visionStart: Int32 = 151652
    public let visionEnd: Int32 = 151653
    public let imagePad: Int32 = 151655

    /// Number of token ids (BPE vocab + added tokens).
    public var vocabSize: Int { nTotal }

    /// mmap a file produced by `Tokenizer.compile`.
    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw TokenizerError.io("cannot open \(path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw TokenizerError.io("cannot stat \(path)") }
        let size = Int(st.st_size)
        guard size >= H.size else { throw TokenizerError.format("\(path): file too small") }
        guard let m = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), m != MAP_FAILED else {
            throw TokenizerError.io("mmap failed for \(path)")
        }
        let hdr = UnsafeRawPointer(m).assumingMemoryBound(to: UInt32.self)
        guard hdr[H.fMagic] == H.magic, hdr[H.fVersion] == H.version, Int(hdr[H.fFileSize]) == size,
              Tokenizer.sectionsInBounds(hdr, size) else {
            munmap(m, size)
            throw TokenizerError.format("\(path): not a valid compiled tokenizer (or wrong version); recompile it")
        }
        base = m
        mappedSize = size
        let raw = UnsafeRawPointer(m)
        func ptr<T>(_ field: Int, _: T.Type) -> UnsafePointer<T> {
            (raw + Int(hdr[field])).assumingMemoryBound(to: T.self)
        }
        let mergeBits = UInt64(hdr[H.fMergeBits]), tokenBits = UInt64(hdr[H.fTokenBits])
        core = Core(
            tokOff: ptr(H.oTokOff, UInt32.self), tokBytes: ptr(H.oTokBytes, UInt8.self),
            byteTok: ptr(H.oByteTok, Int32.self), byteProp: ptr(H.oByteProp, UInt16.self),
            mergeTab: ptr(H.oMerge, UInt64.self), mergeShift: 64 - mergeBits, mergeMask: (1 << Int(mergeBits)) - 1,
            tokTab: ptr(H.oTokTab, UInt64.self), tokShift: 64 - tokenBits, tokMask: (1 << Int(tokenBits)) - 1,
            maxTokenLen: Int(hdr[H.fMaxTokenLen]),
            uni1: ptr(H.oUni1, UInt16.self), uni2: ptr(H.oUni2, UInt16.self),
            decompKeys: ptr(H.oDecompKeys, UInt32.self), decompVals: ptr(H.oDecompVals, UInt32.self),
            decompData: ptr(H.oDecompData, UInt32.self), nDecomp: Int(hdr[H.fNDecomp]),
            compKeys: ptr(H.oCompKeys, UInt64.self), compVals: ptr(H.oCompVals, UInt32.self),
            nComp: Int(hdr[H.fNComp]))
        nTotal = Int(hdr[H.fNTotal])
        specTab = ptr(H.oSpecials, UInt32.self)
        specBytes = ptr(H.oSpecBytes, UInt8.self)
        nSpecials = Int(hdr[H.fNSpecials])
        specFirstByte = Int32(bitPattern: hdr[H.fSpecFirstByte])
        imStart = Int32(bitPattern: hdr[H.fImStart])
        imEnd = Int32(bitPattern: hdr[H.fImEnd])
        endOfText = Int32(bitPattern: hdr[H.fEndOfText])
        // chat template: S0 "system\n" | S1 "\n" S0 "user\n" | S1 "\n" S0 "assistant\n"
        let off = core.tokOff, tb = core.tokBytes
        let s0 = (tb + Int(off[Int(imStart)]), Int(off[Int(imStart) + 1] - off[Int(imStart)]))
        let s1 = (tb + Int(off[Int(imEnd)]), Int(off[Int(imEnd) + 1] - off[Int(imEnd)]))
        let lits: [StaticString] = ["system\n", "\n", "user\n", "\n", "assistant\n"]
        let len = 3 * s0.1 + 2 * s1.1 + lits.reduce(0) { $0 + $1.utf8CodeUnitCount }
        let cb = UnsafeMutablePointer<UInt8>.allocate(capacity: len)
        var w = 0
        func put(_ p: UnsafePointer<UInt8>, _ n: Int) {
            (cb + w).initialize(from: p, count: n)
            w += n
        }
        put(s0.0, s0.1); put(lits[0].utf8Start, lits[0].utf8CodeUnitCount)
        let cut0 = w
        put(s1.0, s1.1); put(lits[1].utf8Start, 1); put(s0.0, s0.1); put(lits[2].utf8Start, lits[2].utf8CodeUnitCount)
        let cut1 = w
        put(s1.0, s1.1); put(lits[3].utf8Start, 1); put(s0.0, s0.1); put(lits[4].utf8Start, lits[4].utf8CodeUnitCount)
        chatBytes = cb
        chatLen = len
        chatCut = (cut0, cut1)
    }

    /// Every section of a compiled file lies inside the file (guards against truncated/corrupt files).
    static func sectionsInBounds(_ h: UnsafePointer<UInt32>, _ size: Int) -> Bool {
        let mergeBits = Int(h[H.fMergeBits]), tokenBits = Int(h[H.fTokenBits]), nTotal = Int(h[H.fNTotal])
        guard mergeBits >= 4 && mergeBits <= 30 && tokenBits >= 4 && tokenBits <= 30, nTotal > 256,
              nTotal < (1 << 18), Int(h[H.fNVocab]) <= nTotal, Int(h[H.fImStart]) < nTotal, Int(h[H.fImEnd]) < nTotal,
              Int(h[H.fEndOfText]) < nTotal, h[H.fMaxTokenLen] <= 255 else { return false }
        let need: [(Int, Int)] = [
            (H.oTokOff, (nTotal + 1) * 4), (H.oByteTok, 1024), (H.oByteProp, 512),
            (H.oMerge, (1 << mergeBits) * 8), (H.oTokTab, (1 << tokenBits) * 8),
            (H.oSpecials, Int(h[H.fNSpecials]) * 12), (H.oUni1, 0x1100 * 2), (H.oUni2, Int(h[H.fNUniBlocks]) * 512),
            (H.oDecompKeys, Int(h[H.fNDecomp]) * 4), (H.oDecompVals, Int(h[H.fNDecomp]) * 4),
            (H.oDecompData, Int(h[H.fNDecompData]) * 4), (H.oCompKeys, Int(h[H.fNComp]) * 8),
            (H.oCompVals, Int(h[H.fNComp]) * 4),
        ]
        for (field, bytes) in need where Int(h[field]) + bytes > size || Int(h[field]) % 8 != 0 { return false }
        let tokOff = (UnsafeRawPointer(h) + Int(h[H.oTokOff])).assumingMemoryBound(to: UInt32.self)
        return Int(h[H.oTokBytes]) + Int(tokOff[nTotal]) <= size
    }

    deinit {
        munmap(base, mappedSize)
        chatBytes.deallocate()
    }

    // MARK: Encoding

    /// Plain text: no added/special-token matching (a literal "<|im_end|>" is encoded as text).
    public func encode(_ text: String) -> [Int32] {
        withBytes(text) { encodeBytes($0, $1, specials: false) }
    }

    /// Added tokens (<|im_start|>, <|im_end|>, <|endoftext|>, <tool_call>, ...) are matched
    /// atomically, exactly like HF `Tokenizer.encode` (leftmost-longest, before normalization).
    public func encodeWithSpecials(_ text: String) -> [Int32] {
        withBytes(text) { encodeBytes($0, $1, specials: true) }
    }

    /// "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    /// tokenized exactly like HF (apply_chat_template + encode): added tokens inside system/user are
    /// matched too, as HF does.
    public func encodeChat(system: String, user: String) -> [Int32] {
        withBytes(system) { sp, sn in
            withBytes(user) { up, un in
                let total = chatLen + sn + un
                return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: total) { buf in
                    let o = buf.baseAddress!
                    var w = 0
                    @inline(__always) func put(_ p: UnsafePointer<UInt8>, _ n: Int) {
                        (o + w).update(from: p, count: n)
                        w += n
                    }
                    put(chatBytes, chatCut.0)
                    put(sp, sn)
                    put(chatBytes + chatCut.0, chatCut.1 - chatCut.0)
                    put(up, un)
                    put(chatBytes + chatCut.1, chatLen - chatCut.1)
                    return encodeBytes(UnsafePointer(o), w, specials: true)
                }
            }
        }
    }

    /// Encodes Qwen3-VL multimodal chat:
    /// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|> * N<|vision_end|>{user}<|im_end|>\n<|im_start|>assistant\n`
    /// Returns `(tokens, imageTokenOffset)` where `tokens[imageTokenOffset ..< imageTokenOffset + numImageTokens]`
    /// are the `imagePad` (151655) tokens.
    public func encodeVisionChat(system: String, user: String, numImageTokens: Int) -> (tokens: [Int32], imageTokenOffset: Int) {
        let prefix = encodeWithSpecials("<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n")
        let suffix = encodeWithSpecials("\(user)<|im_end|>\n<|im_start|>assistant\n")
        let imageTokenOffset = prefix.count + 1
        var out = [Int32]()
        out.reserveCapacity(prefix.count + 2 + numImageTokens + suffix.count)
        out.append(contentsOf: prefix)
        out.append(visionStart)
        for _ in 0..<numImageTokens { out.append(imagePad) }
        out.append(visionEnd)
        out.append(contentsOf: suffix)
        return (out, imageTokenOffset)
    }

    @inline(__always)
    func withBytes<R>(_ s: String, _ body: (UnsafePointer<UInt8>, Int) -> R) -> R {
        if let r = s.utf8.withContiguousStorageIfAvailable({ b -> R in
            if let a = b.baseAddress { return body(a, b.count) }
            return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 1) { body($0.baseAddress!, 0) }
        }) {
            return r
        }
        var c = s
        return c.withUTF8 { b in
            if let a = b.baseAddress { return body(a, b.count) }
            return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 1) { body($0.baseAddress!, 0) }
        }
    }

    func encodeBytes(_ p: UnsafePointer<UInt8>, _ n: Int, specials: Bool) -> [Int32] {
        if n == 0 { return [] }
        let core = self.core
        if !core.needsNFC(p, n) {
            // Fast path: text is already NFC; every piece yields at most one token per byte.
            return [Int32](unsafeUninitializedCapacity: n) { buf, count in
                var cnt = 0
                let out = buf.baseAddress!
                if specials {
                    splitSpecials(p, n) { a, b, id in
                        if id >= 0 {
                            out[cnt] = id
                            cnt += 1
                        } else {
                            core.encodeSegment(p + a, b - a, out, &cnt)
                        }
                    }
                } else {
                    core.encodeSegment(p, n, out, &cnt)
                }
                count = cnt
            }
        }
        return encodeSlow(p, n, specials: specials)
    }

    /// Text that needs NFC somewhere: normalize each non-special segment (HF matches added tokens on
    /// the raw text, then normalizes the remaining segments independently).
    func encodeSlow(_ p: UnsafePointer<UInt8>, _ n: Int, specials: Bool) -> [Int32] {
        let core = self.core
        var norm = [UInt8]()
        norm.reserveCapacity(n + n / 2 + 16)
        var segs = [(Int, Int, Int32)]()
        func addText(_ a: Int, _ b: Int) {
            let s = norm.count
            if core.needsNFC(p + a, b - a) {
                core.nfc(p + a, b - a, into: &norm)
            } else {
                norm.append(contentsOf: UnsafeBufferPointer(start: p + a, count: b - a))
            }
            segs.append((s, norm.count, -1))
        }
        if specials {
            splitSpecials(p, n) { a, b, id in
                if id >= 0 { segs.append((0, 0, id)) } else { addText(a, b) }
            }
        } else {
            addText(0, n)
        }
        return norm.withUnsafeBufferPointer { nb in
            [Int32](unsafeUninitializedCapacity: nb.count + segs.count + 1) { buf, count in
                var cnt = 0
                let out = buf.baseAddress!
                for (a, b, id) in segs {
                    if id >= 0 {
                        out[cnt] = id
                        cnt += 1
                    } else if b > a {
                        core.encodeSegment(nb.baseAddress! + a, b - a, out, &cnt)
                    }
                }
                count = cnt
            }
        }
    }

    /// Calls body(start, end, -1) for each text segment and body(0, 0, id) for each added token.
    @inline(__always)
    func splitSpecials(_ p: UnsafePointer<UInt8>, _ n: Int, _ body: (Int, Int, Int32) -> Void) {
        var segStart = 0
        var i = 0
        while i < n {
            var q: Int
            if specFirstByte >= 0 {
                guard let r = memchr(p + i, specFirstByte, n - i) else { break }
                q = UnsafeRawPointer(r) - UnsafeRawPointer(p)
            } else {
                q = i
            }
            if let (id, len) = matchSpecial(p, q, n) {
                if q > segStart { body(segStart, q, -1) }
                body(0, 0, id)
                i = q + len
                segStart = i
            } else {
                i = q + 1
            }
        }
        if segStart < n { body(segStart, n, -1) }
    }

    /// Longest added token starting at p[q], if any.
    @inline(__always)
    func matchSpecial(_ p: UnsafePointer<UInt8>, _ q: Int, _ n: Int) -> (Int32, Int)? {
        var bestLen = 0
        var bestId: Int32 = -1
        let rem = n - q
        for k in 0..<nSpecials {
            let len = Int(specTab[3 * k + 2])
            if len > rem || len <= bestLen { continue }
            let s = specBytes + Int(specTab[3 * k + 1])
            if s[0] == p[q] && bytesEqual(s, p + q, len) {
                bestLen = len
                bestId = Int32(bitPattern: specTab[3 * k])
            }
        }
        return bestId >= 0 ? (bestId, bestLen) : nil
    }

    // MARK: Decoding

    /// Raw bytes of a token (valid while the Tokenizer lives). Empty for ids without a token.
    @inline(__always)
    public func tokenBytes(_ id: Int32) -> UnsafeRawBufferPointer {
        guard id >= 0 && Int(id) < nTotal else { return UnsafeRawBufferPointer(start: nil, count: 0) }
        let s = Int(core.tokOff[Int(id)]), e = Int(core.tokOff[Int(id) + 1])
        return UnsafeRawBufferPointer(start: core.tokBytes + s, count: e - s)
    }

    /// Like HF decode(ids, skip_special_tokens=False): concatenated bytes, invalid UTF-8 replaced
    /// by U+FFFD; ids without a token are skipped.
    public func decode(_ ids: [Int32]) -> String {
        let off = core.tokOff
        var total = 0
        for id in ids where id >= 0 && Int(id) < nTotal {
            total += Int(off[Int(id) + 1] &- off[Int(id)])
        }
        if total == 0 { return "" }
        let bytes = core.tokBytes
        return String(unsafeUninitializedCapacity: total) { buf in
            var w = 0
            let o = buf.baseAddress!
            for id in ids where id >= 0 && Int(id) < nTotal {
                let s = Int(off[Int(id)]), len = Int(off[Int(id) + 1]) - s
                (o + w).update(from: bytes + s, count: len)
                w += len
            }
            return w
        }
    }

    // MARK: Diagnostics

    /// The NFC normalization HF applies before pre-tokenization.
    public func normalize(_ text: String) -> String {
        withBytes(text) { p, n in
            if !core.needsNFC(p, n) { return text }
            var out = [UInt8]()
            core.nfc(p, n, into: &out)
            return String(decoding: out, as: UTF8.self)
        }
    }

    /// Pre-tokenized pieces (after NFC) of plain text, as raw byte strings.
    public func preTokenize(_ text: String) -> [[UInt8]] {
        withBytes(normalize(text)) { p, n in
            var pieces = [[UInt8]]()
            var i = 0
            while i < n {
                let j = core.nextPieceEnd(p, i, n)
                pieces.append(Array(UnsafeBufferPointer(start: p + i, count: j - i)))
                i = j
            }
            return pieces
        }
    }

    /// Pre-map every page of the file (on macOS, MADV_WILLNEED maps resident pages: ~80 us for the
    /// whole file). Without it the first encode after init pays ~15-20 us of page faults. Call it from
    /// a background thread during startup to take that cost off the critical path.
    public func prefetch() {
        _ = madvise(base, mappedSize, MADV_WILLNEED)
    }
}

/// Incremental detokenizer for streaming output: feed token ids as they are generated and get back
/// text that is safe to show (an incomplete UTF-8 sequence at the end is held back until the next
/// token completes it). The concatenation of all returned chunks plus `finish()` equals
/// `decode(allIds)` exactly (invalid bytes become U+FFFD the same way).
public struct TextStream {
    let tok: Tokenizer
    var pending: [UInt8] = []

    public init(_ tok: Tokenizer) {
        self.tok = tok
        pending.reserveCapacity(16)
    }

    public mutating func push(_ id: Int32) -> String {
        let b = tok.tokenBytes(id)
        if b.isEmpty { return "" }
        pending.append(contentsOf: b)
        // Hold back a trailing prefix of a valid multi-byte sequence (at most 3 bytes).
        let n = pending.count
        var keep = 0
        var k = n - 1
        while k >= 0 && n - k <= 3 {
            let c = pending[k]
            if c & 0xC0 == 0x80 {  // continuation byte: keep looking for the lead
                k -= 1
                continue
            }
            let need = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xC2 && c <= 0xDF ? 2 : 1
            if need > n - k && TextStream.validPrefix(pending, k) { keep = n - k }
            break
        }
        if keep == n { return "" }
        let out = String(decoding: pending[0..<(n - keep)], as: UTF8.self)
        pending.removeFirst(n - keep)
        return out
    }

    /// Flush whatever is held back (an unfinished sequence decodes to U+FFFD, as in `decode`).
    public mutating func finish() -> String {
        defer { pending.removeAll(keepingCapacity: true) }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }

    /// pending[k...] is a proper prefix of some well-formed UTF-8 sequence.
    static func validPrefix(_ p: [UInt8], _ k: Int) -> Bool {
        let c = p[k]
        var i = k + 1
        if i < p.count {
            let d = p[i]
            let ok: Bool
            switch c {
            case 0xE0: ok = d >= 0xA0 && d <= 0xBF
            case 0xED: ok = d >= 0x80 && d <= 0x9F
            case 0xF0: ok = d >= 0x90 && d <= 0xBF
            case 0xF4: ok = d >= 0x80 && d <= 0x8F
            default: ok = d & 0xC0 == 0x80
            }
            if !ok { return false }
            i += 1
        }
        while i < p.count {
            if p[i] & 0xC0 != 0x80 { return false }
            i += 1
        }
        return true
    }
}
