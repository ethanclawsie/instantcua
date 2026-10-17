// Hot path of the Qwen2 tokenizer: hand-written pre-tokenizer (exact Oniguruma semantics of the
// Qwen2 split regex), token-table fast path, BPE merge loop, and NFC. Everything reads the mmapped
// compiled file through raw pointers; nothing here allocates on the common (ASCII/CJK) path.
//
// Split regex (leftmost-first alternation, applied at every position since every char starts a match):
//   A1 (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   A2 [^\r\n\p{L}\p{N}]?\p{L}+
//   A3 \p{N}
//   A4  ?[^\s\p{L}\p{N}]+[\r\n]*
//   A5 \s*[\r\n]+
//   A6 \s+(?!\S)
//   A7 \s+

// Property bits (see tools/tokenizer_unicode.py).
let cL: UInt16 = 0, cN: UInt16 = 1, cO: UInt16 = 2, cWS: UInt16 = 3, cSP: UInt16 = 4, cNL: UInt16 = 5
let cNonASCII: UInt16 = 7
let fNFC: UInt16 = 1 << 3, fC1: UInt16 = 1 << 4, fRV: UInt16 = 1 << 5, fE: UInt16 = 1 << 6, fLL: UInt16 = 1 << 7

let mergeHashMul: UInt64 = 0x9E37_79B9_7F4A_7C15
let noRank: UInt32 = .max
let smallBPEMax = 64

// Hangul
let sBase: UInt32 = 0xAC00, lBase: UInt32 = 0x1100, vBase: UInt32 = 0x1161, tBase: UInt32 = 0x11A7
let lCount: UInt32 = 19, vCount: UInt32 = 21, tCount: UInt32 = 28
let nCount: UInt32 = 21 * 28, sCount: UInt32 = 19 * 21 * 28

@inline(__always) func load64(_ p: UnsafePointer<UInt8>) -> UInt64 {
    UnsafeRawPointer(p).loadUnaligned(as: UInt64.self)
}
@inline(__always) func load32(_ p: UnsafePointer<UInt8>) -> UInt32 {
    UnsafeRawPointer(p).loadUnaligned(as: UInt32.self)
}

/// Hash of a byte string (used identically by the compiler and the lookup). Never reads outside [p, p+n).
@inline(__always) func hashBytes(_ p: UnsafePointer<UInt8>, _ n: Int) -> UInt64 {
    var a: UInt64
    var b: UInt64
    if n >= 8 {
        a = load64(p)
        b = load64(p + (n - 8))
        if n > 16 {
            var i = 8
            while i < n - 8 {
                a = (a ^ load64(p + i)) &* 0xFF51_AFD7_ED55_8CCD
                a ^= a >> 31
                i += 8
            }
        }
    } else if n >= 4 {
        a = UInt64(load32(p))
        b = UInt64(load32(p + (n - 4)))
    } else {
        a = UInt64(p[0]) | UInt64(p[n >> 1]) << 8 | UInt64(p[n - 1]) << 16
        b = 0
    }
    var h = (a &* 0x9E37_79B9_7F4A_7C15) ^ (b &* 0xC2B2_AE3D_27D4_EB4F) ^ (UInt64(n) &* 0x1656_67B1_9E37_79F9)
    h ^= h >> 32
    h = h &* 0xD6E8_FEB8_6659_FD93
    h ^= h >> 29
    return h
}

// Token table slot layout: id (18 bits) | len (8) << 18 | offset (20) << 26 | fingerprint (18) << 46.
let slotLenShift: UInt64 = 18, slotOffShift: UInt64 = 26, slotFpShift: UInt64 = 46

/// All lookup tables, as raw pointers into the mmapped file (or into compiler buffers).
struct Core {
    var tokOff: UnsafePointer<UInt32>
    var tokBytes: UnsafePointer<UInt8>
    var byteTok: UnsafePointer<Int32>
    var byteProp: UnsafePointer<UInt16>
    var mergeTab: UnsafePointer<UInt64>
    var mergeShift: UInt64
    var mergeMask: Int
    var tokTab: UnsafePointer<UInt64>
    var tokShift: UInt64
    var tokMask: Int
    var maxTokenLen: Int
    var uni1: UnsafePointer<UInt16>
    var uni2: UnsafePointer<UInt16>
    var decompKeys: UnsafePointer<UInt32>
    var decompVals: UnsafePointer<UInt32>
    var decompData: UnsafePointer<UInt32>
    var nDecomp: Int
    var compKeys: UnsafePointer<UInt64>
    var compVals: UnsafePointer<UInt32>
    var nComp: Int

    // MARK: Unicode properties

    @inline(__always) func uniProp(_ cp: UInt32) -> UInt16 {
        let blk = Int(uni1[Int(cp >> 8)])
        return uni2[blk << 8 | Int(cp & 0xFF)]
    }

    /// Property and byte length of the (valid UTF-8) char at a non-ASCII lead byte.
    @inline(__always) func propNonASCII(_ p: UnsafePointer<UInt8>, _ i: Int) -> (UInt16, Int) {
        let (cp, l) = decodeUTF8(p, i)
        return (uniProp(cp), l)
    }

    @inline(__always) func propAt(_ p: UnsafePointer<UInt8>, _ i: Int) -> (UInt16, Int) {
        let b = p[i]
        if b < 0x80 { return (byteProp[Int(b)], 1) }
        return propNonASCII(p, i)
    }

    // MARK: Pre-tokenizer

    @inline(__always) func scanClass(_ p: UnsafePointer<UInt8>, _ j0: Int, _ n: Int, _ cls: UInt16) -> Int {
        var j = j0
        while j < n {
            let b = p[j]
            if b < 0x80 {
                if byteProp[Int(b)] & 7 != cls { return j }
                j += 1
            } else {
                let (pr, l) = propNonASCII(p, j)
                if pr & 7 != cls { return j }
                j += l
            }
        }
        return j
    }

    @inline(__always) func scanNL(_ p: UnsafePointer<UInt8>, _ j0: Int, _ n: Int) -> Int {
        var j = j0
        while j < n && (p[j] == 0x0A || p[j] == 0x0D) { j += 1 }
        return j
    }

    /// A5/A6/A7 for a whitespace run starting at i.
    @inline(__always) func wsRun(_ p: UnsafePointer<UInt8>, _ i: Int, _ n: Int) -> Int {
        var j = i
        var lastNL = -1
        var lastStart = i
        while j < n {
            let b = p[j]
            if b < 0x80 {
                let c = byteProp[Int(b)] & 7
                if c < cWS { break }
                if c == cNL { lastNL = j }
                lastStart = j
                j += 1
            } else {
                let (pr, l) = propNonASCII(p, j)
                if pr & 7 < cWS { break }
                lastStart = j
                j += l
            }
        }
        if lastNL >= 0 { return lastNL + 1 }   // A5: up to and including the last CR/LF
        if j == n { return n }                 // A6: run reaches the end
        if lastStart > i { return lastStart }  // A6: give back the last whitespace char
        return j                               // A7: single whitespace char before non-space
    }

    /// A1: k is the index just after the apostrophe. Returns the end of the contraction or -1.
    @inline(__always) func contraction(_ p: UnsafePointer<UInt8>, _ k: Int, _ n: Int) -> Int {
        if k >= n { return -1 }
        let (p1, l1) = propAt(p, k)
        if p1 & fC1 != 0 { return k + l1 }
        if p1 & (fRV | fLL) != 0 {
            let k2 = k + l1
            if k2 >= n { return -1 }
            let (p2, l2) = propAt(p, k2)
            if (p1 & fRV != 0 && p2 & fE != 0) || (p1 & fLL != 0 && p2 & fLL != 0) { return k2 + l2 }
        }
        return -1
    }

    /// End of the pre-tokenizer piece starting at i (leftmost-first match of the split regex at i).
    @inline(__always)
    func nextPieceEnd(_ p: UnsafePointer<UInt8>, _ i: Int, _ n: Int) -> Int {
        let b = p[i]
        let (pr, l) = propAt(p, i)
        switch pr & 7 {
        case 0:  // L: A2 without prefix
            return scanClass(p, i + l, n, cL)
        case 1:  // N: A3
            return i + l
        case 2:  // O
            if b == 0x27 {
                let k = contraction(p, i + 1, n)
                if k >= 0 { return k }  // A1
            }
            let nx = i + l
            if nx >= n { return nx }
            let (pn, ln) = propAt(p, nx)
            if pn & 7 == cL { return scanClass(p, nx + ln, n, cL) }  // A2: punct + letters
            return scanNL(p, scanClass(p, nx, n, cO), n)            // A4
        case 4:  // SPACE
            let nx = i + 1
            if nx >= n { return n }
            let (pn, ln) = propAt(p, nx)
            let cn = pn & 7
            if cn == cL { return scanClass(p, nx + ln, n, cL) }                // A2: " word"
            if cn == cO { return scanNL(p, scanClass(p, nx + ln, n, cO), n) }  // A4: " +=\n"
            return wsRun(p, i, n)
        case 3:  // other whitespace (tab, NBSP, U+3000, ...)
            let nx = i + l
            if nx >= n { return n }
            let (pn, ln) = propAt(p, nx)
            if pn & 7 == cL { return scanClass(p, nx + ln, n, cL) }  // A2: "\tword"
            return wsRun(p, i, n)
        default:  // CR / LF
            return wsRun(p, i, n)
        }
    }

    /// Pre-tokenize one NFC-normalized segment and BPE-encode every piece into out[cnt...].
    /// Capacity required: n tokens.
    @inline(__always)
    func encodeSegment(_ p: UnsafePointer<UInt8>, _ n: Int, _ out: UnsafeMutablePointer<Int32>, _ cnt: inout Int) {
        var i = 0
        while i < n {
            let j = nextPieceEnd(p, i, n)
            emit(p + i, j - i, out, &cnt)
            i = j
        }
    }

    // MARK: Token lookup and BPE

    @inline(__always) func lookupToken(_ s: UnsafePointer<UInt8>, _ len: Int) -> Int32 {
        let h = hashBytes(s, len)
        var slot = Int(truncatingIfNeeded: h >> tokShift)
        let want = (h & 0x3FFFF) << slotFpShift | UInt64(len) << slotLenShift
        let mask: UInt64 = 0x3FFFF << slotFpShift | 0xFF << slotLenShift
        while true {
            let e = tokTab[slot]
            if e == 0 { return -1 }
            if e & mask == want {
                let off = Int(truncatingIfNeeded: (e >> slotOffShift) & 0xFFFFF)
                if bytesEqual(tokBytes + off, s, len) { return Int32(truncatingIfNeeded: e & 0x3FFFF) }
            }
            slot = (slot + 1) & tokMask
        }
    }

    @inline(__always) func pairRank(_ a: Int32, _ b: Int32) -> UInt32 {
        let key = UInt64(UInt32(bitPattern: a)) << 18 | UInt64(UInt32(bitPattern: b))
        var slot = Int(truncatingIfNeeded: (key &* mergeHashMul) >> mergeShift)
        while true {
            let e = mergeTab[slot]
            if e >> 18 == key { return UInt32(truncatingIfNeeded: e & 0x3FFFF) }
            if e == .max { return noRank }
            slot = (slot + 1) & mergeMask
        }
    }

    @inline(__always)
    func emit(_ s: UnsafePointer<UInt8>, _ len: Int, _ out: UnsafeMutablePointer<Int32>, _ cnt: inout Int) {
        if len == 1 {
            out[cnt] = byteTok[Int(s[0])]
            cnt += 1
            return
        }
        if len <= maxTokenLen {
            let id = lookupToken(s, len)
            if id >= 0 {
                out[cnt] = id
                cnt += 1
                return
            }
        }
        bpe(s, len, out, &cnt)
    }

    /// Byte-pair merge of one piece: repeatedly merge the adjacent pair with the lowest merge rank,
    /// leftmost first (exactly HF's (rank, pos) min-heap order).
    @inline(never)
    func bpe(_ s: UnsafePointer<UInt8>, _ n: Int, _ out: UnsafeMutablePointer<Int32>, _ cnt: inout Int) {
        if n > smallBPEMax {
            bpeLarge(s, n, out, &cnt)
            return
        }
        withUnsafeTemporaryAllocation(of: Int32.self, capacity: smallBPEMax) { symBuf in
            withUnsafeTemporaryAllocation(of: UInt32.self, capacity: smallBPEMax) { rankBuf in
                let sym = symBuf.baseAddress!
                let rank = rankBuf.baseAddress!
                for i in 0..<n { sym[i] = byteTok[Int(s[i])] }
                var m = n
                for i in 0..<(m - 1) { rank[i] = pairRank(sym[i], sym[i + 1]) }
                while m > 1 {
                    var best = noRank
                    var bi = 0
                    for i in 0..<(m - 1) where rank[i] < best {
                        best = rank[i]
                        bi = i
                    }
                    if best == noRank { break }
                    sym[bi] = Int32(bitPattern: best &+ 256)
                    m -= 1
                    var k = bi + 1
                    while k < m {
                        sym[k] = sym[k + 1]
                        k += 1
                    }
                    k = bi + 1
                    while k < m - 1 {
                        rank[k] = rank[k + 1]
                        k += 1
                    }
                    if bi > 0 { rank[bi - 1] = pairRank(sym[bi - 1], sym[bi]) }
                    if bi < m - 1 { rank[bi] = pairRank(sym[bi], sym[bi + 1]) }
                }
                for i in 0..<m {
                    out[cnt] = sym[i]
                    cnt += 1
                }
            }
        }
    }

    /// O(n log n) BPE for long pieces (same order as HF: binary heap keyed by (rank, original pos)).
    func bpeLarge(_ s: UnsafePointer<UInt8>, _ n: Int, _ out: UnsafeMutablePointer<Int32>, _ cnt: inout Int) {
        var sym = [Int32](repeating: 0, count: n)
        var prev = [Int32](repeating: 0, count: n)
        var next = [Int32](repeating: 0, count: n)
        var heap = [UInt64]()
        heap.reserveCapacity(n)
        for i in 0..<n {
            sym[i] = byteTok[Int(s[i])]
            prev[i] = Int32(i - 1)
            next[i] = i + 1 < n ? Int32(i + 1) : -1
        }
        func push(_ v: UInt64) {
            heap.append(v)
            var c = heap.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if heap[p] <= v { break }
                heap[c] = heap[p]
                c = p
            }
            heap[c] = v
        }
        func pop() -> UInt64 {
            let top = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                var c = 0
                let cnt = heap.count
                while true {
                    var ch = 2 * c + 1
                    if ch >= cnt { break }
                    if ch + 1 < cnt && heap[ch + 1] < heap[ch] { ch += 1 }
                    if heap[ch] >= last { break }
                    heap[c] = heap[ch]
                    c = ch
                }
                heap[c] = last
            }
            return top
        }
        for i in 0..<(n - 1) {
            let r = pairRank(sym[i], sym[i + 1])
            if r != noRank { push(UInt64(r) << 32 | UInt64(i)) }
        }
        while !heap.isEmpty {
            let top = pop()
            let r = UInt32(truncatingIfNeeded: top >> 32)
            let pos = Int(truncatingIfNeeded: top & 0xFFFF_FFFF)
            if sym[pos] < 0 { continue }
            let nx = Int(next[pos])
            if nx < 0 { continue }
            if pairRank(sym[pos], sym[nx]) != r { continue }  // stale entry
            sym[pos] = Int32(bitPattern: r &+ 256)
            sym[nx] = -1
            let nn = next[nx]
            next[pos] = nn
            if nn >= 0 { prev[Int(nn)] = Int32(pos) }
            let pv = Int(prev[pos])
            if pv >= 0 {
                let r2 = pairRank(sym[pv], sym[pos])
                if r2 != noRank { push(UInt64(r2) << 32 | UInt64(pv)) }
            }
            if nn >= 0 {
                let r3 = pairRank(sym[pos], sym[Int(nn)])
                if r3 != noRank { push(UInt64(r3) << 32 | UInt64(pos)) }
            }
        }
        var i = 0
        while i >= 0 && i < n {
            out[cnt] = sym[i]
            cnt += 1
            i = Int(next[i])
        }
    }

    // MARK: NFC

    /// True if any char in the text may change under NFC (the NFC flag is only set for cp >= U+0300,
    /// whose UTF-8 lead byte is >= 0xCC).
    @inline(__always) func needsNFC(_ p: UnsafePointer<UInt8>, _ n: Int) -> Bool {
        var i = 0
        while i < n {
            if i + 8 <= n && load64(p + i) & 0x8080_8080_8080_8080 == 0 {
                i += 8
                continue
            }
            let b = p[i]
            if b < 0xCC {
                i += 1
                continue
            }
            let (pr, l) = propNonASCII(p, i)
            if pr & fNFC != 0 { return true }
            i += l
        }
        return false
    }

    @inline(__always) func ccc(_ cp: UInt32) -> UInt16 { uniProp(cp) >> 8 }

    func decompose(_ cp: UInt32, into cps: inout [UInt32]) {
        if cp >= sBase && cp < sBase + sCount {
            let s = cp - sBase
            cps.append(lBase + s / nCount)
            cps.append(vBase + (s % nCount) / tCount)
            let t = s % tCount
            if t != 0 { cps.append(tBase + t) }
            return
        }
        if cp >= 0xC0 {
            var lo = 0, hi = nDecomp
            while lo < hi {
                let mid = (lo + hi) >> 1
                if decompKeys[mid] < cp { lo = mid + 1 } else { hi = mid }
            }
            if lo < nDecomp && decompKeys[lo] == cp {
                let v = decompVals[lo]
                let off = Int(v >> 8), len = Int(v & 0xFF)
                for k in 0..<len { cps.append(decompData[off + k]) }
                return
            }
        }
        cps.append(cp)
    }

    func composePair(_ a: UInt32, _ b: UInt32) -> UInt32? {
        if a >= lBase && a < lBase + lCount && b >= vBase && b < vBase + vCount {
            return sBase + ((a - lBase) * vCount + (b - vBase)) * tCount
        }
        if a >= sBase && a < sBase + sCount && (a - sBase) % tCount == 0 && b > tBase && b < tBase + tCount {
            return a + (b - tBase)
        }
        let key = UInt64(a) << 21 | UInt64(b)
        var lo = 0, hi = nComp
        while lo < hi {
            let mid = (lo + hi) >> 1
            if compKeys[mid] < key { lo = mid + 1 } else { hi = mid }
        }
        if lo < nComp && compKeys[lo] == key { return compVals[lo] }
        return nil
    }

    /// NFC exactly as unicode-normalization(-alignments): full canonical decomposition, canonical
    /// ordering, canonical composition. Appends UTF-8 to `out`.
    func nfc(_ p: UnsafePointer<UInt8>, _ n: Int, into out: inout [UInt8]) {
        var cps = [UInt32]()
        cps.reserveCapacity(n + 8)
        var i = 0
        while i < n {
            let (cp, l) = decodeUTF8(p, i)
            decompose(cp, into: &cps)
            i += l
        }
        // canonical ordering (stable insertion sort of each run of non-starters)
        let m = cps.count
        var a = 0
        while a < m {
            if ccc(cps[a]) == 0 {
                a += 1
                continue
            }
            var e = a
            while e < m && ccc(cps[e]) != 0 { e += 1 }
            if e - a > 1 {
                for x in (a + 1)..<e {
                    let v = cps[x]
                    let cv = ccc(v)
                    var y = x
                    while y > a && ccc(cps[y - 1]) > cv {
                        cps[y] = cps[y - 1]
                        y -= 1
                    }
                    cps[y] = v
                }
            }
            a = e
        }
        // canonical composition
        var w = 0
        var starter = -1
        var lastCCC = -1
        for x in 0..<m {
            let c = cps[x]
            let k = Int(ccc(c))
            if starter >= 0 && (lastCCC == -1 || lastCCC < k), let r = composePair(cps[starter], c) {
                cps[starter] = r
                continue
            }
            if k == 0 {
                starter = w
                lastCCC = -1
            } else if starter >= 0 {
                lastCCC = k
            }
            cps[w] = c
            w += 1
        }
        for x in 0..<w { appendUTF8(cps[x], &out) }
    }
}

@inline(__always) func decodeUTF8(_ p: UnsafePointer<UInt8>, _ i: Int) -> (UInt32, Int) {
    let b0 = UInt32(p[i])
    if b0 < 0x80 { return (b0, 1) }
    if b0 < 0xE0 { return ((b0 & 0x1F) << 6 | (UInt32(p[i + 1]) & 0x3F), 2) }
    if b0 < 0xF0 {
        return ((b0 & 0x0F) << 12 | (UInt32(p[i + 1]) & 0x3F) << 6 | (UInt32(p[i + 2]) & 0x3F), 3)
    }
    return ((b0 & 0x07) << 18 | (UInt32(p[i + 1]) & 0x3F) << 12 | (UInt32(p[i + 2]) & 0x3F) << 6
        | (UInt32(p[i + 3]) & 0x3F), 4)
}

@inline(__always) func appendUTF8(_ c: UInt32, _ out: inout [UInt8]) {
    if c < 0x80 {
        out.append(UInt8(c))
    } else if c < 0x800 {
        out.append(UInt8(0xC0 | c >> 6))
        out.append(UInt8(0x80 | c & 0x3F))
    } else if c < 0x10000 {
        out.append(UInt8(0xE0 | c >> 12))
        out.append(UInt8(0x80 | (c >> 6) & 0x3F))
        out.append(UInt8(0x80 | c & 0x3F))
    } else {
        out.append(UInt8(0xF0 | c >> 18))
        out.append(UInt8(0x80 | (c >> 12) & 0x3F))
        out.append(UInt8(0x80 | (c >> 6) & 0x3F))
        out.append(UInt8(0x80 | c & 0x3F))
    }
}

@inline(__always) func bytesEqual(_ a: UnsafePointer<UInt8>, _ b: UnsafePointer<UInt8>, _ n: Int) -> Bool {
    var i = 0
    while i + 8 <= n {
        if load64(a + i) != load64(b + i) { return false }
        i += 8
    }
    while i < n {
        if a[i] != b[i] { return false }
        i += 1
    }
    return true
}
