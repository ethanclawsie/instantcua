import CShim
import Foundation
import Metal

public final class Engine {
    public let device: MTLDevice
    public let model: ModelFile
    public let format: WeightFormat
    public let maxContext: Int
    public let maxPass: Int
    public let maxSteps: Int
    public var eos: (Int32, Int32) = (151645, 151643)

    let queue: MTLCommandQueue
    let wbuf: MTLBuffer

    // Vision buffers
    let rgbBuf, patchBuf, vx, vxn, vq, vk, vv, vattn, vhbuf, vscores, vprobs: MTLBuffer
    let vPosInterp, vRope: MTLBuffer
    public let poolerBuf: MTLBuffer
    public let deepstackBufs: [MTLBuffer]

    // LLM buffers
    let x, xh, qkvBuf, q, attn, attnh, hbuf, hbufh, kc, vc, state, promptToks, outToks, pval, pidx: MTLBuffer
    public let logits: MTLBuffer
    let rope: MTLBuffer

    // Compute pipelines
    let pVitPatchify, pVitPatchEmbed, pVitLN1024, pVitLN4096, pVitQkv, pVitAttn, pVitProj, pVitFc1, pVitFc2: MTLComputePipelineState
    let pMergerFc1, pMergerFc2: MTLComputePipelineState
    let pEmbedPrompt, pDeepstackAdd, pAttnIn, pAttention, pOProj, pMlpIn, pMlpOut, pLMHead, pCommit, pNoop: MTLComputePipelineState
    let pRmsNormRowsH, pQkNormRope, pPrefillAttn, pG3Qkv, pG3O, pG3Gu, pG3Down: MTLComputePipelineState

    var residency: MTLResidencySet?
    private var preparedGrid: (Int, Int) = (0, 0)
    private var preparedMRopeKey: (Int, Int, Int) = (-999, 0, 0)
    private let llmInvFreq: [Double]

    public struct Timing {
        public var visionMs: Double = 0
        public var prefillMs: Double = 0
        public var decodeMs: Double = 0
        public var totalMs: Double = 0
        public var prefillTokens: Int = 0
        public var generatedTokens: Int = 0
    }
    public private(set) var lastTiming = Timing()

    public init(modelPath: String, maxContext: Int = 2048, maxPass: Int = 512, maxSteps: Int = 256,
                copyWeights: Bool = false, pinMemory: Bool = true) throws {
        guard let dev = MTLCopyAllDevices().first ?? MTLCreateSystemDefaultDevice() else {
            throw EngineError("no Metal device")
        }
        device = dev
        model = try ModelFile(path: modelPath)
        format = model.format
        self.maxContext = maxContext
        self.maxPass = maxPass
        self.maxSteps = maxSteps
        queue = dev.makeCommandQueue()!

        if copyWeights {
            guard let wb = dev.makeBuffer(bytes: model.base, length: model.length,
                                          options: [.storageModeShared, .hazardTrackingModeUntracked])
            else { throw EngineError("cannot allocate weight buffer") }
            wbuf = wb
        } else {
            guard let wb = dev.makeBuffer(bytesNoCopy: model.base, length: model.length,
                                          options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: nil)
            else { throw EngineError("cannot wrap model weights in Metal buffer") }
            wbuf = wb
        }

        func buf(_ bytes: Int) -> MTLBuffer {
            dev.makeBuffer(length: max(bytes, 16), options: .storageModeShared)!
        }

        let maxPatches = 1024 // 640x384 uses 960 patches (24x40), 240 merged vision tokens
        let maxVis = maxPatches / 4
        rgbBuf = buf(maxPatches * 16 * 16 * 3)
        patchBuf = buf(maxPatches * Dims.vitPatchDim * 4)
        vx = buf(maxPatches * Dims.vitHidden * 4)
        vxn = buf(maxPatches * Dims.vitHidden * 4)
        vq = buf(maxPatches * Dims.vitHidden * 2)
        vk = buf(maxPatches * Dims.vitHidden * 2)
        vv = buf(maxPatches * Dims.vitHidden * 2)
        vattn = buf(maxPatches * Dims.vitHidden * 2)
        vhbuf = buf(maxPatches * Dims.vitFF * 2)
        vscores = buf(Dims.vitHeads * maxPatches * maxPatches * 4)
        vprobs = buf(Dims.vitHeads * maxPatches * maxPatches * 2)
        vPosInterp = buf(maxPatches * Dims.vitHidden * 4)
        vRope = buf(maxPatches * 32 * 8)
        poolerBuf = buf(maxVis * Dims.hidden * 4)
        deepstackBufs = (0..<Dims.numDeepstack).map { _ in buf(maxVis * Dims.hidden * 4) }

        let H = Dims.hidden
        x = buf(maxPass * H * 4)
        xh = buf(maxPass * H * 2)
        qkvBuf = buf(maxPass * Dims.qkvRows * 4)
        q = buf(maxPass * H * 4)
        attn = buf(maxPass * H * 4)
        attnh = buf(maxPass * H * 2)
        hbuf = buf(maxPass * Dims.ff * 4)
        hbufh = buf(maxPass * Dims.ff * 2)
        let kvBytes = Dims.layers * Dims.kvHeads * maxContext * Dims.headDim * 2
        kc = buf(kvBytes)
        vc = buf(kvBytes)
        state = buf(64)
        promptToks = buf(maxPass * 4)
        outToks = buf(maxSteps * 4)
        pval = buf(Dims.vocab / 32 * 4)
        pidx = buf(Dims.vocab / 32 * 4)
        logits = buf(Dims.vocab * 4)
        rope = buf(maxContext * 64 * 8)

        var invf = [Double](repeating: 0, count: 64)
        for i in 0..<64 {
            invf[i] = 1.0 / pow(Dims.ropeTheta, Double(2 * i) / Double(Dims.headDim))
        }
        llmInvFreq = invf

        let rd = MTLResidencySetDescriptor()
        rd.initialCapacity = 40
        if pinMemory, let rs = try? dev.makeResidencySet(descriptor: rd) {
            var allBufs: [MTLBuffer] = [
                wbuf, rgbBuf, patchBuf, vx, vxn, vq, vk, vv, vattn, vhbuf, vscores, vprobs, vPosInterp, vRope, poolerBuf,
                x, xh, qkvBuf, q, attn, attnh, hbuf, hbufh, kc, vc, state, promptToks, outToks, pval, pidx, logits, rope,
            ]
            allBufs.append(contentsOf: deepstackBufs)
            for b in allBufs { rs.addAllocation(b) }
            rs.commit()
            rs.requestResidency()
            queue.addResidencySet(rs)
            residency = rs
        }

        let opts = MTLCompileOptions()
        opts.preprocessorMacros = ["FMT": NSNumber(value: model.format.rawValue)]
        opts.mathMode = .safe
        opts.languageVersion = .version4_0
        let lib = try dev.makeLibrary(source: shaderSource, options: opts)

        func pipe(_ name: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: name) else { throw EngineError("missing kernel \(name)") }
            return try dev.makeComputePipelineState(function: f)
        }

        pVitPatchify = try pipe("k_vit_patchify_rgb")
        pVitPatchEmbed = try pipe("k_vit_patch_embed")
        pVitLN1024 = try pipe("k_vit_ln1024")
        pVitLN4096 = try pipe("k_vit_ln4096")
        pVitQkv = try pipe("k_vit_qkv")
        pVitAttn = try pipe("k_vit_attn")
        pVitProj = try pipe("k_vit_proj")
        pVitFc1 = try pipe("k_vit_fc1")
        pVitFc2 = try pipe("k_vit_fc2")
        pMergerFc1 = try pipe("k_merger_fc1")
        pMergerFc2 = try pipe("k_merger_fc2")

        pEmbedPrompt = try pipe("k_embed_prompt")
        pDeepstackAdd = try pipe("k_deepstack_add")
        pAttnIn = try pipe("k_attn_in")
        pAttention = try pipe("k_attention")
        pOProj = try pipe("k_o_proj")
        pMlpIn = try pipe("k_mlp_in")
        pMlpOut = try pipe("k_mlp_out")
        pLMHead = try pipe("k_lm_head")
        pCommit = try pipe("k_commit")
        pNoop = try pipe("k_noop")

        pRmsNormRowsH = try pipe("k_rmsnorm_rows_h")
        pQkNormRope = try pipe("k_qk_norm_rope")
        pPrefillAttn = try pipe("k_prefill_attn")
        pG3Qkv = try pipe("k_g3_qkv")
        pG3O = try pipe("k_g3_o")
        pG3Gu = try pipe("k_g3_gu")
        pG3Down = try pipe("k_g3_down")

        prepareVisionGrid(gridH: 24, gridW: 40)
    }

    // Precomputes bilinearly interpolated pos_embed [np, 1024] and 2D Vision RoPE [np, 32] (cos, sin)
    // in merge-block order for (gridH, gridW).
    public func prepareVisionGrid(gridH: Int, gridW: Int) {
        if preparedGrid == (gridH, gridW) { return }
        preparedGrid = (gridH, gridW)
        let np = gridH * gridW
        let mergeH = gridH / 2
        let mergeW = gridW / 2
        let posSrc = model.pointer(TensorID.vitPosEmb).assumingMemoryBound(to: Float.self)
        let posDst = vPosInterp.contents().assumingMemoryBound(to: Float.self)
        let ropeDst = vRope.contents().assumingMemoryBound(to: Float.self)

        let numGrid = Dims.vitPosGrid // 48
        var invFreq = [Float](repeating: 0, count: 16)
        for i in 0..<16 {
            invFreq[i] = Float(1.0 / pow(Dims.vitRopeTheta, Double(2 * i) / 32.0))
        }

        var pIdx = 0
        for br in 0..<mergeH {
            for bc in 0..<mergeW {
                for ir in 0..<2 {
                    for ic in 0..<2 {
                        let r = br * 2 + ir
                        let c = bc * 2 + ic

                        // Bilinear interpolation of 48x48 pos_embed with align_corners=True
                        let y = Float(r) * Float(numGrid - 1) / Float(max(1, gridH - 1))
                        let x = Float(c) * Float(numGrid - 1) / Float(max(1, gridW - 1))
                        let y0 = Int(floor(y)), x0 = Int(floor(x))
                        let y1 = min(numGrid - 1, y0 + 1), x1 = min(numGrid - 1, x0 + 1)
                        let dy = y - Float(y0), dx = x - Float(x0)
                        let w00 = (1.0 - dy) * (1.0 - dx)
                        let w01 = (1.0 - dy) * dx
                        let w10 = dy * (1.0 - dx)
                        let w11 = dy * dx

                        let p00 = posSrc + (y0 * numGrid + x0) * Dims.vitHidden
                        let p01 = posSrc + (y0 * numGrid + x1) * Dims.vitHidden
                        let p10 = posSrc + (y1 * numGrid + x0) * Dims.vitHidden
                        let p11 = posSrc + (y1 * numGrid + x1) * Dims.vitHidden
                        let dstP = posDst + pIdx * Dims.vitHidden
                        for d in 0..<Dims.vitHidden {
                            dstP[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d]
                        }

                        // 2D Vision RoPE: first 16 dims = row * invFreq, next 16 dims = col * invFreq
                        let dstR = ropeDst + pIdx * 32 * 2
                        for i in 0..<16 {
                            let ah = Double(Float(r) * invFreq[i])
                            dstR[i * 2 + 0] = Float(cos(ah))
                            dstR[i * 2 + 1] = Float(sin(ah))
                            let aw = Double(Float(c) * invFreq[i])
                            dstR[(16 + i) * 2 + 0] = Float(cos(aw))
                            dstR[(16 + i) * 2 + 1] = Float(sin(aw))
                        }

                        pIdx += 1
                    }
                }
            }
        }
        _ = np
    }

    // Populates the 3D Interleaved M-RoPE table [maxContext, 64] (cos, sin) for a multimodal prompt
    // with vision tokens starting at visOffset and merged grid (llmH, llmW).
    public func prepareMRoPE(visOffset: Int, llmH: Int, llmW: Int) {
        let key = (visOffset, llmH, llmW)
        if preparedMRopeKey == key { return }
        preparedMRopeKey = key

        let nVis = llmH * llmW
        let ropeDelta = max(llmH, llmW) - nVis
        let rPtr = rope.contents().assumingMemoryBound(to: Float.self)

        for slot in 0..<maxContext {
            let posT: Int
            let posH: Int
            let posW: Int
            if visOffset < 0 || slot < visOffset {
                posT = slot; posH = slot; posW = slot
            } else if slot < visOffset + nVis {
                let v = slot - visOffset
                posT = visOffset
                posH = visOffset + (v / llmW)
                posW = visOffset + (v % llmW)
            } else {
                let p = slot + ropeDelta
                posT = p; posH = p; posW = p
            }

            let base = slot * 64 * 2
            for i in 0..<64 {
                // mrope_section = [24, 20, 20] interleaved:
                // indices 1, 4, ..., 58 (i < 60 && i % 3 == 1) -> H
                // indices 2, 5, ..., 59 (i < 60 && i % 3 == 2) -> W
                // all other indices -> T
                let pVal: Int
                if i < 60 && (i % 3 == 1) {
                    pVal = posH
                } else if i < 60 && (i % 3 == 2) {
                    pVal = posW
                } else {
                    pVal = posT
                }
                let angle = Double(Float(pVal) * Float(llmInvFreq[i]))
                rPtr[base + 2 * i + 0] = Float(cos(angle))
                rPtr[base + 2 * i + 1] = Float(sin(angle))
            }
        }
    }

    @inline(__always) private func w(_ e: MTLComputeCommandEncoder, _ id: Int, _ i: Int) {
        e.setBuffer(wbuf, offset: model.offset(id), index: i)
    }

    @inline(__always) private func grid(_ e: MTLComputeCommandEncoder, _ x: Int, _ y: Int, _ threads: Int = 256) {
        e.dispatchThreadgroups(MTLSize(width: x, height: y, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    private func newEncoder() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        let cb = queue.makeCommandBufferWithUnretainedReferences()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
        return (cb, e)
    }

    // Encodes the full 24-block Vision Transformer + 3 DeepStack mergers + 1 final merger into command encoder `e`
    // using the Apple M5 Neural Accelerators (MetalPerformancePrimitives matmul2d).
    public func encodeVision(_ e: MTLComputeCommandEncoder, gridH: Int = 24, gridW: Int = 40, patchifyFromRGB: Bool = true) {
        prepareVisionGrid(gridH: gridH, gridW: gridW)
        var np = Int32(gridH * gridW)
        var nVis = Int32(np / 4)
        let npTiles = Int(np) / 32
        let visTiles = Int(nVis) / 16

        if patchifyFromRGB {
            var ghw = SIMD2<Int32>(Int32(gridH), Int32(gridW))
            e.setComputePipelineState(pVitPatchify)
            e.setBuffer(rgbBuf, offset: 0, index: 0)
            e.setBuffer(patchBuf, offset: 0, index: 1)
            e.setBytes(&ghw, length: 8, index: 2)
            grid(e, Int(np), 1, 256)
        }

        // Patch embedding + bilinear pos_embed on Neural Accelerators (MODE 4: outf = X @ W^T + bias + pos_interp)
        e.setComputePipelineState(pVitPatchEmbed)
        e.setBuffer(patchBuf, offset: 0, index: 0)
        w(e, TensorID.vitPatchW, 1)
        w(e, TensorID.vitPatchB, 2)
        e.setBuffer(vx, offset: 0, index: 3)
        e.setBuffer(vxn, offset: 0, index: 4)
        e.setBuffer(vPosInterp, offset: 0, index: 5)
        grid(e, Dims.vitHidden / 32, npTiles, 128)

        // 24 Vision Transformer blocks
        for l in 0..<Dims.vitLayers {
            // norm1 -> fp16 vxn
            e.setComputePipelineState(pVitLN1024)
            e.setBuffer(vx, offset: 0, index: 0)
            w(e, TensorID.vitLayer(l, TensorID.vNorm1W), 1)
            w(e, TensorID.vitLayer(l, TensorID.vNorm1B), 2)
            e.setBuffer(vxn, offset: 0, index: 3)
            e.setBytes(&np, length: 4, index: 4)
            grid(e, Int(np), 1, 256)

            // QKV + 2D Vision RoPE on Neural Accelerators -> head-major fp16 vq, vk, vv
            e.setComputePipelineState(pVitQkv)
            e.setBuffer(vxn, offset: 0, index: 0)
            w(e, TensorID.vitLayer(l, TensorID.vQkvW), 1)
            w(e, TensorID.vitLayer(l, TensorID.vQkvB), 2)
            e.setBuffer(vRope, offset: 0, index: 3)
            e.setBuffer(vq, offset: 0, index: 4)
            e.setBuffer(vk, offset: 0, index: 5)
            e.setBuffer(vv, offset: 0, index: 6)
            e.setBytes(&np, length: 4, index: 7)
            grid(e, 48, npTiles, 128)

            // Bidirectional full attention on Neural Accelerators -> fp16 vattn
            e.setComputePipelineState(pVitAttn)
            e.setBuffer(vq, offset: 0, index: 0)
            e.setBuffer(vk, offset: 0, index: 1)
            e.setBuffer(vv, offset: 0, index: 2)
            e.setBuffer(vattn, offset: 0, index: 3)
            e.setBuffer(vscores, offset: 0, index: 4)
            e.setBuffer(vprobs, offset: 0, index: 5)
            e.setBytes(&np, length: 4, index: 6)
            grid(e, Dims.vitHeads, Int(np) / 16, 128)

            // Output projection + residual on Neural Accelerators (MODE 1: outf += X @ W^T + bias)
            e.setComputePipelineState(pVitProj)
            e.setBuffer(vattn, offset: 0, index: 0)
            w(e, TensorID.vitLayer(l, TensorID.vProjW), 1)
            w(e, TensorID.vitLayer(l, TensorID.vProjB), 2)
            e.setBuffer(vx, offset: 0, index: 3)
            e.setBuffer(vxn, offset: 0, index: 4)
            e.setBuffer(vPosInterp, offset: 0, index: 5)
            grid(e, Dims.vitHidden / 32, npTiles, 128)

            // norm2 -> fp16 vxn
            e.setComputePipelineState(pVitLN1024)
            e.setBuffer(vx, offset: 0, index: 0)
            w(e, TensorID.vitLayer(l, TensorID.vNorm2W), 1)
            w(e, TensorID.vitLayer(l, TensorID.vNorm2B), 2)
            e.setBuffer(vxn, offset: 0, index: 3)
            e.setBytes(&np, length: 4, index: 4)
            grid(e, Int(np), 1, 256)

            // MLP fc1 (GELU tanh) on Neural Accelerators (MODE 2: outh = half(gelu_tanh(X @ W^T + bias)))
            e.setComputePipelineState(pVitFc1)
            e.setBuffer(vxn, offset: 0, index: 0)
            w(e, TensorID.vitLayer(l, TensorID.vFc1W), 1)
            w(e, TensorID.vitLayer(l, TensorID.vFc1B), 2)
            e.setBuffer(vx, offset: 0, index: 3)
            e.setBuffer(vhbuf, offset: 0, index: 4)
            e.setBuffer(vPosInterp, offset: 0, index: 5)
            grid(e, Dims.vitFF / 32, npTiles, 128)

            // MLP fc2 + residual on Neural Accelerators (MODE 1: outf += X @ W^T + bias)
            e.setComputePipelineState(pVitFc2)
            e.setBuffer(vhbuf, offset: 0, index: 0)
            w(e, TensorID.vitLayer(l, TensorID.vFc2W), 1)
            w(e, TensorID.vitLayer(l, TensorID.vFc2B), 2)
            e.setBuffer(vx, offset: 0, index: 3)
            e.setBuffer(vxn, offset: 0, index: 4)
            e.setBuffer(vPosInterp, offset: 0, index: 5)
            grid(e, Dims.vitHidden / 32, npTiles, 128)

            // DeepStack mergers at layers 5, 11, 17 (use_postshuffle_norm = True: LN over 4096)
            if let dsIdx = Dims.deepstackLayers.firstIndex(of: l) {
                e.setComputePipelineState(pVitLN4096)
                e.setBuffer(vx, offset: 0, index: 0)
                w(e, TensorID.merger(dsIdx, TensorID.mNormW), 1)
                w(e, TensorID.merger(dsIdx, TensorID.mNormB), 2)
                e.setBuffer(vxn, offset: 0, index: 3)
                e.setBytes(&nVis, length: 4, index: 4)
                grid(e, Int(nVis), 1, 256)

                e.setComputePipelineState(pMergerFc1)
                e.setBuffer(vxn, offset: 0, index: 0)
                w(e, TensorID.merger(dsIdx, TensorID.mFc1W), 1)
                w(e, TensorID.merger(dsIdx, TensorID.mFc1B), 2)
                e.setBuffer(vx, offset: 0, index: 3)
                e.setBuffer(vhbuf, offset: 0, index: 4)
                e.setBuffer(vPosInterp, offset: 0, index: 5)
                grid(e, Dims.mergerHidden / 32, visTiles, 128)

                e.setComputePipelineState(pMergerFc2)
                e.setBuffer(vhbuf, offset: 0, index: 0)
                w(e, TensorID.merger(dsIdx, TensorID.mFc2W), 1)
                w(e, TensorID.merger(dsIdx, TensorID.mFc2B), 2)
                e.setBuffer(deepstackBufs[dsIdx], offset: 0, index: 3)
                e.setBuffer(vxn, offset: 0, index: 4)
                e.setBuffer(vPosInterp, offset: 0, index: 5)
                grid(e, Dims.hidden / 32, visTiles, 128)
            }
        }

        // Final Merger after block 23 (use_postshuffle_norm = False: LN over 1024 on np patches, then view as [nVis, 4096])
        let finalIdx = Dims.numDeepstack
        e.setComputePipelineState(pVitLN1024)
        e.setBuffer(vx, offset: 0, index: 0)
        w(e, TensorID.merger(finalIdx, TensorID.mNormW), 1)
        w(e, TensorID.merger(finalIdx, TensorID.mNormB), 2)
        e.setBuffer(vxn, offset: 0, index: 3)
        e.setBytes(&np, length: 4, index: 4)
        grid(e, Int(np), 1, 256)

        e.setComputePipelineState(pMergerFc1)
        e.setBuffer(vxn, offset: 0, index: 0)
        w(e, TensorID.merger(finalIdx, TensorID.mFc1W), 1)
        w(e, TensorID.merger(finalIdx, TensorID.mFc1B), 2)
        e.setBuffer(vx, offset: 0, index: 3)
        e.setBuffer(vhbuf, offset: 0, index: 4)
        e.setBuffer(vPosInterp, offset: 0, index: 5)
        grid(e, Dims.mergerHidden / 32, visTiles, 128)

        e.setComputePipelineState(pMergerFc2)
        e.setBuffer(vhbuf, offset: 0, index: 0)
        w(e, TensorID.merger(finalIdx, TensorID.mFc2W), 1)
        w(e, TensorID.merger(finalIdx, TensorID.mFc2B), 2)
        e.setBuffer(poolerBuf, offset: 0, index: 3)
        e.setBuffer(vxn, offset: 0, index: 4)
        e.setBuffer(vPosInterp, offset: 0, index: 5)
        grid(e, Dims.hidden / 32, visTiles, 128)
    }

    // Encodes one LLM decoder layer `l` for `n` tokens.
    public func encodeLLMLayer(_ e: MTLComputeCommandEncoder, _ l: Int, n: Int, hasVis: Bool, nVis: Int = 240) {
        let kvOff = l * Dims.kvHeads * maxContext * Dims.headDim * 2
        let L = { TensorID.llmLayer(l, $0) }
        let nTiles = (n + 31) / 32

        // 1. RMSNorm + QKV + per-head q_norm/k_norm + 3D M-RoPE + KV cache write
        if n > 1 {
            e.setComputePipelineState(pRmsNormRowsH)
            e.setBuffer(x, offset: 0, index: 0)
            w(e, L(TensorID.attnNorm), 1)
            e.setBuffer(xh, offset: 0, index: 2)
            e.setBuffer(state, offset: 0, index: 3)
            grid(e, n, 1, 256)

            e.setComputePipelineState(pG3Qkv)
            e.setBuffer(xh, offset: 0, index: 0)
            w(e, L(TensorID.wqkv), 1)
            w(e, L(TensorID.wqkvSB), 2)
            e.setBuffer(qkvBuf, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            grid(e, Dims.qkvRows / 32, nTiles, 128)

            e.setComputePipelineState(pQkNormRope)
            e.setBuffer(qkvBuf, offset: 0, index: 0)
            w(e, L(TensorID.qNorm), 1)
            w(e, L(TensorID.kNorm), 2)
            e.setBuffer(q, offset: 0, index: 3)
            e.setBuffer(kc, offset: kvOff, index: 4)
            e.setBuffer(vc, offset: kvOff, index: 5)
            e.setBuffer(rope, offset: 0, index: 6)
            e.setBuffer(state, offset: 0, index: 7)
            e.setBuffer(vq, offset: 0, index: 8)
            grid(e, 32, n, 128)
        } else {
            e.setComputePipelineState(pAttnIn)
            w(e, L(TensorID.wqkv), 0)
            w(e, L(TensorID.wqkvSB), 1)
            w(e, L(TensorID.attnNorm), 2)
            w(e, L(TensorID.qNorm), 3)
            w(e, L(TensorID.kNorm), 4)
            e.setBuffer(x, offset: 0, index: 5)
            e.setBuffer(q, offset: 0, index: 6)
            e.setBuffer(kc, offset: kvOff, index: 7)
            e.setBuffer(vc, offset: kvOff, index: 8)
            e.setBuffer(rope, offset: 0, index: 9)
            e.setBuffer(state, offset: 0, index: 10)
            grid(e, 32, n, 128)
        }

        // Last-layer prefill optimization: only the final token's hidden state is needed by lm_head
        let isLastPrefill = (l == Dims.layers - 1 && n > 1)
        let activeRows = isLastPrefill ? 1 : n
        var toff = Int32(isLastPrefill ? (n - 1) : 0)

        // 2. Causal GQA attention
        if activeRows > 1 {
            e.setComputePipelineState(pPrefillAttn)
            e.setBuffer(vq, offset: 0, index: 0)
            e.setBuffer(kc, offset: kvOff, index: 1)
            e.setBuffer(vc, offset: kvOff, index: 2)
            e.setBuffer(attn, offset: 0, index: 3)
            e.setBuffer(attnh, offset: 0, index: 4)
            e.setBuffer(vscores, offset: 0, index: 5)
            e.setBuffer(vprobs, offset: 0, index: 6)
            e.setBuffer(state, offset: 0, index: 7)
            grid(e, Dims.heads, (activeRows + 15) / 16, 128)
        } else {
            e.setComputePipelineState(pAttention)
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(kc, offset: kvOff, index: 1)
            e.setBuffer(vc, offset: kvOff, index: 2)
            e.setBuffer(attn, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            e.setBytes(&toff, length: 4, index: 5)
            e.setBuffer(attnh, offset: 0, index: 6)
            grid(e, Dims.heads, activeRows, 256)
        }

        if activeRows > 1 {
            // 3. Output projection + residual on Neural Accelerators
            e.setComputePipelineState(pG3O)
            e.setBuffer(attnh, offset: 0, index: 0)
            w(e, L(TensorID.wo), 1)
            w(e, L(TensorID.woSB), 2)
            e.setBuffer(x, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            grid(e, Dims.hidden / 32, nTiles, 128)

            // 4. MLP RMSNorm + gate/up + SwiGLU on Neural Accelerators
            e.setComputePipelineState(pRmsNormRowsH)
            e.setBuffer(x, offset: 0, index: 0)
            w(e, L(TensorID.mlpNorm), 1)
            e.setBuffer(xh, offset: 0, index: 2)
            e.setBuffer(state, offset: 0, index: 3)
            grid(e, activeRows, 1, 256)

            e.setComputePipelineState(pG3Gu)
            e.setBuffer(xh, offset: 0, index: 0)
            w(e, L(TensorID.wgu), 1)
            w(e, L(TensorID.wguSB), 2)
            e.setBuffer(hbufh, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            grid(e, Dims.ff / 32, nTiles, 128)

            // 5. MLP down + residual on Neural Accelerators
            e.setComputePipelineState(pG3Down)
            e.setBuffer(hbufh, offset: 0, index: 0)
            w(e, L(TensorID.wd), 1)
            w(e, L(TensorID.wdSB), 2)
            e.setBuffer(x, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            grid(e, Dims.hidden / 32, nTiles, 128)
        } else {
            // 3. Output projection + residual (1-token decode or last-layer final token)
            e.setComputePipelineState(pOProj)
            w(e, L(TensorID.wo), 0)
            w(e, L(TensorID.woSB), 1)
            e.setBuffer(attn, offset: 0, index: 2)
            e.setBuffer(x, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            e.setBytes(&toff, length: 4, index: 5)
            grid(e, Dims.hidden / 16, activeRows, 64)

            // 4. MLP gate/up + SwiGLU
            e.setComputePipelineState(pMlpIn)
            w(e, L(TensorID.wgu), 0)
            w(e, L(TensorID.wguSB), 1)
            w(e, L(TensorID.mlpNorm), 2)
            e.setBuffer(x, offset: 0, index: 3)
            e.setBuffer(hbuf, offset: 0, index: 4)
            e.setBuffer(state, offset: 0, index: 5)
            e.setBytes(&toff, length: 4, index: 6)
            grid(e, Dims.ff / 8, activeRows, 64)

            // 5. MLP down + residual
            e.setComputePipelineState(pMlpOut)
            w(e, L(TensorID.wd), 0)
            w(e, L(TensorID.wdSB), 1)
            e.setBuffer(hbuf, offset: 0, index: 2)
            e.setBuffer(x, offset: 0, index: 3)
            e.setBuffer(state, offset: 0, index: 4)
            e.setBytes(&toff, length: 4, index: 5)
            grid(e, Dims.hidden / 16, activeRows, 64)
        }

        // 6. DeepStack feature addition at layers 0, 1, 2 during prefill
        if hasVis && l < Dims.numDeepstack {
            e.setComputePipelineState(pDeepstackAdd)
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(deepstackBufs[l], offset: 0, index: 1)
            e.setBuffer(state, offset: 0, index: 2)
            grid(e, nVis, 1, 256)
        }
    }

    public func encodeLMHead(_ e: MTLComputeCommandEncoder, writeLogits: Bool) {
        e.setComputePipelineState(pLMHead)
        w(e, TensorID.embed, 0)
        w(e, TensorID.embedSB, 1)
        w(e, TensorID.finalNorm, 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(pval, offset: 0, index: 4)
        e.setBuffer(pidx, offset: 0, index: 5)
        e.setBuffer(state, offset: 0, index: 6)
        e.setBuffer(logits, offset: 0, index: 7)
        var wl: Int32 = writeLogits ? 1 : 0
        e.setBytes(&wl, length: 4, index: 8)
        grid(e, Dims.vocab / 32, 1, 128)
    }

    public func encodeCommit(_ e: MTLComputeCommandEncoder) {
        e.setComputePipelineState(pCommit)
        e.setBuffer(pval, offset: 0, index: 0)
        e.setBuffer(pidx, offset: 0, index: 1)
        e.setBuffer(state, offset: 0, index: 2)
        e.setBuffer(outToks, offset: 0, index: 3)
        w(e, TensorID.embed, 4)
        w(e, TensorID.embedSB, 5)
        e.setBuffer(x, offset: 0, index: 6)
        grid(e, 1, 1, 1024)
    }

    public func encodePrefill(_ e: MTLComputeCommandEncoder, n: Int, hasVis: Bool, nVis: Int, writeLogits: Bool) {
        e.setComputePipelineState(pEmbedPrompt)
        w(e, TensorID.embed, 0)
        w(e, TensorID.embedSB, 1)
        e.setBuffer(promptToks, offset: 0, index: 2)
        e.setBuffer(poolerBuf, offset: 0, index: 3)
        e.setBuffer(x, offset: 0, index: 4)
        e.setBuffer(state, offset: 0, index: 5)
        grid(e, n, 1, 256)

        for l in 0..<Dims.layers {
            encodeLLMLayer(e, l, n: n, hasVis: hasVis, nVis: nVis)
        }
        encodeLMHead(e, writeLogits: writeLogits)
        encodeCommit(e)
    }

    public func encodeDecodeStep(_ e: MTLComputeCommandEncoder, writeLogits: Bool = false) {
        for l in 0..<Dims.layers {
            encodeLLMLayer(e, l, n: 1, hasVis: false, nVis: 0)
        }
        encodeLMHead(e, writeLogits: writeLogits)
        encodeCommit(e)
    }

    // Runs full multimodal inference (Vision + Prefill + GPU-resident greedy decode) on an RGB8 image [H, W, 3].
    public func generateVision(rgbBytes: UnsafeRawPointer, width: Int = 640, height: Int = 384,
                               tokens: [Int32], maxNewTokens: Int = 64, writeFirstLogits: Bool = false) -> [Int32] {
        let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let gridH = height / Dims.vitPatchSize
        let gridW = width / Dims.vitPatchSize
        let np = gridH * gridW
        let nVis = np / 4
        let llmH = gridH / Dims.vitMergeSize
        let llmW = gridW / Dims.vitMergeSize

        // Copy RGB bytes into shared GPU buffer
        memcpy(rgbBuf.contents(), rgbBytes, width * height * 3)

        // Find vision start offset (first IMAGE_PAD_ID = 151655)
        let visOffset = tokens.firstIndex(of: 151655) ?? -1
        prepareMRoPE(visOffset: visOffset, llmH: llmH, llmW: llmW)

        // Initialize state
        let n = tokens.count
        let pt = promptToks.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<n { pt[i] = tokens[i] }

        let s = state.contents().assumingMemoryBound(to: Int32.self)
        s[0] = 0 // pos
        s[1] = Int32(n) // n
        s[2] = 0 // step
        s[3] = 0 // done
        s[4] = Int32(maxContext)
        s[5] = Int32(min(maxNewTokens, maxSteps))
        s[6] = eos.0
        s[7] = eos.1
        s[8] = Int32(visOffset)
        s[9] = Int32(nVis)
        s[10] = Int32(max(llmH, llmW) - nVis)
        s[11] = Int32(np)

        // 1. Vision command buffer
        let (cbVis, eVis) = newEncoder()
        encodeVision(eVis, gridH: gridH, gridW: gridW, patchifyFromRGB: true)
        eVis.endEncoding()
        cbVis.commit()

        // 2. Prefill command buffer
        let (cbPre, ePre) = newEncoder()
        encodePrefill(ePre, n: n, hasVis: visOffset >= 0, nVis: nVis, writeLogits: writeFirstLogits)
        ePre.endEncoding()
        cbPre.commit()

        // 3. Pipeline decode steps on GPU
        let batchSize = 4
        var submittedSteps = 1 // prefill commits step 1
        var decodeCBs: [MTLCommandBuffer] = []
        while submittedSteps < maxNewTokens {
            let count = min(batchSize, maxNewTokens - submittedSteps)
            let (cbDec, eDec) = newEncoder()
            for _ in 0..<count {
                encodeDecodeStep(eDec, writeLogits: false)
            }
            eDec.endEncoding()
            cbDec.commit()
            decodeCBs.append(cbDec)
            submittedSteps += count
            if s[3] != 0 { break }
        }

        cbVis.waitUntilCompleted()
        cbPre.waitUntilCompleted()
        for cb in decodeCBs { cb.waitUntilCompleted() }

        let tEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let visMs = (cbVis.gpuEndTime - cbVis.gpuStartTime) * 1e3
        let preMs = (cbPre.gpuEndTime - cbPre.gpuStartTime) * 1e3
        var decMs = 0.0
        for cb in decodeCBs {
            if cb.gpuEndTime > cb.gpuStartTime {
                decMs += (cb.gpuEndTime - cb.gpuStartTime) * 1e3
            }
        }

        let genCount = Int(s[2])
        let outPtr = outToks.contents().assumingMemoryBound(to: Int32.self)
        var result: [Int32] = []
        result.reserveCapacity(genCount)
        for i in 0..<genCount {
            let tok = outPtr[i]
            if tok == eos.0 || tok == eos.1 { break }
            result.append(tok)
        }

        lastTiming = Timing(
            visionMs: visMs,
            prefillMs: preMs,
            decodeMs: decMs,
            totalMs: Double(tEnd - t0) / 1e6,
            prefillTokens: n,
            generatedTokens: result.count
        )
        return result
    }
}
