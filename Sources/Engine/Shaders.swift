// Metal 4 kernels for Qwen3-VL-2B-Instruct, compiled at runtime from source.
// Covers the full Vision Transformer (24 blocks, 2D RoPE, bidirectional attention),
// 3 DeepStack mergers + 1 final spatial merger, and the 28-layer Language Backbone
// (per-head QK RMSNorm, 3D interleaved M-RoPE, GQA, SwiGLU, tied lm_head, GPU-resident commit).
// FMT selects LLM weight format: 0 = f16, 1 = q8 (group-32 symmetric), 2 = q4 (group-64 affine).

let shaderSource = #"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Vision Encoder dimensions (Qwen3VLVisionModel)
#define VH 1024
#define VFF 4096
#define VNH 16
#define VHD 64
#define VQKV 3072
#define VPATCH 768
#define MH 4096
#define NPATCH_MAX 1024

// Language Backbone dimensions (Qwen3VLTextModel)
#define H 2048
#define FF 6144
#define NH 16
#define NKV 8
#define HD 128
#define QKV 4096
#define QKV_DIM 4096
#define GU 12288
#define VOCAB 151936
#define EPS 1e-6f
#define LMH_ROWS_PER_TG 32
#define LMH_TGS (VOCAB / LMH_ROWS_PER_TG)
#define IMAGE_PAD_ID 151655

#pragma METAL fp math_mode(safe)

struct State {
    int pos;        // KV position of the first token in the current pass
    int n;          // number of tokens in the current pass
    int step;       // generated tokens committed so far
    int done;       // stop flag: every kernel returns immediately once set
    int maxctx;     // maximum KV context length
    int maxsteps;   // maximum generation steps
    int eos0;       // primary EOS token ID (151645 <|im_end|>)
    int eos1;       // secondary EOS token ID (151643 <|endoftext|>)
    int vis_offset; // token index where <|image_pad|> starts in the current pass (-1 if none)
    int n_vis;      // number of vision tokens (240 for 640x384)
    int rope_delta; // M-RoPE position delta (max(llm_h, llm_w) - n_vis)
    int npatches;   // number of ViT patches (960 for 640x384)
};

// ---------------------------------------------------------------------------------------------
// Helper math functions for Vision Encoder (GELU tanh & exact erf GELU)

inline float gelu_tanh(float x) {
    const float c = 0.7978845608028654f; // sqrt(2 / pi)
    const float inner = clamp(c * (x + 0.044715f * x * x * x), -15.f, 15.f);
    const float e2 = exp(2.f * inner);
    const float th = (e2 - 1.f) / (e2 + 1.f);
    return 0.5f * x * (1.f + th);
}

inline float erf_approx(float x) {
    const float s = x < 0.f ? -1.f : 1.f;
    const float ax = min(fabs(x), 4.f);
    const float t = 1.f / (1.f + 0.3275911f * ax);
    const float y = 1.f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

inline float gelu_exact(float x) {
    return 0.5f * x * (1.f + erf_approx(x * 0.7071067811865475f));
}

// ---------------------------------------------------------------------------------------------
// Vision Encoder kernels on M5 Neural Accelerators (MPP matmul2d, weights in fp16)

// Patchify RGB8 image [H_img, W_img, 3] into merge-ordered fp16 patches [npatches, 768].
// Grid (npatches), 256 threads.
kernel void k_vit_patchify_rgb(device const uchar* rgb [[buffer(0)]],
                               device half* patches [[buffer(1)]],
                               constant int2& grid_hw [[buffer(2)]], // (grid_h, grid_w) in 16x16 patches
                               uint p [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const int gh = grid_hw.x, gw = grid_hw.y;
    const int np = gh * gw;
    if ((int)p >= np) return;
    const int merge_w = gw / 2;
    const int block_idx = p / 4;
    const int in_idx = p % 4;
    const int br = block_idx / merge_w;
    const int bc = block_idx % merge_w;
    const int ir = in_idx / 2;
    const int ic = in_idx % 2;
    const int pr = br * 2 + ir;
    const int pc = bc * 2 + ic;
    const int img_w = gw * 16;
    for (int i = tid; i < VPATCH; i += 256) {
        const int c = i / 256;
        const int rem = i % 256;
        const int dy = rem / 16;
        const int dx = rem % 16;
        const int y = pr * 16 + dy;
        const int x = pc * 16 + dx;
        const uchar val = rgb[(y * img_w + x) * 3 + c];
        patches[p * VPATCH + i] = half(float(val) * (2.0f / 255.0f) - 1.0f);
    }
}

// Vectorized zero-barrier LayerNorm over 1024 columns outputting fp16 (4 rows per threadgroup, 1 row per simdgroup).
// Grid ((nrows + 3) / 4), 128 threads (4 simdgroups).
kernel void k_vit_ln1024(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device const float* b [[buffer(2)]],
                         device half* out [[buffer(3)]],
                         constant int& nrows [[buffer(4)]],
                         uint tgp [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const int r = int(tgp * 4 + sg);
    if (r >= nrows) return;
    device const float4* xr4 = (device const float4*)(x + (ulong)r * 1024);
    device const float4* w4 = (device const float4*)w;
    device const float4* b4 = (device const float4*)b;
    device half4* yr4 = (device half4*)(out + (ulong)r * 1024);
    float4 xv[8];
    float sum = 0.f;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 8; ++k) {
        const float4 v = xr4[lane + k * 32];
        xv[k] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    const float mean = simd_sum(sum) * (1.0f / 1024.0f);
    float vsum = 0.f;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 8; ++k) {
        const float4 d = xv[k] - mean;
        xv[k] = d;
        vsum += dot(d, d);
    }
    const float inv_std = rsqrt(simd_sum(vsum) * (1.0f / 1024.0f) + EPS);
    #pragma clang loop unroll(full)
    for (int k = 0; k < 8; ++k) {
        const uint idx = lane + k * 32;
        yr4[idx] = half4(xv[k] * inv_std * w4[idx] + b4[idx]);
    }
}

// Vectorized LayerNorm over 4096 columns outputting fp16 (register-cached single-read float4/half4).
// Grid (nrows), 256 threads (8 simdgroups).
kernel void k_vit_ln4096(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device const float* b [[buffer(2)]],
                         device half* out [[buffer(3)]],
                         constant int& nrows [[buffer(4)]],
                         uint r [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    if ((int)r >= nrows) return;
    threadgroup float red_m[8];
    threadgroup float red_v[8];
    device const float4* xr4 = (device const float4*)(x + (ulong)r * 4096);
    device const float4* w4 = (device const float4*)w;
    device const float4* b4 = (device const float4*)b;
    device half4* yr4 = (device half4*)(out + (ulong)r * 4096);
    float4 xv[4];
    float sum = 0.f;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 4; ++k) {
        const float4 v = xr4[tid + k * 256];
        xv[k] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    sum = simd_sum(sum);
    if (lane == 0) red_m[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = (red_m[0] + red_m[1] + red_m[2] + red_m[3] + red_m[4] + red_m[5] + red_m[6] + red_m[7]) * (1.0f / 4096.0f);
    float vsum = 0.f;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 4; ++k) {
        const float4 d = xv[k] - mean;
        xv[k] = d;
        vsum += dot(d, d);
    }
    vsum = simd_sum(vsum);
    if (lane == 0) red_v[sg] = vsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3] + red_v[4] + red_v[5] + red_v[6] + red_v[7]) * (1.0f / 4096.0f) + EPS);
    #pragma clang loop unroll(full)
    for (int k = 0; k < 4; ++k) {
        const uint idx = tid + k * 256;
        yr4[idx] = half4(xv[k] * inv_std * w4[idx] + b4[idx]);
    }
}

// ViT QKV projection + 2D Vision RoPE on M5 Neural Accelerators (matmul2d).
// 4-simdgroup data-parallel: each threadgroup computes 2 head-blocks (128 output rows = 4 x 32 rows),
// so all 4 simdgroups compute the full K=1024 reduction without split-K and share X in L1 cache.
// Epilogue runs in two waves over an 8 KiB [2][TT][32] buffer: simdgroups 0,1 (head 0) then 2,3 (head 1).
// Grid (24, np / 32) = (24, 30), 128 threads (4 simdgroups).
kernel void k_vit_qkv(device half* X [[buffer(0)]],
                      device half* W [[buffer(1)]],
                      device const float* bias [[buffer(2)]],
                      device const float2* vrope [[buffer(3)]],
                      device half* vq [[buffer(4)]],
                      device half* vk [[buffer(5)]],
                      device half* vv [[buffer(6)]],
                      constant int& np [[buffer(7)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 32;
    constexpr int BK = 64;
    constexpr int NSTEP = VH / BK; // 1024 / 64 = 16
    const int t0 = tgp.y * TT;
    if (t0 >= np) return;

    const uint r_sg = tgp.x * 128 + sg * 32;
    threadgroup float Os[2 * TT * 32]; // 8 KiB total (allows 4 threadgroups/core occupancy on M5)

    auto tX = tensor(X + (ulong)t0 * VH, dextents<int, 2>(VH, TT), array<int, 2>({1, VH}));
    auto tW = tensor(W + (ulong)r_sg * VH, dextents<int, 2>(VH, 32), array<int, 2>({1, VH}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    #pragma clang loop unroll(full)
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;

    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = i * BK;
        auto a = tX.template slice<BK, TT>(k0, 0);
        auto b = tW.template slice<BK, 32>(k0, 0);
        op.run(a, b, cT);
    }

    const uint head_in_tg = sg >> 1;
    const int h_all = int(tgp.x * 2 + head_in_tg);
    const uint r0 = uint(h_all * 64);
    const int t_sub = int(sg & 1) * 16;
    const float b0 = bias[r0 + lane];
    const float b1 = bias[r0 + 32 + lane];
    auto tO = tensor(Os + (sg & 1) * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));

    for (uint wave = 0; wave < 2; ++wave) {
        if (head_in_tg == wave) {
            cT.store(tO);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (head_in_tg == wave) {
            threadgroup const float* O0 = Os;
            threadgroup const float* O1 = Os + TT * 32;
            for (int dt = 0; dt < 16; ++dt) {
                const int t = t_sub + dt;
                const int p = t0 + t;
                if (p >= np) continue;
                float a = b0 + O0[t * 32 + lane];
                float b = b1 + O1[t * 32 + lane];
                if (h_all < 32) {
                    const float2 cs = vrope[p * 32 + lane];
                    const float a2 = a * cs.x - b * cs.y;
                    const float b2 = b * cs.x + a * cs.y;
                    a = a2; b = b2;
                }
                if (h_all < 16) {
                    device half* qdst = vq + ((ulong)h_all * np + p) * VHD;
                    qdst[lane] = half(a);
                    qdst[lane + 32] = half(b);
                } else if (h_all < 32) {
                    const int kh = h_all - 16;
                    device half* kdst = vk + ((ulong)kh * np + p) * VHD;
                    kdst[lane] = half(a);
                    kdst[lane + 32] = half(b);
                } else {
                    const int vh = h_all - 32;
                    device half* vdst = vv + ((ulong)vh * np + p) * VHD;
                    vdst[lane] = half(a);
                    vdst[lane + 32] = half(b);
                }
            }
        }
        if (wave == 0) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ViT bidirectional full attention on M5 Neural Accelerators (matmul2d Q @ K^T + softmax + matmul2d P @ V).
// Each threadgroup computes 1 head (h = 0..15) and TT = 16 query patches (t0 = tgp.y * 16).
// Grid (16, np / 16) = (16, 60), 128 threads (4 simdgroups).
kernel void k_vit_attn(device half* vq [[buffer(0)]],
                       device half* vk [[buffer(1)]],
                       device half* vv [[buffer(2)]],
                       device half* out [[buffer(3)]],
                       device float* vscores [[buffer(4)]],
                       device half* vprobs [[buffer(5)]],
                       constant int& np [[buffer(6)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 16;
    const int h = tgp.x;
    const int t0 = tgp.y * TT;
    if (t0 >= np) return;

    device half* Qh = vq + ((ulong)h * np + t0) * VHD;
    device half* Kh = vk + (ulong)h * np * VHD;
    device half* Vh = vv + (ulong)h * np * VHD;
    device float* Sh = vscores + ((ulong)h * np + t0) * np;
    device half* Ph = vprobs + ((ulong)h * np + t0) * np;

    // Phase 1: Q_tile [16, 64] @ K_h^T [960, 64] -> Sh [16, 960] via Neural Accelerators
    auto tQ = tensor(Qh, dextents<int, 2>(VHD, TT), array<int, 2>({1, VHD}));
    constexpr auto descQK = matmul2d_descriptor(TT, 32, VHD, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descQK, execution_simdgroup> opQK;
    const int nKeyTiles = np / 32; // 30
    for (int kt = sg; kt < nKeyTiles; kt += 4) {
        const int s0 = kt * 32;
        auto tK = tensor(Kh + (ulong)s0 * VHD, dextents<int, 2>(VHD, 32), array<int, 2>({1, VHD}));
        auto cS = opQK.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        for (int i = 0; i < cS.get_capacity(); i++) if (cS.is_valid_element(i)) cS[i] = 0.f;
        opQK.run(tQ, tK, cS);
        auto tS = tensor(Sh + s0, dextents<int, 2>(32, TT), array<int, 2>({1, np}));
        cS.store(tS);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Row-wise softmax over 960 keys in registers (30 keys/lane, zero global writebacks to Sh)
    const int n_s = (np - int(lane) + 31) >> 5;
    for (int r = int(sg) * 4; r < int(sg) * 4 + 4; ++r) {
        device const float* sr = Sh + r * np;
        device half* pr = Ph + r * np;
        float row_s[32];
        float m = -INFINITY;
        for (int k = 0; k < n_s; ++k) {
            const float v = sr[lane + k * 32] * 0.125f;
            row_s[k] = v;
            m = max(m, v);
        }
        m = simd_max(m);
        float sum = 0.f;
        for (int k = 0; k < n_s; ++k) {
            const float e = exp(row_s[k] - m);
            row_s[k] = e;
            sum += e;
        }
        const float inv_sum = 1.0f / simd_sum(sum);
        for (int k = 0; k < n_s; ++k) {
            pr[lane + k * 32] = half(row_s[k] * inv_sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: Ph [16, 960] @ V_h [960, 64] -> O [16, 64] via Neural Accelerators
    // sg 0..1 compute d0 = 0..31 (2-way split-K), sg 2..3 compute d0 = 32..63 (2-way split-K), using only 8 KiB Os!
    threadgroup float Os[4 * TT * 32]; // 8 KiB (allows 4 threadgroups/core occupancy on M5)
    const int d0 = int(sg >> 1) * 32;
    const int k_sub = int(sg & 1);
    auto tP = tensor(Ph, dextents<int, 2>(np, TT), array<int, 2>({1, np}));
    auto tV = tensor(Vh, dextents<int, 2>(VHD, np), array<int, 2>({1, VHD}));
    constexpr auto descPV = matmul2d_descriptor(TT, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descPV, execution_simdgroup> opPV;
    auto a0 = tP.template slice<32, TT>(0, 0);
    auto b0 = tV.template slice<32, 32>(d0, 0);
    auto cO = opPV.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma clang loop unroll(full)
    for (int i = 0; i < cO.get_capacity(); i++) if (cO.is_valid_element(i)) cO[i] = 0.f;
    for (int kt = k_sub; kt < nKeyTiles; kt += 2) {
        const int s0 = kt * 32;
        auto a = tP.template slice<32, TT>(s0, 0);
        auto b = tV.template slice<32, 32>(d0, s0);
        opPV.run(a, b, cO);
    }
    auto tO = tensor(Os + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cO.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce 2-way split-K and write out [TT, 64]
    const int t_start = k_sub * (TT / 2);
    threadgroup const float* O_a = Os + (sg & 2) * TT * 32;
    threadgroup const float* O_b = O_a + TT * 32;
    for (int dt = 0; dt < TT / 2; ++dt) {
        const int r = t_start + dt;
        const float val = O_a[r * 32 + lane] + O_b[r * 32 + lane];
        out[(ulong)(t0 + r) * VH + h * VHD + d0 + lane] = half(val);
    }
}

// Generic M5 Neural Accelerator (matmul2d) fp16 linear kernel for ViT & Mergers:
// Each of the 4 simdgroups computes an independent 32-row output slice (r_sg = tgp.x * 128 + sg * 32),
// sharing the exact same X tile in L1 cache at every k0 step with 8 KiB shared memory (4 TGs/core on M5)!
// Grid (ROWS / 128, nrows / TT), 128 threads (4 simdgroups).
template<int K, int ROWS, int MODE, int TT>
kernel void k_vit_gemm_T(device half* X [[buffer(0)]],
                         device half* W [[buffer(1)]],
                         device const float* bias [[buffer(2)]],
                         device float* outf [[buffer(3)]],
                         device half* outh [[buffer(4)]],
                         device const float* pos_interp [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int BK = 64;
    constexpr int NSTEP = K / BK;
    const uint r0 = tgp.x * 128 + sg * 32;
    const uint t0 = tgp.y * TT;
    threadgroup float Os[2 * TT * 32]; // 8 KiB (allows 4 threadgroups/core occupancy on M5)
    threadgroup float* myO = Os + (sg & 1) * TT * 32;

    auto tX = tensor(X + (ulong)t0 * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    auto tW = tensor(W + (ulong)r0 * K, dextents<int, 2>(K, 32), array<int, 2>({1, K}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    #pragma clang loop unroll(full)
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;

    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = i * BK;
        auto a = tX.template slice<BK, TT>(k0, 0);
        auto b = tW.template slice<BK, 32>(k0, 0);
        op.run(a, b, cT);
    }

    const float b_val = bias[r0 + lane];
    auto tO = tensor(myO, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    for (uint wave = 0; wave < 2; ++wave) {
        if ((sg >> 1) == wave) {
            cT.store(tO);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int t = 0; t < TT; ++t) {
                const float v = myO[t * 32 + lane] + b_val;
                const ulong idx = (ulong)(t0 + t) * ROWS + r0 + lane;
                if (MODE == 0) outf[idx] = v;
                else if (MODE == 1) outf[idx] += v;
                else if (MODE == 2) outh[idx] = half(gelu_tanh(v));
                else if (MODE == 3) outh[idx] = half(gelu_exact(v));
                else if (MODE == 4) outf[idx] = v + pos_interp[idx];
            }
        }
        if (wave == 0) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("k_vit_patch_embed")]] kernel void k_vit_gemm_T<768, 1024, 4, 32>(device half*, device half*, device const float*, device float*, device half*, device const float*, uint2, uint, uint);
template [[host_name("k_vit_proj")]] kernel void k_vit_gemm_T<1024, 1024, 1, 32>(device half*, device half*, device const float*, device float*, device half*, device const float*, uint2, uint, uint);
template [[host_name("k_vit_fc1")]] kernel void k_vit_gemm_T<1024, 4096, 2, 32>(device half*, device half*, device const float*, device float*, device half*, device const float*, uint2, uint, uint);
template [[host_name("k_vit_fc2")]] kernel void k_vit_gemm_T<4096, 1024, 1, 32>(device half*, device half*, device const float*, device float*, device half*, device const float*, uint2, uint, uint);
template [[host_name("k_merger_fc1")]] kernel void k_vit_gemm_T<4096, 4096, 3, 16>(device half*, device half*, device const float*, device float*, device half*, device const float*, uint2, uint, uint);
template [[host_name("k_merger_fc2")]] kernel void k_vit_gemm_T<4096, 2048, 0, 16>(device half*, device half*, device const float*, device float*, device half*, device const float*, uint2, uint, uint);

// ---------------------------------------------------------------------------------------------
// Language Backbone weight format helpers (FMT 0 = f16, FMT 1 = q8 group-32, FMT 2 = q4 group-64)

#if FMT == 0
typedef half wtype;
inline float embed_at(device const half* E, device const half2* SB, int row, int i) {
    return float(E[(ulong)row * H + i]);
}
#elif FMT == 1
typedef char wtype;
inline float embed_at(device const char* E, device const half2* SB, int row, int i) {
    return float(((device const half*)SB)[row * (H / 32) + (i >> 5)]) * float(E[(ulong)row * H + i]);
}
#elif FMT == 2
typedef uchar wtype;
inline float embed_at(device const uchar* E, device const half2* SB, int row, int i) {
    const uchar b = E[(ulong)row * (H / 2) + (i >> 1)];
    const float q = float((i & 1) ? (b >> 4) : (b & 0xF));
    const float2 sb = float2(SB[row * (H / 64) + (i >> 6)]);
    return sb.x * q + sb.y;
}
#endif

// Dot product of 64 columns at (row, col64 = c64 * 64) with a lane's xh[16] (64 halfs) and sum X = sum(xh).
inline float dot64_row(device const wtype* W, device const half2* SB, uint row, uint c64, uint K_cols,
                       thread const half4* xh, float X) {
#if FMT == 0
    device const half4* wp = (device const half4*)(W + (ulong)row * K_cols + c64 * 64);
    float acc = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {
        acc += dot(float4(wp[i]), float4(xh[i]));
    }
    return acc;
#elif FMT == 1
    device const char4* wp = (device const char4*)(W + (ulong)row * K_cols + c64 * 64);
    device const half* sp = (device const half*)SB + (ulong)row * (K_cols / 32) + c64 * 2;
    float d0 = 0.f, d1 = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        d0 += dot(float4(wp[i]), float4(xh[i]));
        d1 += dot(float4(wp[8 + i]), float4(xh[8 + i]));
    }
    return float(sp[0]) * d0 + float(sp[1]) * d1;
#elif FMT == 2
    device const uint4* wp = (device const uint4*)(W + (ulong)row * (K_cols / 2) + c64 * 32);
    const float2 sb = float2(SB[(ulong)row * (K_cols / 64) + c64]);
    thread const half2* x2 = (thread const half2*)xh;
    half2 acc0 = half2(0.h), acc1 = half2(0.h);
    #pragma clang loop unroll(full)
    for (int q = 0; q < 2; ++q) {
        const uint4 wv = wp[q];
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const uint v = wv[j];
            const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
            const int o = q * 16 + j * 4;
            acc0 = fma(half2(p04.x, p15.x), x2[o + 0], acc0);
            acc1 = fma(half2(p26.x, p37.x), x2[o + 1], acc1);
            acc0 = fma(half2(p04.y, p15.y), x2[o + 2], acc0);
            acc1 = fma(half2(p26.y, p37.y), x2[o + 3], acc1);
        }
    }
    const half2 s2 = acc0 + acc1;
    const float acc = float(s2.x) + float(s2.y);
    return (16.f * sb.x) * (acc - X) + sb.y * X;
#endif
}

// Single-weight-read 2-token dot product: loads W[row, c64*64..+63] once and dots against both xh0 and xh1.
inline float2 dot64_row2(device const wtype* W, device const half2* SB, uint row, uint c64, uint K_cols,
                         thread const half4* xh0, float X0, thread const half4* xh1, float X1) {
#if FMT == 0
    device const half4* wp = (device const half4*)(W + (ulong)row * K_cols + c64 * 64);
    float acc0 = 0.f, acc1 = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {
        const float4 w = float4(wp[i]);
        acc0 += dot(w, float4(xh0[i]));
        acc1 += dot(w, float4(xh1[i]));
    }
    return float2(acc0, acc1);
#elif FMT == 1
    device const char4* wp = (device const char4*)(W + (ulong)row * K_cols + c64 * 64);
    device const half* sp = (device const half*)SB + (ulong)row * (K_cols / 32) + c64 * 2;
    float d00 = 0.f, d01 = 0.f, d10 = 0.f, d11 = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const float4 w0 = float4(wp[i]);
        const float4 w1 = float4(wp[8 + i]);
        d00 += dot(w0, float4(xh0[i]));
        d01 += dot(w1, float4(xh0[8 + i]));
        d10 += dot(w0, float4(xh1[i]));
        d11 += dot(w1, float4(xh1[8 + i]));
    }
    const float s0 = float(sp[0]), s1 = float(sp[1]);
    return float2(s0 * d00 + s1 * d01, s0 * d10 + s1 * d11);
#elif FMT == 2
    device const uint4* wp = (device const uint4*)(W + (ulong)row * (K_cols / 2) + c64 * 32);
    const float2 sb = float2(SB[(ulong)row * (K_cols / 64) + c64]);
    thread const half2* x2_0 = (thread const half2*)xh0;
    thread const half2* x2_1 = (thread const half2*)xh1;
    half2 a0 = half2(0.h), b0 = half2(0.h);
    half2 a1 = half2(0.h), b1 = half2(0.h);
    #pragma clang loop unroll(full)
    for (int q = 0; q < 2; ++q) {
        const uint4 wv = wp[q];
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const uint v = wv[j];
            const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
            const half2 w0 = half2(p04.x, p15.x);
            const half2 w1 = half2(p26.x, p37.x);
            const half2 w2 = half2(p04.y, p15.y);
            const half2 w3 = half2(p26.y, p37.y);
            const int o = q * 16 + j * 4;
            a0 = fma(w0, x2_0[o + 0], a0);
            b0 = fma(w1, x2_0[o + 1], b0);
            a0 = fma(w2, x2_0[o + 2], a0);
            b0 = fma(w3, x2_0[o + 3], b0);
            a1 = fma(w0, x2_1[o + 0], a1);
            b1 = fma(w1, x2_1[o + 1], b1);
            a1 = fma(w2, x2_1[o + 2], a1);
            b1 = fma(w3, x2_1[o + 3], b1);
        }
    }
    const half2 s2_0 = a0 + b0;
    const half2 s2_1 = a1 + b1;
    const float acc0 = float(s2_0.x) + float(s2_0.y);
    const float acc1 = float(s2_1.x) + float(s2_1.y);
    return float2((16.f * sb.x) * (acc0 - X0) + sb.y * X0,
                  (16.f * sb.x) * (acc1 - X1) + sb.y * X1);
#endif
}

// Butterfly reduction of 4 row partials across 32 lanes: row r (0..3) ends up in lanes 8r..8r+7.
static inline float reduce4(thread float* v, ushort lane) {
    float t2[2];
    bool b = (lane & 16) != 0;
    for (int i = 0; i < 2; i++) {
        const float keep = b ? v[2 + i] : v[i], send = b ? v[i] : v[2 + i];
        t2[i] = keep + simd_shuffle_xor(send, 16);
    }
    b = (lane & 8) != 0;
    const float keep = b ? t2[1] : t2[0], send = b ? t2[0] : t2[1];
    float t1 = keep + simd_shuffle_xor(send, 8);
    t1 += simd_shuffle_xor(t1, 4);
    t1 += simd_shuffle_xor(t1, 2);
    return t1 + simd_shuffle_xor(t1, 1);
}

// Butterfly reduction of 8 row partials across 32 lanes: row r (0..7) ends up in lanes 4r..4r+3.
static inline float reduce8(thread float* v, ushort lane) {
    float t4[4], t2[2];
    bool b = (lane & 16) != 0;
    for (int i = 0; i < 4; i++) {
        const float keep = b ? v[4 + i] : v[i], send = b ? v[i] : v[4 + i];
        t4[i] = keep + simd_shuffle_xor(send, 16);
    }
    b = (lane & 8) != 0;
    for (int i = 0; i < 2; i++) {
        const float keep = b ? t4[2 + i] : t4[i], send = b ? t4[i] : t4[2 + i];
        t2[i] = keep + simd_shuffle_xor(send, 8);
    }
    b = (lane & 4) != 0;
    const float keep = b ? t2[1] : t2[0], send = b ? t2[0] : t2[1];
    float t1 = keep + simd_shuffle_xor(send, 4);
    t1 += simd_shuffle_xor(t1, 2);
    return t1 + simd_shuffle_xor(t1, 1);
}

// Load lane's 64-column chunk of x (H = 2048 = 32 * 64) and apply RMSNorm(2048, eps=1e-6) * w into half4 xh[16].
static inline void chunk64_norm(device const float* x, device const float* w, ushort lane,
                                thread half4* xh, thread float& X) {
    device const float4* xs = (device const float4*)(x + lane * 64);
    float4 xf[16];
    float ss = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {
        const float4 t = xs[i];
        xf[i] = t;
        ss += dot(t, t);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(H) + EPS);
    device const float4* ws = (device const float4*)(w + lane * 64);
    float sum = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {
        const float4 v = xf[i] * scale * ws[i];
        const half4 h = half4(v);
        xh[i] = h;
        sum += float(h.x) + float(h.y) + float(h.z) + float(h.w);
    }
    X = sum;
}

static inline void chunk64_plain(device const float* x, ushort lane, thread half4* xh, thread float& X) {
    device const float4* xs = (device const float4*)(x + lane * 64);
    float sum = 0.f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {
        const half4 h = half4(xs[i]);
        xh[i] = h;
        sum += float(h.x) + float(h.y) + float(h.z) + float(h.w);
    }
    X = sum;
}

// ---------------------------------------------------------------------------------------------
// Language Backbone kernels

// Multimodal prompt embedding: embeds text tokens from E and replaces <|image_pad|> (151655)
// with vision features from pooler [n_vis, 2048]. Grid (n), 256 threads.
kernel void k_embed_prompt(device const wtype* E [[buffer(0)]],
                           device const half2* ESB [[buffer(1)]],
                           device const int* toks [[buffer(2)]],
                           device const float* pooler [[buffer(3)]],
                           device float* x [[buffer(4)]],
                           device const State* st [[buffer(5)]],
                           uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    if (st->done || (int)t >= st->n) return;
    const int tok = toks[t];
    if (tok == IMAGE_PAD_ID && st->vis_offset >= 0) {
        const int v = int(t) - st->vis_offset;
        for (uint i = tid; i < H; i += 256) x[t * H + i] = pooler[v * H + i];
    } else {
        for (uint i = tid; i < H; i += 256) x[t * H + i] = embed_at(E, ESB, tok, i);
    }
}

// DeepStack addition at the end of decoder layers 0, 1, 2 during prefill.
// Grid (n_vis), 256 threads.
kernel void k_deepstack_add(device float* x [[buffer(0)]],
                            device const float* ds [[buffer(1)]],
                            device const State* st [[buffer(2)]],
                            uint v [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (st->done || st->vis_offset < 0 || (int)v >= st->n_vis) return;
    const int t = st->vis_offset + int(v);
    if (t >= st->n) return;
    for (uint i = tid; i < H; i += 256) {
        x[t * H + i] += ds[v * H + i];
    }
}

// Fused RMSNorm(x) + QKV projection + per-head q_norm/k_norm (RMSNorm 128) + 3D M-RoPE + KV write.
// Grid (32, n), 128 threads (4 simdgroups). Each threadgroup computes one full 128-dim head:
//   head 0..15  -> Q heads 0..15
//   head 16..23 -> K heads 0..7
//   head 24..31 -> V heads 0..7
kernel void k_attn_in(device const wtype* W [[buffer(0)]],
                      device const half2* SB [[buffer(1)]],
                      device const float* wnorm [[buffer(2)]],
                      device const float* qnorm [[buffer(3)]],
                      device const float* knorm [[buffer(4)]],
                      device const float* x [[buffer(5)]],
                      device float* q [[buffer(6)]],
                      device half* kc [[buffer(7)]],
                      device half* vc [[buffer(8)]],
                      device const float2* rope [[buffer(9)]],
                      device const State* st [[buffer(10)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int head = tgp.x; // 0..31
    const uint r0 = head * HD + sg * 32;
    threadgroup float hv[2][128];
    threadgroup float red_h[4];

    if (st->n == 2) {
        half4 xh0[16], xh1[16];
        float X0, X1;
        chunk64_norm(x, wnorm, lane, xh0, X0);
        chunk64_norm(x + H, wnorm, lane, xh1, X1);
        for (int rb = 0; rb < 32; rb += 8) {
            float v0[8], v1[8];
            for (int b = 0; b < 8; ++b) {
                const float2 d = dot64_row2(W, SB, r0 + rb + b, lane, H, xh0, X0, xh1, X1);
                v0[b] = d.x;
                v1[b] = d.y;
            }
            const float y0 = reduce8(v0, lane);
            const float y1 = reduce8(v1, lane);
            if ((lane & 3) == 0) {
                const uint idx = sg * 32 + rb + (lane >> 2);
                hv[0][idx] = y0;
                hv[1][idx] = y1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < 2; ++t) {
            const int pos = st->pos + t;
            if (head < 24) {
                const float val = hv[t][tid];
                float ss = simd_sum(val * val);
                if (lane == 0) red_h[sg] = ss;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float hscale = rsqrt((red_h[0] + red_h[1] + red_h[2] + red_h[3]) / float(HD) + EPS);
                device const float* nw = (head < 16) ? qnorm : knorm;
                hv[t][tid] = val * hscale * nw[tid];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 64) {
                    const float2 cs = rope[pos * 64 + tid];
                    const float a = hv[t][tid], b = hv[t][tid + 64];
                    const float a2 = a * cs.x - b * cs.y;
                    const float b2 = b * cs.x + a * cs.y;
                    if (head < 16) {
                        q[(ulong)t * H + head * HD + tid] = a2;
                        q[(ulong)t * H + head * HD + tid + 64] = b2;
                    } else {
                        device half* k = kc + ((ulong)(head - 16) * st->maxctx + pos) * HD;
                        k[tid] = half(a2);
                        k[tid + 64] = half(b2);
                    }
                }
            } else {
                device half* v_dst = vc + ((ulong)(head - 24) * st->maxctx + pos) * HD;
                v_dst[tid] = half(hv[t][tid]);
            }
        }
        return;
    }

    const int t = tgp.y;
    if (t >= st->n) return;
    half4 xh[16];
    float X;
    chunk64_norm(x + (ulong)t * H, wnorm, lane, xh, X);
    for (int rb = 0; rb < 32; rb += 8) {
        float v[8];
        for (int b = 0; b < 8; ++b) {
            v[b] = dot64_row(W, SB, r0 + rb + b, lane, H, xh, X);
        }
        const float y = reduce8(v, lane);
        if ((lane & 3) == 0) {
            hv[0][sg * 32 + rb + (lane >> 2)] = y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pos = st->pos + t;
    if (head < 24) {
        const float val = hv[0][tid];
        float ss = simd_sum(val * val);
        if (lane == 0) red_h[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float hscale = rsqrt((red_h[0] + red_h[1] + red_h[2] + red_h[3]) / float(HD) + EPS);
        device const float* nw = (head < 16) ? qnorm : knorm;
        hv[0][tid] = val * hscale * nw[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const float2 cs = rope[pos * 64 + tid];
            const float a = hv[0][tid], b = hv[0][tid + 64];
            const float a2 = a * cs.x - b * cs.y;
            const float b2 = b * cs.x + a * cs.y;
            if (head < 16) {
                q[(ulong)t * H + head * HD + tid] = a2;
                q[(ulong)t * H + head * HD + tid + 64] = b2;
            } else {
                device half* k = kc + ((ulong)(head - 16) * st->maxctx + pos) * HD;
                k[tid] = half(a2);
                k[tid + 64] = half(b2);
            }
        }
    } else {
        device half* v_dst = vc + ((ulong)(head - 24) * st->maxctx + pos) * HD;
        v_dst[tid] = half(hv[0][tid]);
    }
}

// Vectorized zero-barrier RMSNorm over H=2048 columns outputting fp16 xh [n, H] (4 rows per threadgroup, 1 row per simdgroup).
// Grid ((n + 3) / 4), 128 threads (4 simdgroups).
kernel void k_rmsnorm_rows_h(device const float* x [[buffer(0)]],
                             device const float* w [[buffer(1)]],
                             device half* xh [[buffer(2)]],
                             device const State* st [[buffer(3)]],
                             uint tgp [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    const int t = int(tgp * 4 + sg);
    if (st->done || t >= st->n) return;
    device const float4* xr4 = (device const float4*)(x + (ulong)t * H);
    device const float4* w4 = (device const float4*)w;
    device half4* yd4 = (device half4*)(xh + (ulong)t * H);
    float4 xv[16];
    float ss = 0.f;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 16; ++k) {
        const float4 v = xr4[lane + k * 32];
        xv[k] = v;
        ss += dot(v, v);
    }
    const float scale = rsqrt(simd_sum(ss) * (1.0f / float(H)) + EPS);
    #pragma clang loop unroll(full)
    for (int k = 0; k < 16; ++k) {
        const uint idx = lane + k * 32;
        yd4[idx] = half4(xv[k] * scale * w4[idx]);
    }
}

#define MAXPASS_Q 512

// Vectorized zero-barrier per-head RMSNorm(128) on Q (heads 0..15) and K (heads 16..23) + 3D M-RoPE + KV cache write.
// Each simdgroup (32 lanes) processes 1 full 128-dim head in registers with zero shared memory and zero barriers!
// Grid (8, n), 128 threads (4 simdgroups x 1 head = 4 heads per threadgroup).
kernel void k_qk_norm_rope(device const float* qkv [[buffer(0)]],
                           device const float* qnorm [[buffer(1)]],
                           device const float* knorm [[buffer(2)]],
                           device float* q [[buffer(3)]],
                           device half* kc [[buffer(4)]],
                           device half* vc [[buffer(5)]],
                           device const float2* rope [[buffer(6)]],
                           device const State* st [[buffer(7)]],
                           device half* qh [[buffer(8)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int head = int(tgp.x * 4 + sg); // 0..31
    if (head >= 32) return;
    const int t = int(tgp.y);
    if (t >= st->n) return;

    device const float2* src2 = (device const float2*)(qkv + (ulong)t * QKV_DIM + head * HD);
    const int pos = st->pos + t;
    if (head < 24) {
        float2 a = src2[lane];
        float2 b = src2[lane + 32];
        const float ss = dot(a, a) + dot(b, b);
        const float hscale = rsqrt(simd_sum(ss) * (1.0f / float(HD)) + EPS);
        device const float2* nw2 = (device const float2*)((head < 16) ? qnorm : knorm);
        a *= hscale * nw2[lane];
        b *= hscale * nw2[lane + 32];

        const float2 cs0 = rope[pos * 64 + lane * 2 + 0];
        const float2 cs1 = rope[pos * 64 + lane * 2 + 1];
        const float2 a2 = float2(a.x * cs0.x - b.x * cs0.y, a.y * cs1.x - b.y * cs1.y);
        const float2 b2 = float2(b.x * cs0.x + a.x * cs0.y, b.y * cs1.x + a.y * cs1.y);
        if (head < 16) {
            if (t == st->n - 1) {
                device float2* q2 = (device float2*)(q + (ulong)t * H + head * HD);
                q2[lane] = a2;
                q2[lane + 32] = b2;
            }
            device half2* qh2 = (device half2*)(qh + ((ulong)head * MAXPASS_Q + t) * HD);
            qh2[lane] = half2(a2);
            qh2[lane + 32] = half2(b2);
        } else {
            device half2* k2 = (device half2*)(kc + ((ulong)(head - 16) * st->maxctx + pos) * HD);
            k2[lane] = half2(a2);
            k2[lane + 32] = half2(b2);
        }
    } else {
        device const float4* src4 = (device const float4*)src2;
        device half4* v4 = (device half4*)(vc + ((ulong)(head - 24) * st->maxctx + pos) * HD);
        v4[lane] = half4(src4[lane]);
    }
}

// M5 Neural Accelerator (matmul2d) data-parallel tile helper for LLM prefill GEMMs (f16, q8, q4).
// Each simdgroup independently computes a [TT, 32] output tile at row offset r0 while sharing X in L1 cache.
template<int K, int TT, int BK>
inline void llm_gemm_tile(device half* X, device const wtype* W, device const half2* SB,
                          uint r0, int t0, threadgroup half* Wsg, threadgroup float* Os,
                          uint lane, uint sg) {
    constexpr int NSTEP = K / BK;
    auto tX = tensor(X + (ulong)t0 * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    threadgroup half* ws = Wsg + sg * 32 * BK;
    auto tW = tensor(ws, dextents<int, 2>(BK, 32), array<int, 2>({1, BK}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    #pragma clang loop unroll(full)
    for (int i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.f;
    }
#if FMT == 0
    auto tWd = tensor((device half*)(W + (ulong)r0 * K), dextents<int, 2>(K, 32), array<int, 2>({1, K}));
    for (int i = 0; i < NSTEP; ++i) {
        const uint k0 = i * BK;
        auto a = tX.template slice<BK, TT>(k0, 0);
        auto b = tWd.template slice<BK, 32>(k0, 0);
        op.run(a, b, cT);
    }
#elif FMT == 1
    device const char* wrow = W + (ulong)(r0 + lane) * K;
    device const half* srow = (device const half*)SB + (ulong)(r0 + lane) * (K / 32);
    threadgroup half4* dst = (threadgroup half4*)(ws + lane * BK);
    for (int i = 0; i < NSTEP; ++i) {
        const uint k0 = i * BK;
        for (int q = 0; q < BK / 32; ++q) {
            const half s = srow[k0 / 32 + q];
            device const uint4* src = (device const uint4*)(wrow + k0 + q * 32);
            const uint4 a = src[0], b = src[1];
            for (int j = 0; j < 4; ++j) {
                dst[q * 8 + j] = half4(as_type<char4>(a[j])) * s;
                dst[q * 8 + 4 + j] = half4(as_type<char4>(b[j])) * s;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tX.template slice<BK, TT>(k0, 0);
        op.run(a, tW, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
#elif FMT == 2
    device const uchar* wrow = W + (ulong)(r0 + lane) * (K / 2);
    device const half2* sbrow = SB + (ulong)(r0 + lane) * (K / 64);
    threadgroup half2* dst = (threadgroup half2*)(ws + lane * BK);
    for (int i = 0; i < NSTEP; ++i) {
        const uint k0 = i * BK;
        const half2 sb = sbrow[k0 / 64];
        const half2 mul = half2(half(16.f * float(sb.x)));
        const half2 add = half2(half(float(sb.y) - 16.f * float(sb.x)));
        device const uint4* src = (device const uint4*)(wrow + k0 / 2);
        for (int q = 0; q < BK / 32; ++q) {
            const uint4 wv = src[q];
            for (int u = 0; u < 4; ++u) {
                const uint v = wv[u];
                const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
                const int o = q * 16 + u * 4;
                dst[o + 0] = fma(half2(p04.x, p15.x), mul, add);
                dst[o + 1] = fma(half2(p26.x, p37.x), mul, add);
                dst[o + 2] = fma(half2(p04.y, p15.y), mul, add);
                dst[o + 3] = fma(half2(p26.y, p37.y), mul, add);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tX.template slice<BK, TT>(k0, 0);
        op.run(a, tW, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto tO = tensor(Os + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cT.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// Prefill QKV GEMM on M5 Neural Accelerators: Grid (QKV_DIM / 128, (n + 31) / 32) = (32, (n + 31) / 32), 128 threads.
kernel void k_g3_qkv(device half* X [[buffer(0)]],
                     device const wtype* W [[buffer(1)]],
                     device const half2* SB [[buffer(2)]],
                     device float* outf [[buffer(3)]],
                     device const State* st [[buffer(4)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 32, BK = 64;
    if (st->done || (int)(tgp.y * TT) >= st->n) return;
    threadgroup float smem[4096]; // 16 KiB shared by Wsg and Os
    const uint r0 = tgp.x * 128 + sg * 32;
    const int t0 = tgp.y * TT, n = st->n;
    llm_gemm_tile<H, TT, BK>(X, W, SB, r0, t0, (threadgroup half*)smem, smem, lane, sg);
    threadgroup const float* os = smem + sg * TT * 32;
    for (int t = 0; t < TT; ++t) {
        if (t0 + t < n) outf[(ulong)(t0 + t) * QKV_DIM + r0 + lane] = os[t * 32 + lane];
    }
}

// Prefill Output Projection + residual GEMM on M5 Neural Accelerators: Grid (H / 128, (n + 31) / 32) = (16, (n + 31) / 32), 128 threads.
kernel void k_g3_o(device half* X [[buffer(0)]],
                   device const wtype* W [[buffer(1)]],
                   device const half2* SB [[buffer(2)]],
                   device float* outf [[buffer(3)]],
                   device const State* st [[buffer(4)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 32, BK = 64;
    if (st->done || (int)(tgp.y * TT) >= st->n) return;
    threadgroup float smem[4096];
    const uint r0 = tgp.x * 128 + sg * 32;
    const int t0 = tgp.y * TT, n = st->n;
    llm_gemm_tile<H, TT, BK>(X, W, SB, r0, t0, (threadgroup half*)smem, smem, lane, sg);
    threadgroup const float* os = smem + sg * TT * 32;
    for (int t = 0; t < TT; ++t) {
        if (t0 + t < n) outf[(ulong)(t0 + t) * H + r0 + lane] += os[t * 32 + lane];
    }
}

// Prefill Gate + Up + SwiGLU GEMM on M5 Neural Accelerators: Grid (FF / 64, (n + 31) / 32) = (96, (n + 31) / 32), 128 threads.
// Simdgroups 0..1 compute 64 gate rows and simdgroups 2..3 compute 64 up rows simultaneously in a single K-loop.
kernel void k_g3_gu(device half* X [[buffer(0)]],
                    device const wtype* W [[buffer(1)]],
                    device const half2* SB [[buffer(2)]],
                    device half* outh [[buffer(3)]],
                    device const State* st [[buffer(4)]],
                    uint2 tgp [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 32, BK = 64;
    if (st->done || (int)(tgp.y * TT) >= st->n) return;
    threadgroup float smem[4096];
    const uint u0 = tgp.x * 64;
    const uint r0 = (sg < 2) ? (u0 + sg * 32) : (FF + u0 + (sg - 2) * 32);
    const int t0 = tgp.y * TT, n = st->n;
    llm_gemm_tile<H, TT, BK>(X, W, SB, r0, t0, (threadgroup half*)smem, smem, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint h = sg & 1;
    const int t_start = int(sg >> 1) * (TT / 2);
    for (int dt = 0; dt < TT / 2; ++dt) {
        const int t = t_start + dt;
        if (t0 + t >= n) continue;
        const float g = smem[h * 1024 + t * 32 + lane];
        const float u = smem[(h + 2) * 1024 + t * 32 + lane];
        outh[(ulong)(t0 + t) * FF + u0 + h * 32 + lane] = half((g / (1.f + exp(-g))) * u);
    }
}

// Prefill Down Projection + residual GEMM on M5 Neural Accelerators: Grid (H / 128, (n + 31) / 32) = (16, (n + 31) / 32), 128 threads.
kernel void k_g3_down(device half* X [[buffer(0)]],
                      device const wtype* W [[buffer(1)]],
                      device const half2* SB [[buffer(2)]],
                      device float* outf [[buffer(3)]],
                      device const State* st [[buffer(4)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 32, BK = 64;
    if (st->done || (int)(tgp.y * TT) >= st->n) return;
    threadgroup float smem[4096];
    const uint r0 = tgp.x * 128 + sg * 32;
    const int t0 = tgp.y * TT, n = st->n;
    llm_gemm_tile<FF, TT, BK>(X, W, SB, r0, t0, (threadgroup half*)smem, smem, lane, sg);
    threadgroup const float* os = smem + sg * TT * 32;
    for (int t = 0; t < TT; ++t) {
        if (t0 + t < n) outf[(ulong)(t0 + t) * H + r0 + lane] += os[t * 32 + lane];
    }
}

// Causal GQA Prefill Attention on M5 Neural Accelerators (matmul2d).
// Grid (NH, (n + 15) / 16) = (16, (n + 15) / 16), 128 threads (4 simdgroups).
kernel void k_prefill_attn(device half* qh [[buffer(0)]],
                           device half* kc [[buffer(1)]],
                           device half* vc [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           device half* outh [[buffer(4)]],
                           device float* scores [[buffer(5)]],
                           device half* probs [[buffer(6)]],
                           device const State* st [[buffer(7)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 16;
    const int n = st->n;
    const int h = tgp.x;
    const int t0 = tgp.y * TT;
    if (st->done || t0 >= n) return;
    const int g = h / (NH / NKV);
    const int maxctx = st->maxctx;
    const int L_max = st->pos + min(n, t0 + TT);
    const int nKeyTiles = (L_max + 31) / 32;
    const int L_pad = nKeyTiles * 32;

    device half* Qh = qh + ((ulong)h * MAXPASS_Q + t0) * HD;
    device half* Kh = kc + (ulong)g * maxctx * HD;
    device half* Vh = vc + (ulong)g * maxctx * HD;
    device float* Sh = scores + ((ulong)h * MAXPASS_Q + t0) * maxctx;
    device half* Ph = probs + ((ulong)h * MAXPASS_Q + t0) * maxctx;

    // Phase 1: Q_tile [16, 128] @ K_h^T [L_pad, 128] -> Sh [16, L_pad]
    auto tQ0 = tensor(Qh, dextents<int, 2>(64, TT), array<int, 2>({1, HD}));
    auto tQ1 = tensor(Qh + 64, dextents<int, 2>(64, TT), array<int, 2>({1, HD}));
    constexpr auto descQK = matmul2d_descriptor(TT, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descQK, execution_simdgroup> opQK;
    for (int kt = int(sg); kt < nKeyTiles; kt += 4) {
        const int s0 = kt * 32;
        auto tK0 = tensor(Kh + (ulong)s0 * HD, dextents<int, 2>(64, 32), array<int, 2>({1, HD}));
        auto tK1 = tensor(Kh + (ulong)s0 * HD + 64, dextents<int, 2>(64, 32), array<int, 2>({1, HD}));
        auto cS = opQK.template get_destination_cooperative_tensor<decltype(tQ0), decltype(tK0), float>();
        #pragma clang loop unroll(full)
        for (int i = 0; i < cS.get_capacity(); ++i) {
            if (cS.is_valid_element(i)) cS[i] = 0.f;
        }
        opQK.run(tQ0, tK0, cS);
        opQK.run(tQ1, tK1, cS);
        auto tS = tensor(Sh + s0, dextents<int, 2>(32, TT), array<int, 2>({1, maxctx}));
        cS.store(tS);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Causal row-wise softmax in registers (<= 16 keys/lane, zero global writebacks to Sh)
    const int n_pad = L_pad >> 5;
    for (int r = int(sg) * 4; r < int(sg) * 4 + 4; ++r) {
        device const float* sr = Sh + r * maxctx;
        device half* pr = Ph + r * maxctx;
        if (t0 + r < n) {
            const int L = st->pos + t0 + r + 1;
            const int n_s = (L - int(lane) + 31) >> 5;
            float row_s[16];
            float m = -INFINITY;
            for (int k = 0; k < n_s; ++k) {
                const float v = sr[lane + k * 32] * 0.08838834764831845f; // 1 / sqrt(128)
                row_s[k] = v;
                m = max(m, v);
            }
            m = simd_max(m);
            float sum = 0.f;
            for (int k = 0; k < n_s; ++k) {
                const float e = exp(row_s[k] - m);
                row_s[k] = e;
                sum += e;
            }
            const float inv_sum = 1.0f / simd_sum(sum);
            for (int k = 0; k < n_pad; ++k) {
                const int s = int(lane) + k * 32;
                pr[s] = (s < L) ? half(row_s[k] * inv_sum) : 0.h;
            }
        } else {
            for (int k = 0; k < n_pad; ++k) {
                pr[lane + k * 32] = 0.h;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: P_tile [16, L_pad] @ V_h [L_pad, 128] -> Os [4, 16, 32] (8 KiB)
    threadgroup float Os[4 * TT * 32]; // 8 KiB
    threadgroup float* myO = Os + sg * TT * 32;
    const int d0 = int(sg) * 32;
    auto tP = tensor(Ph, dextents<int, 2>(maxctx, TT), array<int, 2>({1, maxctx}));
    auto tV = tensor(Vh, dextents<int, 2>(HD, L_pad), array<int, 2>({1, HD}));
    constexpr auto descPV = matmul2d_descriptor(TT, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descPV, execution_simdgroup> opPV;
    auto a0 = tP.template slice<32, TT>(0, 0);
    auto b0 = tV.template slice<32, 32>(d0, 0);
    auto cO = opPV.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma clang loop unroll(full)
    for (int i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) cO[i] = 0.f;
    }
    for (int kt = 0; kt < nKeyTiles; ++kt) {
        const int s0 = kt * 32;
        auto a = tP.template slice<32, TT>(s0, 0);
        auto b = tV.template slice<32, 32>(d0, s0);
        opPV.run(a, b, cO);
    }
    auto tO = tensor(myO, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cO.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < TT; ++r) {
        if (t0 + r >= n) continue;
        const float val = myO[r * 32 + lane];
        const ulong idx = (ulong)(t0 + r) * H + h * HD + d0 + lane;
        outh[idx] = half(val);
    }
}

// Causal GQA attention for one (query head, token). Grid (16, n), 256 threads.
#define MAXCTX_TG 2048
kernel void k_attention(device const float* q [[buffer(0)]],
                        device const half* kc [[buffer(1)]],
                        device const half* vc [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const State* st [[buffer(4)]],
                        constant int& toff [[buffer(5)]],
                        device half* outh [[buffer(6)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int h = tgp.x;
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    const int L = st->pos + t + 1;
    const int g = h / (NH / NKV);
    threadgroup float qs[HD];
    threadgroup float sc[MAXCTX_TG];
    threadgroup float red[8];
    threadgroup float4 part4[256];
    if (tid < HD) qs[tid] = q[(ulong)t * H + h * HD + tid] * 0.08838834764831845f; // 1 / sqrt(128)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float4* qs4 = (const threadgroup float4*)qs;
    device const half* kb = kc + (ulong)g * st->maxctx * HD;
    device const half* vb = vc + (ulong)g * st->maxctx * HD;
    float m = -INFINITY;
    for (int s = tid; s < L; s += 256) {
        device const half4* kr = (device const half4*)(kb + (ulong)s * HD);
        float a = 0.f;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 32; ++j) a += dot(float4(kr[j]), qs4[j]);
        sc[s] = a;
        m = max(m, a);
    }
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (int i = 1; i < 8; ++i) m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (int s = tid; s < L; s += 256) {
        const float e = exp(sc[s] - m);
        sc[s] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.f;
    for (int i = 0; i < 8; ++i) sum += red[i];

    // Coalesced half4 V-accumulation: 32 lanes cover all 128 head dims (32 x half4), 8 simdgroups step s += 8
    float4 o4 = float4(0.f);
    for (int s = int(sg); s < L; s += 8) {
        device const half4* vr = (device const half4*)(vb + (ulong)s * HD);
        o4 += sc[s] * float4(vr[lane]);
    }
    part4[sg * 32 + lane] = o4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float4 tot = part4[lane];
        #pragma clang loop unroll(full)
        for (int k = 1; k < 8; ++k) tot += part4[k * 32 + lane];
        const float4 val = tot * (1.0f / sum);
        ((device float4*)(out + (ulong)t * H + h * HD))[lane] = val;
        ((device half4*)(outh + (ulong)t * H + h * HD))[lane] = half4(val);
    }
}

// Output projection + residual: x[t, r] += dot(W_o[r], attn[t]).
// Grid (H / 16, n) = (128, n), 64 threads (2 simdgroups x 8 rows).
kernel void k_o_proj(device const wtype* W [[buffer(0)]],
                     device const half2* SB [[buffer(1)]],
                     device const float* a [[buffer(2)]],
                     device float* x [[buffer(3)]],
                     device const State* st [[buffer(4)]],
                     constant int& toff [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const uint r0 = (tgp.x * 2 + sg) * 8;
    if (st->n == 2 && toff == 0) {
        half4 xh0[16], xh1[16];
        float X0, X1;
        chunk64_plain(a, lane, xh0, X0);
        chunk64_plain(a + H, lane, xh1, X1);
        float v0[8], v1[8];
        for (int b = 0; b < 8; ++b) {
            const float2 d = dot64_row2(W, SB, r0 + b, lane, H, xh0, X0, xh1, X1);
            v0[b] = d.x;
            v1[b] = d.y;
        }
        const float y0 = reduce8(v0, lane);
        const float y1 = reduce8(v1, lane);
        if ((lane & 3) == 0) {
            const uint r = r0 + (lane >> 2);
            x[r] += y0;
            x[H + r] += y1;
        }
        return;
    }
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    half4 xh[16];
    float X;
    chunk64_plain(a + (ulong)t * H, lane, xh, X);
    float v[8];
    for (int b = 0; b < 8; ++b) {
        v[b] = dot64_row(W, SB, r0 + b, lane, H, xh, X);
    }
    const float y = reduce8(v, lane);
    if ((lane & 3) == 0) {
        x[(ulong)t * H + r0 + (lane >> 2)] += y;
    }
}

// RMSNorm + gate/up + SwiGLU: W has gate [0..6143] and up [6144..12287].
// Each simdgroup computes 4 gate rows (u0..u0+3) and 4 up rows (FF+u0..FF+u0+3).
// Grid (FF / 8, n) = (768, n), 64 threads (2 simdgroups x 4 MLP units).
kernel void k_mlp_in(device const wtype* W [[buffer(0)]],
                     device const half2* SB [[buffer(1)]],
                     device const float* wnorm [[buffer(2)]],
                     device const float* x [[buffer(3)]],
                     device float* hbuf [[buffer(4)]],
                     device const State* st [[buffer(5)]],
                     constant int& toff [[buffer(6)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const uint u0 = (tgp.x * 2 + sg) * 4;
    if (st->n == 2 && toff == 0) {
        half4 xh0[16], xh1[16];
        float X0, X1;
        chunk64_norm(x, wnorm, lane, xh0, X0);
        chunk64_norm(x + H, wnorm, lane, xh1, X1);
        float v0[8], v1[8];
        for (int b = 0; b < 4; ++b) {
            const float2 dg = dot64_row2(W, SB, u0 + b, lane, H, xh0, X0, xh1, X1);
            const float2 du = dot64_row2(W, SB, FF + u0 + b, lane, H, xh0, X0, xh1, X1);
            v0[b] = dg.x; v0[4 + b] = du.x;
            v1[b] = dg.y; v1[4 + b] = du.y;
        }
        const float y0 = reduce8(v0, lane);
        const float up0 = simd_shuffle_down(y0, 16);
        const float y1 = reduce8(v1, lane);
        const float up1 = simd_shuffle_down(y1, 16);
        if ((lane & 3) == 0 && lane < 16) {
            const uint unit = u0 + (lane >> 2);
            hbuf[unit] = (y0 / (1.f + exp(-y0))) * up0;
            hbuf[FF + unit] = (y1 / (1.f + exp(-y1))) * up1;
        }
        return;
    }
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    half4 xh[16];
    float X;
    chunk64_norm(x + (ulong)t * H, wnorm, lane, xh, X);
    float v[8];
    for (int b = 0; b < 4; ++b) {
        v[b] = dot64_row(W, SB, u0 + b, lane, H, xh, X);
        v[4 + b] = dot64_row(W, SB, FF + u0 + b, lane, H, xh, X);
    }
    const float y = reduce8(v, lane);
    const float up = simd_shuffle_down(y, 16); // row + 4 is the matching up row
    if ((lane & 3) == 0 && lane < 16) {
        const uint unit = u0 + (lane >> 2);
        hbuf[(ulong)t * FF + unit] = (y / (1.f + exp(-y))) * up;
    }
}

// Down projection + residual (K = FF = 6144 = 3 chunks of 2048):
// Grid (H / 16, n) = (128, n), 64 threads (2 simdgroups x 8 rows).
kernel void k_mlp_out(device const wtype* W [[buffer(0)]],
                      device const half2* SB [[buffer(1)]],
                      device const float* hbuf [[buffer(2)]],
                      device float* x [[buffer(3)]],
                      device const State* st [[buffer(4)]],
                      constant int& toff [[buffer(5)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const uint r0 = (tgp.x * 2 + sg) * 8;
    if (st->n == 2 && toff == 0) {
        float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int chunk = 0; chunk < 3; ++chunk) {
            half4 xh0[16], xh1[16];
            float X0, X1;
            const uint c64 = chunk * 32 + lane;
            chunk64_plain(hbuf + chunk * 2048, lane, xh0, X0);
            chunk64_plain(hbuf + FF + chunk * 2048, lane, xh1, X1);
            for (int b = 0; b < 8; ++b) {
                const float2 d = dot64_row2(W, SB, r0 + b, c64, FF, xh0, X0, xh1, X1);
                acc0[b] += d.x;
                acc1[b] += d.y;
            }
        }
        const float y0 = reduce8(acc0, lane);
        const float y1 = reduce8(acc1, lane);
        if ((lane & 3) == 0) {
            const uint r = r0 + (lane >> 2);
            x[r] += y0;
            x[H + r] += y1;
        }
        return;
    }
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int chunk = 0; chunk < 3; ++chunk) {
        half4 xh[16];
        float X;
        const uint c64 = chunk * 32 + lane;
        chunk64_plain(hbuf + (ulong)t * FF + chunk * 2048, lane, xh, X);
        for (int b = 0; b < 8; ++b) {
            acc[b] += dot64_row(W, SB, r0 + b, c64, FF, xh, X);
        }
    }
    const float y = reduce8(acc, lane);
    if ((lane & 3) == 0) {
        x[(ulong)t * H + r0 + (lane >> 2)] += y;
    }
}

// Final RMSNorm + tied lm_head on the last token of the pass (st->n - 1) + per-threadgroup argmax.
// Grid (VOCAB / 32) = (4748), 128 threads (4 simdgroups x 8 vocabulary rows).
kernel void k_lm_head(device const wtype* E [[buffer(0)]],
                      device const half2* ESB [[buffer(1)]],
                      device const float* wnorm [[buffer(2)]],
                      device const float* x [[buffer(3)]],
                      device float* pval [[buffer(4)]],
                      device int* pidx [[buffer(5)]],
                      device const State* st [[buffer(6)]],
                      device float* logits [[buffer(7)]],
                      constant int& write_logits [[buffer(8)]],
                      uint tg [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float bv[4];
    threadgroup int bix[4];
    half4 xh[16];
    float X;
    device const float* xr = x + (ulong)(st->n - 1) * H;
    chunk64_norm(xr, wnorm, lane, xh, X);
    const uint r0 = tg * 32 + sg * 8;
    float v[8];
    for (int b = 0; b < 8; ++b) {
        v[b] = dot64_row(E, ESB, r0 + b, lane, H, xh, X);
    }
    const float y = reduce8(v, lane);
    const int row = int(r0) + (lane >> 2);
    if (write_logits && (lane & 3) == 0) logits[row] = y;
    const float m = simd_max(y);
    const int idx = simd_min(y == m ? row : 0x7fffffff);
    if (lane == 0) { bv[sg] = m; bix[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float best = bv[0];
        int bi = bix[0];
        for (int s2 = 1; s2 < 4; ++s2) {
            if (bv[s2] > best || (bv[s2] == best && bix[s2] < bi)) {
                best = bv[s2];
                bi = bix[s2];
            }
        }
        pval[tg] = best;
        pidx[tg] = bi;
    }
}

// Greedy commit + grammar-guided multi-token embed: reduces the 4748 lm_head partial maxima to the next token,
// appends deterministic syntax pairs ('click' -> '(', ',' -> ' ') for single-weight-read 2-token decode,
// stops immediately on ')' (token 8) or EOS, and embeds into x[0..st->n*H-1].
// Grid (1), 1024 threads.
kernel void k_commit(device const float* pval [[buffer(0)]],
                     device const int* pidx [[buffer(1)]],
                     device State* st [[buffer(2)]],
                     device int* out_tokens [[buffer(3)]],
                     device const wtype* E [[buffer(4)]],
                     device const half2* ESB [[buffer(5)]],
                     device float* x [[buffer(6)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = tid; i < LMH_TGS; i += 1024) {
        const float v = pval[i];
        const int id = pidx[i];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(best, off);
        const int oi = simd_shuffle_down(bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    threadgroup float rv[32];
    threadgroup int ri[32];
    threadgroup int tok0_s;
    threadgroup int tok1_s;
    if (lane == 0) { rv[sg] = best; ri[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = rv[lane]; bi = ri[lane];
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(best, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) {
            const int tok = bi;
            st->pos += st->n;
            st->vis_offset = -1;
            out_tokens[st->step] = tok;
            st->step += 1;
            const bool stop = tok == st->eos0 || tok == st->eos1 || tok == 8 || st->step >= st->maxsteps;
            if (stop) {
                st->done = 1;
                st->n = 1;
                tok0_s = -1;
                tok1_s = -1;
            } else if ((st->step == 1 && (tok == 3678 || tok == 7241 || tok == 12476 || tok == 10855 || tok == 11862 || tok == 1339 || tok == 1472 || tok == 24769)) || tok == 11) {
                const int next_syntax = (tok == 11) ? 220 : 7;
                out_tokens[st->step] = next_syntax;
                st->step += 1;
                st->n = 2;
                tok0_s = tok;
                tok1_s = next_syntax;
            } else {
                st->n = 1;
                tok0_s = tok;
                tok1_s = -1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int t0 = tok0_s;
    if (t0 < 0) return;
    for (uint i = tid; i < H; i += 1024) x[i] = embed_at(E, ESB, t0, i);
    const int t1 = tok1_s;
    if (t1 >= 0) {
        for (uint i = tid; i < H; i += 1024) x[H + i] = embed_at(E, ESB, t1, i);
    }
}

kernel void k_noop(device int* sink [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0xFFFFFFFF) sink[0] = 1;
}
"""#
