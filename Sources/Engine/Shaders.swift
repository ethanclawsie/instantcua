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

// Standard LayerNorm over D columns outputting fp16 for Neural Accelerator GEMM consumption.
// Grid (nrows), 256 threads.
template<int D>
kernel void k_layernorm_h_T(device const float* x [[buffer(0)]],
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
    device const float* xr = x + r * D;
    float sum = 0.f;
    for (int i = tid; i < D; i += 256) sum += xr[i];
    sum = simd_sum(sum);
    if (lane == 0) red_m[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = (red_m[0] + red_m[1] + red_m[2] + red_m[3] + red_m[4] + red_m[5] + red_m[6] + red_m[7]) / float(D);
    float vsum = 0.f;
    for (int i = tid; i < D; i += 256) {
        const float d = xr[i] - mean;
        vsum += d * d;
    }
    vsum = simd_sum(vsum);
    if (lane == 0) red_v[sg] = vsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3] + red_v[4] + red_v[5] + red_v[6] + red_v[7]) / float(D) + EPS);
    device half* yr = out + r * D;
    for (int i = tid; i < D; i += 256) {
        yr[i] = half((xr[i] - mean) * inv_std * w[i] + b[i]);
    }
}

template [[host_name("k_vit_ln1024")]] kernel void k_layernorm_h_T<1024>(device const float*, device const float*, device const float*, device half*, constant int&, uint, uint, uint, uint);
template [[host_name("k_vit_ln4096")]] kernel void k_layernorm_h_T<4096>(device const float*, device const float*, device const float*, device half*, constant int&, uint, uint, uint, uint);

// M5 Neural Accelerator (matmul2d) ViT QKV projection + 2D Vision RoPE:
// Each threadgroup computes one 64-dim head (h_all = 0..47) across TT = 32 patches,
// applies 2D Vision RoPE to Q and K, and writes head-major fp16 vq, vk, vv [16, np, 64].
// Grid (48, np / 32), 128 threads (4 simdgroups splitting K = 1024 in steps of BK = 64).
kernel void k_vit_qkv(device half* X [[buffer(0)]],
                      device half* W [[buffer(1)]],
                      device const float* bias [[buffer(2)]],
                      device const float2* vrope [[buffer(3)]],
                      device half* vq [[buffer(4)]],
                      device half* vk [[buffer(5)]],
                      device half* vv [[buffer(6)]],
                      constant int& np [[buffer(7)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TT = 32;
    constexpr int BK = 64;
    constexpr int NS = 4;
    constexpr int NSTEP = VH / BK / NS; // 1024 / 64 / 4 = 4
    const int h_all = tgp.x; // 0..47
    const int t0 = tgp.y * TT;
    if (t0 >= np) return;

    const uint r0 = h_all * 64;
    threadgroup float Os0[NS * TT * 32];
    threadgroup float Os1[NS * TT * 32];

    auto tX = tensor(X + (ulong)t0 * VH, dextents<int, 2>(VH, TT), array<int, 2>({1, VH}));
    auto tW0 = tensor(W + (ulong)r0 * VH, dextents<int, 2>(VH, 32), array<int, 2>({1, VH}));
    auto tW1 = tensor(W + (ulong)(r0 + 32) * VH, dextents<int, 2>(VH, 32), array<int, 2>({1, VH}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT0 = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW0), float>();
    auto cT1 = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW1), float>();
    for (int i = 0; i < cT0.get_capacity(); i++) if (cT0.is_valid_element(i)) cT0[i] = 0.f;
    for (int i = 0; i < cT1.get_capacity(); i++) if (cT1.is_valid_element(i)) cT1[i] = 0.f;

    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = (sg * NSTEP + i) * BK;
        auto a = tX.template slice<BK, TT>(k0, 0);
        auto b0 = tW0.template slice<BK, 32>(k0, 0);
        auto b1 = tW1.template slice<BK, 32>(k0, 0);
        op.run(a, b0, cT0);
        op.run(a, b1, cT1);
    }

    auto tO0 = tensor(Os0 + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    auto tO1 = tensor(Os1 + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cT0.store(tO0);
    cT1.store(tO1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int w = tid; w < TT * 32; w += 128) {
        const int t = w / 32, d = w % 32;
        const int p = t0 + t;
        if (p >= np) continue;
        float a = bias[r0 + d], b = bias[r0 + 32 + d];
        for (int q = 0; q < NS; ++q) {
            a += Os0[q * TT * 32 + w];
            b += Os1[q * TT * 32 + w];
        }
        if (h_all < 32) {
            const float2 cs = vrope[p * 32 + d];
            const float a2 = a * cs.x - b * cs.y;
            const float b2 = b * cs.x + a * cs.y;
            a = a2; b = b2;
        }
        if (h_all < 16) {
            device half* qdst = vq + ((ulong)h_all * np + p) * VHD;
            qdst[d] = half(a);
            qdst[d + 32] = half(b);
        } else if (h_all < 32) {
            const int kh = h_all - 16;
            device half* kdst = vk + ((ulong)kh * np + p) * VHD;
            kdst[d] = half(a);
            kdst[d + 32] = half(b);
        } else {
            const int vh = h_all - 32;
            device half* vdst = vv + ((ulong)vh * np + p) * VHD;
            vdst[d] = half(a);
            vdst[d + 32] = half(b);
        }
    }
}

// ViT bidirectional full attention across np patches (head-major fp16 vq, vk, vv [16, np, 64] -> fp16 out [np, 1024]).
// Grid (16, np), 256 threads.
kernel void k_vit_attn(device const half* vq [[buffer(0)]],
                       device const half* vk [[buffer(1)]],
                       device const half* vv [[buffer(2)]],
                       device half* out [[buffer(3)]],
                       constant int& np [[buffer(4)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    const int h = tgp.x;
    const int p = tgp.y;
    if (p >= np) return;
    threadgroup float qs[VHD];
    threadgroup float sc[NPATCH_MAX];
    threadgroup float red[8];
    threadgroup float part[256];
    device const half* Qh = vq + (ulong)h * np * VHD;
    device const half* Kh = vk + (ulong)h * np * VHD;
    device const half* Vh = vv + (ulong)h * np * VHD;
    if (tid < VHD) qs[tid] = float(Qh[p * VHD + tid]) * 0.125f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float4* qs4 = (const threadgroup float4*)qs;
    float m = -INFINITY;
    for (int s = tid; s < np; s += 256) {
        device const half4* kr = (device const half4*)(Kh + s * VHD);
        float a = 0.f;
        for (int j = 0; j < 16; ++j) a += dot(float4(kr[j]), qs4[j]);
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
    for (int s = tid; s < np; s += 256) {
        const float e = exp(sc[s] - m);
        sc[s] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.f;
    for (int i = 0; i < 8; ++i) sum += red[i];
    const int d = tid & 63;
    const int q = tid >> 6;
    float o = 0.f;
    for (int s = q; s < np; s += 4) {
        o += sc[s] * float(Vh[s * VHD + d]);
    }
    part[tid] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < VHD) {
        out[p * VH + h * VHD + tid] = half((part[tid] + part[tid + 64] + part[tid + 128] + part[tid + 192]) / sum);
    }
}

// Generic M5 Neural Accelerator (matmul2d) fp16 linear kernel for ViT & Mergers:
// MODE 0: outf[p, r] = dot(W[r], X[p]) + bias[r] (Merger fc2 -> f32)
// MODE 1: outf[p, r] += dot(W[r], X[p]) + bias[r] (ViT proj & fc2 residual -> f32)
// MODE 2: outh[p, r] = half(gelu_tanh(dot(W[r], X[p]) + bias[r])) (ViT fc1 -> f16)
// MODE 3: outh[p, r] = half(gelu_exact(dot(W[r], X[p]) + bias[r])) (Merger fc1 -> f16)
// MODE 4: outf[p, r] = dot(W[r], X[p]) + bias[r] + pos_interp[p, r] (ViT patch_embed -> f32)
// Grid (ROWS / 32, nrows / TT), 128 threads (4 simdgroups splitting K in steps of BK = 64).
template<int K, int ROWS, int MODE, int TT>
kernel void k_vit_gemm_T(device half* X [[buffer(0)]],
                         device half* W [[buffer(1)]],
                         device const float* bias [[buffer(2)]],
                         device float* outf [[buffer(3)]],
                         device half* outh [[buffer(4)]],
                         device const float* pos_interp [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NS = 4;
    constexpr int BK = 64;
    constexpr int NSTEP = K / BK / NS;
    const uint r0 = tgp.x * 32;
    const uint t0 = tgp.y * TT;
    threadgroup float Os[NS * TT * 32];

    auto tX = tensor(X + (ulong)t0 * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    auto tW = tensor(W + (ulong)r0 * K, dextents<int, 2>(K, 32), array<int, 2>({1, K}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;

    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = (sg * NSTEP + i) * BK;
        auto a = tX.template slice<BK, TT>(k0, 0);
        auto b = tW.template slice<BK, 32>(k0, 0);
        op.run(a, b, cT);
    }

    auto tO = tensor(Os + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cT.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int w = tid; w < TT * 32; w += 128) {
        const int t = w / 32, r = w % 32;
        float v = bias[r0 + r];
        for (int q = 0; q < NS; ++q) v += Os[q * TT * 32 + w];
        const ulong idx = (ulong)(t0 + t) * ROWS + r0 + r;
        if (MODE == 0) outf[idx] = v;
        else if (MODE == 1) outf[idx] += v;
        else if (MODE == 2) outh[idx] = half(gelu_tanh(v));
        else if (MODE == 3) outh[idx] = half(gelu_exact(v));
        else if (MODE == 4) outf[idx] = v + pos_interp[idx];
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

// Dot product of 64 columns at (row, col64 = c64 * 64) with a lane's xv[64] (and sum X = sum(xv)).
inline float dot64_row(device const wtype* W, device const half2* SB, uint row, uint c64, uint K_cols,
                       thread const float* xv, float X) {
#if FMT == 0
    device const half4* wp = (device const half4*)(W + (ulong)row * K_cols + c64 * 64);
    float acc = 0.f;
    for (int i = 0; i < 16; ++i) {
        acc += dot(float4(wp[i]), float4(xv[4*i], xv[4*i+1], xv[4*i+2], xv[4*i+3]));
    }
    return acc;
#elif FMT == 1
    device const char4* wp = (device const char4*)(W + (ulong)row * K_cols + c64 * 64);
    device const half* sp = (device const half*)SB + (ulong)row * (K_cols / 32) + c64 * 2;
    float d0 = 0.f, d1 = 0.f;
    for (int i = 0; i < 8; ++i) {
        d0 += dot(float4(wp[i]), float4(xv[4*i], xv[4*i+1], xv[4*i+2], xv[4*i+3]));
        d1 += dot(float4(wp[8 + i]), float4(xv[32 + 4*i], xv[32 + 4*i+1], xv[32 + 4*i+2], xv[32 + 4*i+3]));
    }
    return float(sp[0]) * d0 + float(sp[1]) * d1;
#elif FMT == 2
    device const uint4* wp = (device const uint4*)(W + (ulong)row * (K_cols / 2) + c64 * 32);
    const float2 sb = float2(SB[(ulong)row * (K_cols / 64) + c64]);
    const uint4 w0 = wp[0], w1 = wp[1];
    float dq = 0.f;
    for (int j = 0; j < 4; ++j) {
        const uint v0 = w0[j];
        dq += float(v0 & 0xFu) * xv[8*j + 0] + float((v0 >> 4) & 0xFu) * xv[8*j + 1]
            + float((v0 >> 8) & 0xFu) * xv[8*j + 2] + float((v0 >> 12) & 0xFu) * xv[8*j + 3]
            + float((v0 >> 16) & 0xFu) * xv[8*j + 4] + float((v0 >> 20) & 0xFu) * xv[8*j + 5]
            + float((v0 >> 24) & 0xFu) * xv[8*j + 6] + float((v0 >> 28) & 0xFu) * xv[8*j + 7];
        const uint v1 = w1[j];
        dq += float(v1 & 0xFu) * xv[32 + 8*j + 0] + float((v1 >> 4) & 0xFu) * xv[32 + 8*j + 1]
            + float((v1 >> 8) & 0xFu) * xv[32 + 8*j + 2] + float((v1 >> 12) & 0xFu) * xv[32 + 8*j + 3]
            + float((v1 >> 16) & 0xFu) * xv[32 + 8*j + 4] + float((v1 >> 20) & 0xFu) * xv[32 + 8*j + 5]
            + float((v1 >> 24) & 0xFu) * xv[32 + 8*j + 6] + float((v1 >> 28) & 0xFu) * xv[32 + 8*j + 7];
    }
    return sb.x * dq + sb.y * X;
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

// Load lane's 64-column chunk of x (H = 2048 = 32 * 64) and apply RMSNorm(2048, eps=1e-6) * w.
static inline void chunk64_norm(device const float* x, device const float* w, ushort lane,
                                thread float* xv, thread float& X) {
    device const float4* xs = (device const float4*)(x + lane * 64);
    float ss = 0.f;
    for (int i = 0; i < 16; ++i) {
        const float4 t = xs[i];
        xv[4*i] = t.x; xv[4*i+1] = t.y; xv[4*i+2] = t.z; xv[4*i+3] = t.w;
        ss += dot(t, t);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(H) + EPS);
    device const float4* ws = (device const float4*)(w + lane * 64);
    float sum = 0.f;
    for (int i = 0; i < 16; ++i) {
        const float4 wv = ws[i];
        xv[4*i+0] = xv[4*i+0] * scale * wv.x;
        xv[4*i+1] = xv[4*i+1] * scale * wv.y;
        xv[4*i+2] = xv[4*i+2] * scale * wv.z;
        xv[4*i+3] = xv[4*i+3] * scale * wv.w;
        sum += xv[4*i+0] + xv[4*i+1] + xv[4*i+2] + xv[4*i+3];
    }
    X = sum;
}

static inline void chunk64_plain(device const float* x, ushort lane, thread float* xv, thread float& X) {
    device const float4* xs = (device const float4*)(x + lane * 64);
    float sum = 0.f;
    for (int i = 0; i < 16; ++i) {
        const float4 t = xs[i];
        xv[4*i] = t.x; xv[4*i+1] = t.y; xv[4*i+2] = t.z; xv[4*i+3] = t.w;
        sum += t.x + t.y + t.z + t.w;
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
    const int t = tgp.y;
    if (t >= st->n) return;
    threadgroup float hv[128];
    threadgroup float red_h[4];

    float xv[64], X;
    chunk64_norm(x + (ulong)t * H, wnorm, lane, xv, X);

    // Each of the 4 simdgroups computes 32 rows of this 128-row head in 4 batches of 8 rows.
    const uint r0 = head * HD + sg * 32;
    for (int rb = 0; rb < 32; rb += 8) {
        float v[8];
        for (int b = 0; b < 8; ++b) {
            v[b] = dot64_row(W, SB, r0 + rb + b, lane, H, xv, X);
        }
        const float y = reduce8(v, lane);
        if ((lane & 3) == 0) {
            hv[sg * 32 + rb + (lane >> 2)] = y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pos = st->pos + t;
    if (head < 24) {
        // Per-head RMSNorm(128, eps=1e-6) on Q (head 0..15) or K (head 16..23)
        const float val = hv[tid];
        float ss = simd_sum(val * val);
        if (lane == 0) red_h[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float hscale = rsqrt((red_h[0] + red_h[1] + red_h[2] + red_h[3]) / float(HD) + EPS);
        device const float* nw = (head < 16) ? qnorm : knorm;
        hv[tid] = val * hscale * nw[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply 3D M-RoPE (64 frequency pairs over 128 head dims, rotate_half d and d+64)
        if (tid < 64) {
            const float2 cs = rope[pos * 64 + tid];
            const float a = hv[tid], b = hv[tid + 64];
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
        v_dst[tid] = half(hv[tid]);
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
    threadgroup float part[256];
    if (tid < HD) qs[tid] = q[(ulong)t * H + h * HD + tid] * 0.08838834764831845f; // 1 / sqrt(128)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float4* qs4 = (const threadgroup float4*)qs;
    device const half* kb = kc + (ulong)g * st->maxctx * HD;
    device const half* vb = vc + (ulong)g * st->maxctx * HD;
    float m = -INFINITY;
    for (int s = tid; s < L; s += 256) {
        device const half4* kr = (device const half4*)(kb + (ulong)s * HD);
        float a = 0.f;
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
    const int d = tid & 127;
    const int p = tid >> 7;
    float o = 0.f;
    for (int s = p; s < L; s += 2) {
        o += sc[s] * float(vb[(ulong)s * HD + d]);
    }
    part[tid] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HD) {
        out[(ulong)t * H + h * HD + tid] = (part[tid] + part[tid + 128]) / sum;
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
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    float xv[64], X;
    chunk64_plain(a + (ulong)t * H, lane, xv, X);
    const uint r0 = (tgp.x * 2 + sg) * 8;
    float v[8];
    for (int b = 0; b < 8; ++b) {
        v[b] = dot64_row(W, SB, r0 + b, lane, H, xv, X);
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
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    float xv[64], X;
    chunk64_norm(x + (ulong)t * H, wnorm, lane, xv, X);
    const uint u0 = (tgp.x * 2 + sg) * 4;
    float v[8];
    for (int b = 0; b < 4; ++b) {
        v[b] = dot64_row(W, SB, u0 + b, lane, H, xv, X);
        v[4 + b] = dot64_row(W, SB, FF + u0 + b, lane, H, xv, X);
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
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    const uint r0 = (tgp.x * 2 + sg) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int chunk = 0; chunk < 3; ++chunk) {
        float xv[64], X;
        const uint c64 = chunk * 32 + lane;
        chunk64_plain(hbuf + (ulong)t * FF + chunk * 2048, lane, xv, X);
        for (int b = 0; b < 8; ++b) {
            acc[b] += dot64_row(W, SB, r0 + b, c64, FF, xv, X);
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
    float xv[64], X;
    device const float* xr = x + (ulong)(st->n - 1) * H;
    chunk64_norm(xr, wnorm, lane, xv, X);
    const uint r0 = tg * 32 + sg * 8;
    float v[8];
    for (int b = 0; b < 8; ++b) {
        v[b] = dot64_row(E, ESB, r0 + b, lane, H, xv, X);
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

// Greedy commit + embed: reduces the 4748 lm_head partial maxima to the next token,
// appends it to out_tokens, updates M-RoPE position for the next decode step, and embeds it into x[0..H-1].
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
    threadgroup int tok_s;
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
            out_tokens[st->step] = tok;
            st->step += 1;
            st->pos += st->n;
            st->n = 1;
            st->vis_offset = -1;
            const bool stop = tok == st->eos0 || tok == st->eos1 || st->step >= st->maxsteps;
            if (stop) st->done = 1;
            tok_s = stop ? -1 : tok;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tok = tok_s;
    if (tok < 0) return;
    for (uint i = tid; i < H; i += 1024) x[i] = embed_at(E, ESB, tok, i);
}

kernel void k_noop(device int* sink [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0xFFFFFFFF) sink[0] = 1;
}
"""#
