#!/usr/bin/env python3
"""
Exact float32 reference implementation of Qwen3-VL-2B-Instruct (Vision Encoder + DeepStack + M-RoPE LLM).
Used to generate golden reference tensors and greedy outputs for verifying the custom Metal 4 engine.
"""
import json
import math
import os
import struct
import sys
import time
import mlx.core as mx
import numpy as np
from PIL import Image
from safetensors import safe_open
from tokenizers import Tokenizer as HFTokenizer

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
IM_END = 151645
ENDOFTEXT = 151643


def load_image_patches(image_path: str, patch_size: int = 16, merge_size: int = 2):
    """Load RGB image, normalize with mean=0.5, std=0.5, and extract patches in 2x2 merge-block order."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) * (2.0 / 255.0) - 1.0  # [H, W, 3]
    H, W, C = arr.shape
    assert H % (patch_size * merge_size) == 0 and W % (patch_size * merge_size) == 0
    grid_h = H // patch_size
    grid_w = W // patch_size
    # Transpose to [C, H, W] -> reshape [C, grid_h//2, 2, 16, grid_w//2, 2, 16]
    chw = np.transpose(arr, (2, 0, 1))
    patches = chw.reshape(C, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size)
    # Permute to [grid_h//2, grid_w//2, 2, 2, C, 16, 16]
    patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6)).reshape(grid_h * grid_w, C * patch_size * patch_size)
    return patches, grid_h, grid_w


def compute_vision_pos_embed(pos_embed_weight: np.ndarray, grid_h: int, grid_w: int, merge_size: int = 2) -> np.ndarray:
    """Bilinear interpolation (align_corners=True) of [2304, 1024] (48x48) pos_embed table in merge-block order."""
    side = 48
    table = pos_embed_weight.reshape(side, side, -1)
    n_patches = grid_h * grid_w
    within = np.arange(n_patches)
    blocks_w = grid_w // merge_size
    in_col = within % merge_size
    in_row = (within // merge_size) % merge_size
    block_col = (within // (merge_size * merge_size)) % blocks_w
    block_row = within // (merge_size * merge_size * blocks_w)
    row = (block_row * merge_size + in_row).astype(np.float32)
    col = (block_col * merge_size + in_col).astype(np.float32)

    src_h = row * (side - 1) / max(grid_h - 1, 1)
    src_w = col * (side - 1) / max(grid_w - 1, 1)
    h0 = np.floor(src_h).astype(np.int32)
    w0 = np.floor(src_w).astype(np.int32)
    h1 = np.minimum(h0 + 1, side - 1)
    w1 = np.minimum(w0 + 1, side - 1)
    dh = (src_h - h0)[:, None]
    dw = (src_w - w0)[:, None]

    v00 = table[h0, w0]
    v01 = table[h0, w1]
    v10 = table[h1, w0]
    v11 = table[h1, w1]
    return (1.0 - dh) * ((1.0 - dw) * v00 + dw * v01) + dh * ((1.0 - dw) * v10 + dw * v11)


def compute_vision_rope(grid_h: int, grid_w: int, merge_size: int = 2, head_dim: int = 64, theta: float = 10000.0):
    """2D axial RoPE cos/sin of shape [N_patches, 1, head_dim] in float32."""
    spatial_dim = head_dim // 2  # 32
    inv_freq = 1.0 / (theta ** (np.arange(0, spatial_dim, 2, dtype=np.float32) / spatial_dim))  # [16]

    hpos, wpos = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")
    hpos = hpos.reshape(grid_h // merge_size, merge_size, grid_w // merge_size, merge_size).transpose(0, 2, 1, 3).reshape(-1)
    wpos = wpos.reshape(grid_h // merge_size, merge_size, grid_w // merge_size, merge_size).transpose(0, 2, 1, 3).reshape(-1)

    freq_h = hpos[:, None] * inv_freq[None, :]  # [N, 16]
    freq_w = wpos[:, None] * inv_freq[None, :]  # [N, 16]
    freq_hw = np.concatenate([freq_h, freq_w], axis=-1)  # [N, 32]
    angles = np.concatenate([freq_hw, freq_hw], axis=-1)  # [N, 64]
    return np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]


def compute_mrope_cos_sin(seq_len: int, vis_offset: int, llm_h: int, llm_w: int, head_dim: int = 128, theta: float = 5000000.0):
    """Compute 3D interleaved M-RoPE cos/sin [seq_len, 1, head_dim] and rope_delta for Qwen3-VL."""
    n_vis = llm_h * llm_w
    pos_ids = np.zeros((3, seq_len), dtype=np.float32)
    # Prefix text
    for i in range(vis_offset):
        pos_ids[:, i] = float(i)
    # Vision tokens
    for r in range(llm_h):
        for c in range(llm_w):
            idx = vis_offset + r * llm_w + c
            pos_ids[0, idx] = float(vis_offset)
            pos_ids[1, idx] = float(vis_offset + r)
            pos_ids[2, idx] = float(vis_offset + c)
    # Suffix text
    after_pos = vis_offset + max(llm_h, llm_w)
    for i in range(vis_offset + n_vis, seq_len):
        p = float(after_pos + (i - (vis_offset + n_vis)))
        pos_ids[:, i] = p
    rope_delta = int(after_pos - (vis_offset + n_vis))

    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))  # [64]
    freq_t = pos_ids[0, :, None] * inv_freq[None, :]
    freq_h = pos_ids[1, :, None] * inv_freq[None, :]
    freq_w = pos_ids[2, :, None] * inv_freq[None, :]

    # Interleaved mrope_section = [24, 20, 20] -> H at 1:60:3, W at 2:60:3
    freqs = freq_t.copy()
    freqs[:, 1:60:3] = freq_h[:, 1:60:3]
    freqs[:, 2:60:3] = freq_w[:, 2:60:3]
    angles = np.concatenate([freqs, freqs], axis=-1)  # [seq_len, 128]
    return np.cos(angles)[:, None, :], np.sin(angles)[:, None, :], rope_delta


def compute_decode_rope_cos_sin(pos: int, head_dim: int = 128, theta: float = 5000000.0):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = float(pos) * inv_freq
    angles = np.concatenate([freqs, freqs], axis=-1)[None, None, :]
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


class Qwen3VLRef:
    def __init__(self, model_dir: str):
        st_path = os.path.join(model_dir, "model.safetensors")
        self.tok = HFTokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        raw_w = mx.load(st_path)
        self.w = {k: v.astype(mx.float32) for k, v in raw_w.items()}
        # Pre-fold temporal patch conv weight: [1024, 3, 2, 16, 16] -> sum over temporal axis -> [1024, 768]
        pw = self.w["model.visual.patch_embed.proj.weight"]
        self.patch_w = (pw[:, :, 0, :, :] + pw[:, :, 1, :, :]).reshape(1024, 768)
        self.patch_b = self.w["model.visual.patch_embed.proj.bias"]
        self.pos_embed_np = np.array(self.w["model.visual.pos_embed.weight"])
        mx.eval(self.patch_w, self.patch_b)

    @staticmethod
    def layernorm(x: mx.array, w: mx.array, b: mx.array, eps: float = 1e-6) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + eps) * w + b

    @staticmethod
    def rmsnorm(x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
        var = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(var + eps) * w

    @staticmethod
    def rotate_half(x: mx.array) -> mx.array:
        d = x.shape[-1] // 2
        return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)

    @staticmethod
    def gelu_tanh(x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x * x * x))))

    @staticmethod
    def gelu_exact(x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))

    def encode_vision(self, patches_np: np.ndarray, grid_h: int, grid_w: int):
        n_patches = grid_h * grid_w
        n_vis = n_patches // 4
        pos_np = compute_vision_pos_embed(self.pos_embed_np, grid_h, grid_w)
        cos_np, sin_np = compute_vision_rope(grid_h, grid_w)

        x = mx.array(patches_np) @ self.patch_w.T + self.patch_b + mx.array(pos_np)
        cos = mx.array(cos_np)
        sin = mx.array(sin_np)

        deepstack_indexes = [5, 11, 17]
        deepstack_outs = []

        for l in range(24):
            p = f"model.visual.blocks.{l}."
            h = self.layernorm(x, self.w[p + "norm1.weight"], self.w[p + "norm1.bias"])
            qkv = h @ self.w[p + "attn.qkv.weight"].T + self.w[p + "attn.qkv.bias"]
            qkv = qkv.reshape(n_patches, 3, 16, 64)
            q = qkv[:, 0] * cos + self.rotate_half(qkv[:, 0]) * sin
            k = qkv[:, 1] * cos + self.rotate_half(qkv[:, 1]) * sin
            v = qkv[:, 2]

            # [16, N, 64]
            qt = mx.transpose(q, (1, 0, 2))
            kt = mx.transpose(k, (1, 0, 2))
            vt = mx.transpose(v, (1, 0, 2))
            scores = (qt @ mx.transpose(kt, (0, 2, 1))) * (64.0 ** -0.5)
            probs = mx.softmax(scores, axis=-1)
            attn_out = mx.transpose(probs @ vt, (1, 0, 2)).reshape(n_patches, 1024)
            x = x + (attn_out @ self.w[p + "attn.proj.weight"].T + self.w[p + "attn.proj.bias"])

            h2 = self.layernorm(x, self.w[p + "norm2.weight"], self.w[p + "norm2.bias"])
            fc1 = self.gelu_tanh(h2 @ self.w[p + "mlp.linear_fc1.weight"].T + self.w[p + "mlp.linear_fc1.bias"])
            fc2 = fc1 @ self.w[p + "mlp.linear_fc2.weight"].T + self.w[p + "mlp.linear_fc2.bias"]
            x = x + fc2

            if l in deepstack_indexes:
                ds_idx = deepstack_indexes.index(l)
                dp = f"model.visual.deepstack_merger_list.{ds_idx}."
                # postshuffle norm on 4096
                xm = self.layernorm(x.reshape(n_vis, 4096), self.w[dp + "norm.weight"], self.w[dp + "norm.bias"])
                xm = self.gelu_exact(xm @ self.w[dp + "linear_fc1.weight"].T + self.w[dp + "linear_fc1.bias"])
                xm = xm @ self.w[dp + "linear_fc2.weight"].T + self.w[dp + "linear_fc2.bias"]
                deepstack_outs.append(xm)

        # Final merger (preshuffle norm on 1024)
        mp = "model.visual.merger."
        xm = self.layernorm(x, self.w[mp + "norm.weight"], self.w[mp + "norm.bias"]).reshape(n_vis, 4096)
        xm = self.gelu_exact(xm @ self.w[mp + "linear_fc1.weight"].T + self.w[mp + "linear_fc1.bias"])
        pooler = xm @ self.w[mp + "linear_fc2.weight"].T + self.w[mp + "linear_fc2.bias"]
        mx.eval(pooler, *deepstack_outs)
        return pooler, deepstack_outs

    def generate(self, system_prompt: str, user_prompt: str, image_path: str, max_new_tokens: int = 32):
        patches_np, grid_h, grid_w = load_image_patches(image_path)
        llm_h, llm_w = grid_h // 2, grid_w // 2
        n_vis = llm_h * llm_w

        prefix_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|>"
        suffix_str = f"<|vision_end|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = self.tok.encode(prefix_str, add_special_tokens=False).ids
        suffix_ids = self.tok.encode(suffix_str, add_special_tokens=False).ids
        vis_offset = len(prefix_ids)
        input_ids = prefix_ids + [IMAGE_PAD] * n_vis + suffix_ids
        seq_len = len(input_ids)

        pooler, deepstack_outs = self.encode_vision(patches_np, grid_h, grid_w)

        emb_w = self.w["model.language_model.embed_tokens.weight"]
        tok_embeds = emb_w[mx.array(input_ids, dtype=mx.int32)]
        # Scatter pooler into image token positions
        parts = [tok_embeds[:vis_offset], pooler, tok_embeds[vis_offset + n_vis :]]
        x = mx.concatenate(parts, axis=0)

        cos_np, sin_np, rope_delta = compute_mrope_cos_sin(seq_len, vis_offset, llm_h, llm_w)
        cos = mx.array(cos_np)
        sin = mx.array(sin_np)

        k_caches = []
        v_caches = []
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9, dtype=mx.float32), k=1)

        for l in range(28):
            lp = f"model.language_model.layers.{l}."
            h = self.rmsnorm(x, self.w[lp + "input_layernorm.weight"])
            q = (h @ self.w[lp + "self_attn.q_proj.weight"].T).reshape(seq_len, 16, 128)
            k = (h @ self.w[lp + "self_attn.k_proj.weight"].T).reshape(seq_len, 8, 128)
            v = (h @ self.w[lp + "self_attn.v_proj.weight"].T).reshape(seq_len, 8, 128)

            q = self.rmsnorm(q, self.w[lp + "self_attn.q_norm.weight"])
            k = self.rmsnorm(k, self.w[lp + "self_attn.k_norm.weight"])

            q = q * cos + self.rotate_half(q) * sin
            k = k * cos + self.rotate_half(k) * sin
            k_caches.append(k)
            v_caches.append(v)

            # GQA 16:8 -> repeat k, v by 2
            kr = mx.repeat(k, 2, axis=1)
            vr = mx.repeat(v, 2, axis=1)
            qt = mx.transpose(q, (1, 0, 2))
            kt = mx.transpose(kr, (1, 0, 2))
            vt = mx.transpose(vr, (1, 0, 2))
            scores = (qt @ mx.transpose(kt, (0, 2, 1))) * (128.0 ** -0.5) + mask
            probs = mx.softmax(scores, axis=-1)
            attn_out = mx.transpose(probs @ vt, (1, 0, 2)).reshape(seq_len, 2048)
            x = x + (attn_out @ self.w[lp + "self_attn.o_proj.weight"].T)

            h2 = self.rmsnorm(x, self.w[lp + "post_attention_layernorm.weight"])
            gate = h2 @ self.w[lp + "mlp.gate_proj.weight"].T
            up = h2 @ self.w[lp + "mlp.up_proj.weight"].T
            act = (gate * mx.sigmoid(gate)) * up
            x = x + (act @ self.w[lp + "mlp.down_proj.weight"].T)

            if l < len(deepstack_outs):
                x = mx.concatenate(
                    [x[:vis_offset], x[vis_offset : vis_offset + n_vis] + deepstack_outs[l], x[vis_offset + n_vis :]],
                    axis=0,
                )

        last_h = self.rmsnorm(x[-1:], self.w["model.language_model.norm.weight"])
        first_logits = (last_h @ emb_w.T)[0]
        mx.eval(first_logits)
        first_logits_np = np.array(first_logits)

        gen_tokens = []
        next_tok = int(np.argmax(first_logits_np))
        gen_tokens.append(next_tok)

        cur_len = seq_len
        while len(gen_tokens) < max_new_tokens and next_tok not in (IM_END, ENDOFTEXT):
            x_step = emb_w[mx.array([next_tok], dtype=mx.int32)]  # [1, 2048]
            d_cos, d_sin = compute_decode_rope_cos_sin(cur_len + rope_delta)
            for l in range(28):
                lp = f"model.language_model.layers.{l}."
                h = self.rmsnorm(x_step, self.w[lp + "input_layernorm.weight"])
                q = (h @ self.w[lp + "self_attn.q_proj.weight"].T).reshape(1, 16, 128)
                k = (h @ self.w[lp + "self_attn.k_proj.weight"].T).reshape(1, 8, 128)
                v = (h @ self.w[lp + "self_attn.v_proj.weight"].T).reshape(1, 8, 128)

                q = self.rmsnorm(q, self.w[lp + "self_attn.q_norm.weight"])
                k = self.rmsnorm(k, self.w[lp + "self_attn.k_norm.weight"])
                q = q * d_cos + self.rotate_half(q) * d_sin
                k = k * d_cos + self.rotate_half(k) * d_sin

                k_caches[l] = mx.concatenate([k_caches[l], k], axis=0)
                v_caches[l] = mx.concatenate([v_caches[l], v], axis=0)

                kr = mx.repeat(k_caches[l], 2, axis=1)
                vr = mx.repeat(v_caches[l], 2, axis=1)
                qt = mx.transpose(q, (1, 0, 2))
                kt = mx.transpose(kr, (1, 0, 2))
                vt = mx.transpose(vr, (1, 0, 2))
                scores = (qt @ mx.transpose(kt, (0, 2, 1))) * (128.0 ** -0.5)
                probs = mx.softmax(scores, axis=-1)
                attn_out = mx.transpose(probs @ vt, (1, 0, 2)).reshape(1, 2048)
                x_step = x_step + (attn_out @ self.w[lp + "self_attn.o_proj.weight"].T)

                h2 = self.rmsnorm(x_step, self.w[lp + "post_attention_layernorm.weight"])
                gate = h2 @ self.w[lp + "mlp.gate_proj.weight"].T
                up = h2 @ self.w[lp + "mlp.up_proj.weight"].T
                act = (gate * mx.sigmoid(gate)) * up
                x_step = x_step + (act @ self.w[lp + "mlp.down_proj.weight"].T)

            last_h = self.rmsnorm(x_step, self.w["model.language_model.norm.weight"])
            logits = (last_h @ emb_w.T)[0]
            mx.eval(logits)
            next_tok = int(mx.argmax(logits).item())
            if next_tok in (IM_END, ENDOFTEXT):
                break
            gen_tokens.append(next_tok)
            cur_len += 1

        text = self.tok.decode(gen_tokens, skip_special_tokens=True)
        return {
            "input_ids": input_ids,
            "vis_offset": vis_offset,
            "rope_delta": rope_delta,
            "pooler": np.array(pooler),
            "deepstack": [np.array(d) for d in deepstack_outs],
            "first_logits": first_logits_np,
            "gen_tokens": gen_tokens,
            "text": text,
        }


def main():
    model_dir = "models/hf"
    with open("bench/system.txt") as f:
        system_prompt = f.read().strip()
    with open("bench/tasks.json") as f:
        tasks = json.load(f)

    os.makedirs("scratch/golden", exist_ok=True)
    ref = Qwen3VLRef(model_dir)

    summary = []
    for idx, t in enumerate(tasks):
        t0 = time.perf_counter()
        out = ref.generate(system_prompt, t["prompt"], t["image"], max_new_tokens=32)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[{t['id']}] ({dt:.1f} ms) -> {out['text']!r} (expected box: {t.get('targetBox1000')})")

        if idx == 0:
            np.save("scratch/golden/task0_pooler.npy", out["pooler"])
            for di, d in enumerate(out["deepstack"]):
                np.save(f"scratch/golden/task0_deepstack_{di}.npy", d)
            np.save("scratch/golden/task0_first_logits.npy", out["first_logits"])

        summary.append({
            "id": t["id"],
            "image": t["image"],
            "prompt": t["prompt"],
            "targetBox1000": t.get("targetBox1000"),
            "input_ids": out["input_ids"],
            "vis_offset": out["vis_offset"],
            "rope_delta": out["rope_delta"],
            "gen_tokens": out["gen_tokens"],
            "text": out["text"],
        })

    with open("scratch/golden/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved golden reference data to scratch/golden/")


if __name__ == "__main__":
    main()
