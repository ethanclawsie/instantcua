# instantcua: shared spec

Goal: the lowest end-to-end latency for local visual computer-use (GUI grounding and action
generation) with **Qwen3-VL-2B-Instruct** on a **base Apple M5** (10-core GPU with Neural
Accelerators, 4P+6E CPU, 24 GB, ~153 GB/s unified memory, macOS 26.3). Single-model,
single-machine engine: Swift host code with zero external dependencies, hand-fused Metal 4 /
Neural Accelerator (`matmul2d`) kernels for both the Vision Transformer (ViT + DeepStack mergers)
and the M-RoPE Language Model, zero-copy GPU screen capture, and native `CGEvent` control.

## Environment facts

- Swift 6.2+ Command Line Tools. Metal shaders are compiled at runtime from source
  (`MTLDevice.makeLibrary(source:options:)`), targeting MSL 4.0 and
  `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`.
- Python for offline reference/conversion/benchmark tooling only (`models/`, `tools/reference/`,
  `bench/`). Always invoke Python with `-I`.
- Competitors:
  - `llama.cpp` 0.6.0 (`libllama` + `libmtmd` Metal backend) from `/opt/homebrew/opt/llama.cpp`
  - `mlx-vlm` (Apple MLX backend)
- GPU timing etiquette: wrap every GPU timing run in `lockf -k /tmp/instantllm-gpu.lock <cmd>`.

## Model facts (`Qwen3VLForConditionalGeneration` — `Qwen/Qwen3-VL-2B-Instruct`)

### 1. Vision Encoder (`Qwen3VLVisionModel`)

| Parameter | Value |
|---|---|
| `depth` | 24 blocks (`0..23`) |
| `hidden_size` | 1024 |
| `intermediate_size` | 4096 (`gelu_pytorch_tanh`) |
| `num_heads` | 16 (`head_dim = 64`, bidirectional attention) |
| `patch_size` | 16 (`temporal_patch_size = 2`, `in_channels = 3`) |
| `spatial_merge_size` | 2 ($2\times 2$ blocks of $16\times 16$ patches $\to$ $32\times 32$ px per visual token) |
| `num_position_embeddings` | 2304 ($48\times 48$ learned grid, bilinear `align_corners=True`) |
| `rotary_pos_emb` | 2D axial RoPE (`theta = 10000.0`, `spatial_dim = 32`) |
| `deepstack_visual_indexes` | `[5, 11, 17]` (post-shuffle `LayerNorm(4096)` $\to$ `FC1(4096,4096)+GELU` $\to$ `FC2(4096,2048)`) |
| `merger` (after block 23) | pre-shuffle `LayerNorm(1024)` $\to$ $2\times 2$ view `4096` $\to$ `FC1(4096,4096)+GELU` $\to$ `FC2(4096,2048)` |
| Pixel normalization | `mean = [0.5, 0.5, 0.5]`, `std = [0.5, 0.5, 0.5]` $\implies x = \text{pixel} \times (2/255) - 1$ |

### 2. Language Backbone (`Qwen3VLTextModel`)

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 28 (`0..27`) |
| `hidden_size` | 2048 |
| `intermediate_size` | 6144 (SwiGLU: `down(silu(gate(x)) * up(x))`) |
| `num_attention_heads` | 16 Q heads, 8 KV heads (GQA 2:1, `head_dim = 128`, no QKV/O bias) |
| `q_norm` / `k_norm` | Per-head `RMSNorm(128, eps=1e-6)` applied to Q and K before RoPE |
| `rope_theta` | `5,000,000.0`, interleaved 3D M-RoPE (`mrope_section = [24, 20, 20]`) |
| `deepstack` injection | After decoder layers `0, 1, 2`, add `deepstack_embeds[layer_idx]` to visual token positions |
| `vocab_size` | 151936 (tied `lm_head = embed_tokens`) |
| Special tokens | `<|vision_start|>` 151652, `<|vision_end|>` 151653, `<|image_pad|>` 151655, `<|im_end|>` 151645, `<|endoftext|>` 151643 |

## GUI Prompt & Action Protocol

Canonical benchmark resolution: `640 x 384` (`grid_thw = [1, 24, 40]`, `960` ViT patches,
`12 x 20 = 240` merged visual tokens entering the LLM; exact multiples of `32`). Engine also
supports dynamic `(W, H)` multiples of `32`.

Chat template (exact):

```
<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>...(240 times)...<|vision_end|>{PROMPT}<|im_end|>
<|im_start|>assistant
```

Actions output coordinates in Qwen3-VL's native `0..1000` normalized screen space:
- `click(x, y)`
- `double_click(x, y)`
- `right_click(x, y)`
- `move(x, y)`
- `drag(x1, y1, x2, y2)`
- `type("text")`
- `key("shortcut")`
- `scroll(dx, dy, x, y)`
- `wait(ms)`
- `done()`
