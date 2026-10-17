# instantcua

Ultra-low-latency local macOS computer-use agent powered by **Qwen3-VL-2B-Instruct** running on a
custom, zero-dependency **Swift + Metal 4 (Apple M5 Neural Accelerators)** vision-language engine.

Built for the base **Apple M5** (10-core GPU with Neural Accelerators, 24 GB unified memory, macOS
26.3):
- **Custom Metal 4 Vision Encoder (`Qwen3VLVisionModel`):** 24-layer ViT + DeepStack multi-scale
  mergers + 2×2 spatial patch merger with GPU-native patchification (`IOSurface` / `CGImage` $\to$
  normalized patches) and temporal screen-delta caching.
- **Custom Metal 4 Language Backbone (`Qwen3VLTextModel`):** 28-layer GQA (16:8) decoder with
  per-head QK RMSNorm, interleaved 3D M-RoPE, DeepStack visual injection, and GPU-resident greedy /
  speculative action decoding.
- **Native macOS Actuator (`CGEvent` + `ScreenCaptureKit`):** Direct sub-millisecond mouse and
  keyboard control on the real display with a hardware `Esc` kill switch via `CGEventTap`.

See [`docs/SPEC.md`](docs/SPEC.md) for architecture details and benchmark protocol.
