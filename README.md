# Gufork: The Open Source Strix Halo AI Engine (Fork of Gufo)

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufork logo" width="180">
</p>

> [!NOTE]
> **Fork Notice & Upstream Credits**: Gufork is a specialized community fork of **[Gufo](https://github.com/gufo-org/gufo)** (`https://github.com/gufo-org/gufo`).
> Full credit for the core engine architecture, memory management, server pipeline, and baseline Strix Halo (`gfx1151`) optimizations belongs to the [upstream Gufo project and its contributors](https://github.com/gufo-org/gufo). Gufork extends upstream Gufo with native support for ROCmFPX / ROCmFP4 quantization formats and Ornith-1.5-122B hybrid MoE model execution.

**Gufork** is a high-performance local inference engine for Linux and Windows, specifically engineered and optimized for AMD Strix Halo hardware: **Ryzen AI MAX+ 395** systems with Radeon 8060S (`gfx1151`) and up to 128 GiB of unified memory.

Binaries are built as **`gufork.exe`** (with convenience aliases **`gurfork.exe`** and **`gufo.exe`**) and **`gufork-server.exe`** (alias **`gufo-server.exe`**).
---

## Key Features

- **Native ROCmFPX & ROCmFP4 Quantization Support**: Full dequantizer and HIP kernel pipeline for AMD-optimized low-bit formats (`Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST` / `STRIX`, `Q2_0_ROCMFPX`, `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX`, and `Q4_0_ROCMI4`).
- **Ornith-1.5-122B-ROCmFP4-Strix Support**: Native execution of [Harsha001/Ornith-1.5-122B-ROCmFPX](https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX) (122B total parameters, 10B active parameters - 122B-A10B) on AMD Strix Halo:
  - **65 GB Strix Recipe (`Q4_0_ROCMFP4_STRIX`)** and **80 GB Balanced Recipe (`Q4_0_ROCMFP4_BALANCED`)** with auto-loading multi-part GGUFs.
  - **F16 Vision Projector Sidecar (`mmproj`)**: Multimodal image understanding supported out of the box.
  - **Hybrid Architecture**: 48 layers combining 12 Gated GQA Full Attention layers and 36 Gated DeltaNet SSM Linear Recurrence layers.
  - **Grouped MoE**: 256 routing experts (8 active per token) plus shared expert, accelerated with contiguous coalesced kernels.
  - **Dual-Stream Execution**: Concurrent Compute 0 + 3D memory overlap for shared expert and SSM computation.
  - **Tensor-Core WMMA**: Accelerated prefill GEMMs and tiled attention routines.
- **Batteries-Included Modalities**: Text LLMs, text-to-image (Qwen-Image-2.1), text-to-video (MiniMax H3), speech synthesis (Qwen3-TTS 1.7B), and speech recognition (Qwen3-ASR 1.7B).
- **OpenAI & Anthropic Compatible Server**: Full streaming, function calling, tool use, structured output, and vision inputs.

---

## Supported Models

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Architecture / Mode | Quantization / Weights | Hardware & Performance Highlights |
|---|---|---|---|
| **[Ornith-1.5-122B](https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX)** | Hybrid Gated GQA + DeltaNet SSM + 256-Expert MoE (122B-A10B, 48 layers) | [Harsha001/Ornith-1.5-122B-ROCmFPX](https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX): `Q4_0_ROCMFP4_STRIX` (65 GB), `Q4_0_ROCMFP4_BALANCED` (80 GB), `mmproj-f16` (0.9 GB) | **Fits on 96 GB Strix Halo VRAM**; dual-stream shared expert + SSM overlap; contiguous grouped MoE kernels; WMMA tensor-core prefill GEMMs; 4K chunked prefill; up to 256K context. |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2 | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) · [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) | Q4: **648.89 tok/s pp**; up to **71.55 tok/s tg** single user and **128.42 aggregated tok/s** on 8 concurrent requests with DFlash2. |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Q4, images, AR, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | **1,801.01 tok/s pp** (AR); up to **116.93 tok/s tg** single user and **245.81 aggregated tok/s** on 8 concurrent requests with MTP. |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [antirez Flash 0731 IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf) · [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf) | **484.62 tok/s pp**; up to **26.62 tok/s tg** single user and **54.74 aggregated tok/s** on 8 concurrent requests with DSpark. |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | Speech recognition | [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | **15.27× realtime** audio transcription. |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Text and image embeddings | [BF16](https://huggingface.co/google/embeddinggemma-2) | Up to **41.5k tokens/s** over HTTP; **13.0 ms** for a 404-token request. |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | Speech synthesis & voice cloning | BF16 [CustomVoice / VoiceDesign / Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Up to **2.54× realtime**; **201 ms** to first audio. |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation & editing | [Complete pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1) | Native DiT image generation on gfx1151. |
| [MiniMax H3](docs/models/minimax-h3/README.md) | BF16 text to video/audio | [FL2VA pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Native DiT video synthesis on gfx1151. |

---

## ROCmFPX & ROCmFP4 Quantization Architecture

ROCmFPX is an AMD-focused low-bit quantization family tailored to maximize bandwidth and compute utilization on AMD RDNA 3.5 (`gfx1151`) architectures.

### Block Formats & Memory Layouts

| GGML Type ID | Format Name | Quant Block Size ($QK$) | Encoded Block Size | Description & Scaling |
|---|---|---|---|---|
| **100** | `Q4_0_ROCMFP4` | 32 elements | 18 bytes | Dual-scale layout: 16 bytes packed 4-bit Codebook10 codes + two UE4M3 scale bytes. |
| **101** | `Q4_0_ROCMFP4_FAST` (`_STRIX`) | 32 elements | 17 bytes | Fast single-scale layout: 16 bytes packed 4-bit Codebook10 codes + one UE4M3 scale byte. |
| **102** | `Q6_0_ROCMFPX` | 32 elements | 26 bytes | 24 bytes packed 6-bit codes + two UE4M3 scale bytes. |
| **103** | `Q8_0_ROCMFPX` | 32 elements | 33 bytes | 32 int8 codes + one UE4M3 scale byte. |
| **104** | `Q3_0_ROCMFPX` | 32 elements | 14 bytes | 12 bytes packed 3-bit codes + two UE4M3 scale bytes. |
| **107** | `Q2_0_ROCMFPX` | 32 elements | 10 bytes | 8 bytes packed 2-bit codes + two UE4M3 scale bytes. |
| **108** | `Q4_0_ROCMI4` | 32 elements | 17 bytes | 16 bytes packed two's-complement signed nibbles + one UE4M3 scale byte. |

### Hardware-Accelerated Expansion

Gufork executes ROCmFP4 unpack and dot-product kernels directly on `gfx1151` vector units using the AMD GCN byte-permute intrinsic:
```cpp
__builtin_amdgcn_perm(v_high, v_low, mask)
```
This expands packed nibbles into int32 operands at line-rate without table lookups, eliminating memory stalls in the GEMV/MMVQ and WMMA paths.

---

## Ornith-1.5-122B on Strix Halo

Ornith-1.5-122B is a 122-billion parameter model (10B active parameters per token) utilizing a cutting-edge hybrid architecture:
- **48 Total Layers**: 12 Full Attention layers (every 4th layer, using Gated Grouped Query Attention) and 36 Linear Attention layers (using Gated DeltaNet State Space Model recurrence).
- **256 Routing Experts + 1 Shared Expert**: 8 experts activated per token.
- **Up to 262,144 Context Length**: Full 256k sequence support with chunked prefill (default chunk size 4096).
- **Multimodal Support**: High-resolution vision inputs via the paired F16 vision projector (`mmproj`).

### VRAM Configuration for 128 GiB Strix Halo

On a 128 GB unified memory AMD Strix Halo system (such as Ryzen AI MAX+ 395):
- **Set GPU Memory to 96 GB**:
  - **Windows**: Select 96 GB dedicated VRAM in your BIOS / UEFI settings.
  - **Linux**: Configure the GTT/TTM kernel parameters (`amdgpu.gttsize=98304`).
- **Memory Footprint**:
  - `Q4_0_ROCMFP4_STRIX`: **65 GB** model weights + ~0.9 GB vision projector. Leaves ~30 GB VRAM for long context windows and concurrent requests.
  - `Q4_0_ROCMFP4_BALANCED`: **80 GB** model weights + ~0.9 GB vision projector. Leaves ~15 GB VRAM for context.

### Downloading the Weights

Download either the **Strix** or **Balanced** variant from Hugging Face:

```bash
# Strix-tuned 65 GB variant (Recommended for speed and memory headroom)
hf download Harsha001/Ornith-1.5-122B-ROCmFPX \
  --include "*Q4_0_STRIX-*" "mmproj*" \
  --local-dir models/Ornith-1.5-122B

# Or the Balanced 80 GB variant
hf download Harsha001/Ornith-1.5-122B-ROCmFPX \
  --include "*Q4_0_BALANCED-*" "mmproj*" \
  --local-dir models/Ornith-1.5-122B
```

---

## Quickstart & Running Gufork

### 1. Execute One-Off Prompts (`gufork prompt` / `gurfork.exe`)

Point Gufork to the first shard; it automatically discovers and streams all remaining shards:

```bash
# Windows
gufork.exe prompt -m models/Ornith-1.5-122B/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf \
  -p "Explain the difference between Full Attention and DeltaNet SSM in detail." \
  --think on -v

# Linux
./gufork prompt -m models/Ornith-1.5-122B/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf \
  -p "Hello Ornith! What is your architecture?"
```

### 2. Interactive Terminal Conversation (`gufork chat`)

```bash
gufork.exe chat -m models/Ornith-1.5-122B/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf
```

### 3. Start the OpenAI-Compatible API Server (`gufork serve llm` / `gufork-server`)

You can launch the server either via subcommand or via the dedicated server executable:

```bash
# Using gufork serve
gufork.exe serve llm \
  --model models/Ornith-1.5-122B/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf \
  --vision-model models/Ornith-1.5-122B/mmproj-Ornith-1.5-122B-f16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --context 32768

# Or directly using gufork-server:
gufork-server.exe llm \
  --model models/Ornith-1.5-122B/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port 8080
```

### 4. Query the API

Send OpenAI-compatible chat requests:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a fast HIP kernel for vector addition."}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

---

## Prefill & Kernel Verification Harness

Gufork includes a specialized verification harness for Ornith-1.5-122B prefill timings, memory checks, and parity:

```powershell
# Run the prefill verification suite
powershell -ExecutionPolicy Bypass -File tools\verify\ornith_prefill_verify.ps1
```

This verifies:
1. GPU hardware diagnosis and unified VRAM allocation.
2. Prefill token rates across 512, 2658, and 3800 context sizes.
3. Prompt parity across English and multilingual prompts.
4. Output logs saved to `build/verify_logs/<timestamp>/summary.txt`.

---

## Building from Source

### Windows (MSVC + AMD TheRock HIP SDK)

Prerequisites:
- Visual Studio 2022 (with C++ Desktop workload and CMake tools).
- AMD TheRock HIP SDK for Windows (`gfx1151`).
- Ninja build system.

Build with CMake presets:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release --parallel 8
```

The build produces:
- `build/windows-release/gufork.exe` (Primary CLI binary)
- `build/windows-release/gurfork.exe` (Convenience alias)
- `build/windows-release/gufork-server.exe` (Dedicated server binary)
- `build/windows-release/gufo.exe` and `gufo-server.exe` (Backwards-compatible symlinks/copies)

### Linux (Nix)

```bash
nix build
./result/bin/gufork diagnose
./result/bin/gufork serve llm --model /path/to/Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf
```

### Linux (Without Nix)

Install GCC 14+, CMake 3.21+, Ninja, ROCm 10.1+ (HIP Clang, hipBLAS, hipBLASLt, rocWMMA):

```bash
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 8
./build/release/gufork diagnose
```

---

## Upstream Credits & License

Gufork is forked from **[Gufo](https://github.com/gufo-org/gufo)** (`https://github.com/gufo-org/gufo`).

Full credit for the groundbreaking zero-overhead vertical inference engine on AMD Strix Halo belongs to the upstream [Gufo project](https://github.com/gufo-org/gufo) and its contributors. Please check out and star upstream:
- **Upstream Repository**: [https://github.com/gufo-org/gufo](https://github.com/gufo-org/gufo)
- **Upstream Releases**: [https://github.com/gufo-org/gufo/releases](https://github.com/gufo-org/gufo/releases)

Gufork's original code is licensed under the [MIT License](LICENSE). Adapted code and dependencies retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and `licenses/`.

Acknowledgements and references:
- **[Gufo](https://github.com/gufo-org/gufo)**: Upstream vertical Strix Halo inference engine.
- **[Harsha001](https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX)**: ROCmFPX quantization and GGUF conversion for Ornith-1.5-122B.
- **[ROCmFPX](https://github.com/charlie12345/ROCmFPX)**: AMD low-bit quantization specifications and kernels.
- **[Ornith AI](https://huggingface.co/ornith-ai/Ornith-1.5-122B)**: Base model architecture and BF16 model weights.
- **[bartowski](https://huggingface.co/bartowski/Ornith-1.5-122B-GGUF)**: Importance matrix calibration datasets.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: GGUF specification and quantized representation foundations.
