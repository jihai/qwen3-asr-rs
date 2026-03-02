# qwen3-asr-rs

Pure-Rust inference engine for **Qwen3-ASR** automatic speech recognition models ([Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)) built on [candle](https://github.com/huggingface/candle). Runs fully locally — no Python, no PyTorch.

## Features

- **All model sizes** — 0.6B and 1.7B work out of the box; select the model directory at runtime
- **Metal GPU acceleration** — Apple Silicon (M1/M2/M3/M4) via candle's Metal backend
- **CUDA support** — enable the `cuda` feature for NVIDIA GPUs
- **Multilingual** — English, Chinese, and code-switched audio (mixed-language)
- **Sharded weights** — loads both single-file and multi-shard `safetensors` models
- **Accurate mel extraction** — matches the official `WhisperFeatureExtractor` (Slaney-normalized, 128 mel bins)
- **BF16 throughout** — matches the reference PyTorch BF16 output exactly
- **MRoPE** — full multi-dimensional rotary position embedding for the Qwen3 decoder
- **No runtime dependencies** — statically linked, single binary

## Architecture

Qwen3-ASR combines a Whisper-style audio encoder with a Qwen3 causal language model decoder:

```
Audio → Mel spectrogram (128 bins) → Conv2d ×3 downsampler
      → Transformer encoder (18L / 0.6B, 24L / 1.7B)
      → Linear projection → Qwen3 decoder (28L GQA + MRoPE) → Text
```

## Quick Start

### 1. Download a model

```bash
pip install huggingface_hub

# 0.6B (~3.4 GB)
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models

# 1.7B (~4.5 GB)
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models_1.7b
```

### 2. Build and run

```bash
# Apple Silicon (Metal)
cargo run --release

# 1.7B model
cargo run --release -- models_1.7b

# CPU only
cargo run --release --no-default-features
```

### 3. Transcribe your own audio

```bash
MODEL_DIR=models cargo run --release -- path/to/audio.wav
```

> Audio is automatically resampled to 16 kHz mono. WAV, and any format supported by `hound` are accepted.

## Test Results

Tested on Apple M-series with Metal acceleration.

| Sample | Duration | Model | Result |
|--------|----------|-------|--------|
| English sentence | 3 s | 0.6B | ✓ exact match |
| English sentence | 3 s | 1.7B | ✓ exact match |
| Long English paragraph | 45 s | 0.6B | ✓ exact match |
| Long English paragraph | 45 s | 1.7B | ✓ exact match |
| Long Chinese paragraph | 30 s | 0.6B | ✓ exact match |
| Long Chinese paragraph | 30 s | 1.7B | ✓ near-exact match |
| Mixed Chinese-English | 25 s | 0.6B | ✓ full transcription |
| Mixed Chinese-English | 25 s | 1.7B | ✓ full transcription |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` / `candle-nn` | Tensor ops, Metal/CUDA backends |
| `tokenizers` | HuggingFace tokenizer (BPE) |
| `hound` | WAV file I/O |
| `rubato` | High-quality audio resampling |
| `rustfft` | FFT for mel spectrogram |
| `safetensors` | Model weight loading |

## Enabling CUDA

```toml
# Cargo.toml
[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

## Implementation Notes

- **Mel extraction** matches `WhisperFeatureExtractor` exactly: Slaney-normalized filterbanks, `n_fft=400`, `hop_length=160`, `n_mels=128`, max diff < 3e-5 vs PyTorch reference
- **Positional embeddings** in the audio encoder are sinusoidal and computed per-chunk (positions reset to 0 for each 30-second window), matching the Python reference
- **BF16 precision** is used throughout — LayerNorm and softmax are computed in F32 and cast back — this matches the official PyTorch BF16 output
- **Token 151704** (`<asr_sep>`) splits the decoder output into `language` and `text` fields; it is absent from the base Qwen3 tokenizer (decodes to `""`) so it is detected by token ID directly

## License

MIT
