# qwen3-asr-rs

Pure-Rust inference engine for **Qwen3-ASR** automatic speech recognition models ([Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)) built on [candle](https://github.com/huggingface/candle). Runs fully locally — no Python, no PyTorch.

## Features

- **All model sizes** — 0.6B and 1.7B work out of the box; select the model directory at runtime
- **Metal GPU acceleration** — Apple Silicon (M1/M2/M3/M4) via candle's Metal backend (default)
- **CUDA GPU acceleration** — NVIDIA GPUs via candle's CUDA backend (`--features cuda`)
- **Multilingual** — English, Chinese, and code-switched audio (mixed-language)
- **Sharded weights** — loads both single-file and multi-shard `safetensors` models
- **Accurate mel extraction** — matches the official `WhisperFeatureExtractor` (Slaney-normalized, 128 mel bins)
- **MRoPE** — full multi-dimensional rotary position embedding for the Qwen3 decoder
- **No runtime dependencies** — statically linked, single binary

## Demo

Five audio samples are included in [`audio/`](audio/) — two short and three long — covering English, Mandarin, and code-switched speech. Click the links below to play each file directly in your browser (GitHub renders WAV files inline).

### Short Samples — exact match

#### sample1.wav · English · 3 s

[▶ audio/sample1.wav](audio/sample1.wav)

| | Text |
|---|---|
| **Expected** | The quick brown fox jumps over the lazy dog. |
| **Rust output** | The quick brown fox jumps over the lazy dog. |

---

#### sample2.wav · English · 4 s

[▶ audio/sample2.wav](audio/sample2.wav)

| | Text |
|---|---|
| **Expected** | Speech recognition has improved a lot in recent years. |
| **Rust output** | Speech recognition has improved a lot in recent years. |

---

### Long Samples

#### sample4.wav · English paragraph · 36 s

[▶ audio/sample4.wav](audio/sample4.wav)

**Expected:**
> Artificial intelligence has rapidly transformed numerous industries over the past decade. From healthcare diagnostics to autonomous vehicles, machine learning models are now capable of performing tasks that once required years of human expertise. Natural language processing, in particular, has seen dramatic improvements, enabling computers to understand, generate, and translate human speech with remarkable accuracy. Researchers continue to push the boundaries of what is possible, developing systems that can reason, plan, and even demonstrate creativity.

**Rust output (0.6B):**
> Artificial intelligence has rapidly transformed numerous industries over the past decade. From healthcare diagnostics to autonomous vehicles, machine learning models are now capable of performing tasks that once required years of human expertise. Natural language processing, in particular, has seen dramatic improvements, enabling computers to understand, generate, and translate human speech with remarkable accuracy. Researchers continue to push the boundaries of what is possible, developing systems that can reason, plan, and even demonstrate creativity.

---

#### sample5.wav · Mandarin paragraph · 30 s

[▶ audio/sample5.wav](audio/sample5.wav)

**Expected:**
> 随着科技的不断进步，人工智能已经深入到我们日常生活的每个角落。在医疗领域，智能诊断系统能够通过分析医学影像，快速准确地识别疾病。在交通领域，自动驾驶技术正在逐步走向成熟。在教育领域，个性化学习系统能够根据每个学生的学习进度，提供量身定制的教学内容，让每个孩子都能得到最适合自己的教育。

**Rust output (0.6B):**
> 随着科技的不断进步，人工智能已经深入到我们日常生活的每个角落。在医疗领域，智能诊断系统能够通过分析医学影像，快速准确地识别疾病。在交通领域，自动驾驶技术正在逐步走向成熟。在教育领域，个性化学习系统能够根据每个学生的学习进度，提供量身定制的教学内容，让每个孩子都能得到最适合自己的教育。

---

#### sample6.wav · Code-switched (Chinese + English) · 29 s

[▶ audio/sample6.wav](audio/sample6.wav)

**Expected:**
> 今天我们来讨论一下大语言模型的发展现状。Large language models like GPT and Claude have shown impressive results on a wide range of benchmarks, demonstrating strong reasoning and language understanding capabilities. 未来，随着多模态技术的进步，这些模型将能够同时处理文字、图像和语音，实现更加自然和智能的人机交互。

**Rust output (0.6B):**
> 今天我们来讨论一下大语言模型的发展现状。Large language models like GPT and Claude have shown impressive results on a wide range of benchmarks demonstrating strong reasoning and language understanding capabilities. 未来，随着多模态技术的进步，这些模型将能够同时处理文字、图像和语音，实现更加自然和智能的人机交互。

> ⚠️ Minor difference: comma after `benchmarks` is missing in Rust output.

---

## Architecture

Qwen3-ASR combines a Whisper-style audio encoder with a Qwen3 causal language model decoder:

```
Audio → Mel spectrogram (128 bins) → Conv2d ×3 downsampler
      → Transformer encoder (18L / 0.6B, 24L / 1.7B)
      → Linear projection → Qwen3 decoder (28L GQA + MRoPE) → Text
```

## Quick Start

### 0. Use directly from HuggingFace Hub

With the `hub` feature, no manual download is needed:

```rust
use qwen3_asr::{AsrInference, TranscribeOptions, best_device};
use std::path::Path;

let device = best_device(); // CUDA → Metal → CPU (based on enabled features)
let engine = AsrInference::from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    Path::new("models/"),
    device,
)?;
let result = engine.transcribe("audio.wav", TranscribeOptions::default())?;
println!("{}", result.text);
```

Or run the bundled example:

```bash
cargo run --example transcribe --features hub --release -- audio/sample1.wav
```

Models are cached in the `cache_dir` you specify (e.g. `models/`). A `.complete` marker file inside the model subdirectory indicates a successful download; subsequent calls skip the download entirely.

### 1. Download a model

```bash
pip install huggingface_hub

# 0.6B (~1.7 GB safetensors)
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models

# 1.7B (~4.5 GB safetensors)
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
MODEL_DIR=models cargo run --release
```

> Audio is automatically resampled to 16 kHz mono. WAV files are accepted.

### 4. Run the benchmark

```bash
cargo run --bin benchmark --release -- --model-dir models --runs 3
cargo run --bin benchmark --release -- --model-dir models_1.7b --runs 3
```

## Benchmark

### Test Environment

| Item | Value |
|------|-------|
| Hardware | Apple Mac mini, M4 (4P+6E cores), 16 GB unified memory |
| OS | macOS 26.3 (Darwin 25.3.0) |
| Rust | 1.93.1 |
| candle | 0.9.2 |
| Backend | Metal (Apple GPU) |
| Condition | Warm file-system cache; single-threaded inference; mean of 3 runs per sample |

### Audio Samples

| Sample | Language | Duration |
|--------|----------|----------|
| sample1.wav | English | 3.4 s |
| sample2.wav | English | 4.0 s |
| sample4.wav | English (long) | 36.4 s |
| sample5.wav | Mandarin (long) | 30.4 s |
| sample6.wav | Code-switched (long) | 28.7 s |

### Model Load Time and Memory

| Model | File Size | Load Time | Peak RSS | Phys Footprint |
|-------|-----------|-----------|----------|----------------|
| Qwen3-ASR-0.6B | 1.7 GB | 489 ms | 3707 MiB | 1883 MiB |
| Qwen3-ASR-1.7B | 4.5 GB | 4250 ms | 3493 MiB | 4569 MiB |

Two memory metrics are reported:

- **Peak RSS** — `getrusage(RUSAGE_SELF)` high-water mark over the whole process lifetime; never decreases; inflated by transient allocations during safetensors loading
- **Phys footprint** — `task_info TASK_VM_INFO phys_footprint` current physical pages owned by the process (including Metal GPU buffers); reflects actual live model memory after loading

### Real-Time Factor (RTF)

**RTF = inference\_time / audio\_duration.** Values below 1.0 mean faster-than-real-time. Lower is better.

| Model | sample1 | sample2 | sample4 | sample5 | sample6 | **Avg RTF** |
|-------|---------|---------|---------|---------|---------|-------------|
| 0.6B BF16 | 0.149 | 0.136 | 0.254 | 0.237 | 0.216 | **0.230** |
| 1.7B BF16 | 0.307 | 0.253 | 0.338 | 0.324 | 0.302 | **0.319** |

Both models run well below real-time on Apple M4 Metal. Short samples show higher variance because Metal shader warm-up cost is amortised over fewer tokens.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` / `candle-nn` | Tensor ops, Metal/CUDA backends |
| `tokenizers` | HuggingFace tokenizer (BPE) |
| `hound` | WAV file I/O |
| `rubato` | High-quality audio resampling |
| `rustfft` | FFT for mel spectrogram |
| `safetensors` | Model weight loading |

## Device Selection

`qwen3_asr::best_device()` automatically selects the best available backend based on compile-time features:

| Priority | Feature | Backend | Typical Platform |
|----------|---------|---------|-----------------|
| 1 | `cuda` | NVIDIA CUDA | Linux / Windows with NVIDIA GPU |
| 2 | `metal` (default) | Apple Metal | macOS with Apple Silicon |
| 3 | *(fallback)* | CPU | Any platform |

### Build examples

```bash
# Apple Silicon (Metal, default)
cargo run --release

# NVIDIA GPU (CUDA)
cargo run --release --no-default-features --features cuda

# CPU only
cargo run --release --no-default-features
```

### As a library dependency

```toml
# Cargo.toml — Metal (macOS default)
[dependencies]
qwen3-asr = "0.1"

# Cargo.toml — CUDA (Linux / Windows)
[dependencies]
qwen3-asr = { version = "0.1", default-features = false, features = ["cuda"] }

# Cargo.toml — CPU only
[dependencies]
qwen3-asr = { version = "0.1", default-features = false }
```

Then in your code:

```rust
use qwen3_asr::{AsrInference, TranscribeOptions, best_device};

let device = best_device();
let engine = AsrInference::load("path/to/model", device)?;
let result = engine.transcribe("audio.wav", TranscribeOptions::default())?;
println!("{}", result.text);
```

## Streaming Transcription

This crate provides a streaming API for low-latency, real-time transcription (live subtitles, voice assistants, meeting captioning).

### Basic usage

```rust
let mut state = engine.init_streaming(StreamingOptions::default());

// Feed audio chunks as they arrive from the microphone
for chunk in mic_chunks {
    if let Some(result) = engine.feed_audio(&mut state, &chunk)? {
        display_subtitle(&result.text);
    }
}

let final_result = engine.finish_streaming(&mut state)?;
```

### How it works

The streaming algorithm follows Qwen3-ASR's chunked AED with prefix conditioning:

1. Audio accumulates in an internal buffer; every `chunk_size_sec` (default 2s) triggers inference
2. The encoder uses windowed attention (104 tokens / ~16s window) — completed windows are cached, only the current partial window is recomputed
3. The decoder generates text conditioned on a **prefix** built from the previous output minus the last `unfixed_token_num` tokens (rollback strategy)
4. For the first `unfixed_chunk_num` chunks (cold start), no prefix is used

### Memory and duration limits

Each streaming step re-encodes all accumulated audio from scratch (mel extraction, decoder prefill). This is by design — the official Python implementation does the same. Costs grow with session duration:

| Duration | Audio tokens | Mel extraction | Decoder prefill | Practical |
|----------|-------------|----------------|-----------------|-----------|
| 2 min | ~780 | fast | fast | smooth |
| 10 min | ~3,900 | noticeable | ~1s/step | acceptable |
| 20 min | ~7,800 | slow | ~3-5s/step | upper limit |
| 60 min | ~23,400 | very slow | 10s+/step | not feasible |

The official Qwen3-ASR technical report states the model supports "single speech no longer than 20 minutes" for streaming.

### Recommended patterns for production

**Short sessions (< 20 min)** — use the streaming API directly:

```rust
let mut state = engine.init_streaming(StreamingOptions::default());
// feed_audio / finish_streaming as usual
```

**Long-running streams (meetings, lectures)** — reset sessions at silence boundaries using VAD. Starting a new session is instant (zero-cost struct init, no model reload):

```rust
let mut state = engine.init_streaming(opts.clone());

loop {
    let chunk = read_mic();
    if vad_detects_silence(&chunk, threshold) {
        let result = engine.finish_streaming(&mut state)?;
        save_transcript(&result);
        state = engine.init_streaming(opts.clone()); // instant, zero cost
    } else {
        if let Some(result) = engine.feed_audio(&mut state, &chunk)? {
            display_subtitle(&result.text);
        }
    }
}
```

This keeps each session short (one utterance or a few sentences), so memory and latency stay constant. This pattern can run indefinitely.

**Long pre-recorded files (podcasts, recordings)** — do not use streaming. Instead, split the audio into segments at the app layer and call `transcribe_samples()` on each:

```rust
let segments = split_at_silence(&audio); // your VAD / energy-based splitter
for seg in &segments {
    let result = engine.transcribe_samples(seg, TranscribeOptions::default())?;
    output.push(result.text);
}
let full_text = output.join("");
```

This matches the official Qwen3-ASR approach — their `transcribe()` uses `split_audio_into_chunks()` (energy-based splitting, max 1200s per segment) and processes segments independently.

## Implementation Notes

- **Mel extraction** matches `WhisperFeatureExtractor` exactly: Slaney-normalized filterbanks, `n_fft=400`, `hop_length=160`, `n_mels=128`, max diff < 3e-5 vs PyTorch reference
- **Positional embeddings** in the audio encoder are sinusoidal and computed per-chunk (positions reset to 0 for each 30-second window), matching the Python reference
- **BF16 precision** throughout — `candle_nn::LayerNorm` and `candle_nn::RmsNorm` handle the F32 upcast internally; attention softmax uses `softmax_last_dim` (subtract-max stable, native Metal kernel), matching Qwen2 / candle-transformers convention
- **Token 151704** (`<asr_sep>`) splits the decoder output into `language` and `text` fields; it is absent from the base Qwen3 tokenizer (decodes to `""`) so it is detected by token ID directly

## License

MIT
