"""
Qwen3-ASR-0.6B Audio Encoding Pipeline - Reference Notes
=========================================================
Source repositories:
  - https://github.com/QwenLM/Qwen3-ASR  (official Python/transformers backend)
  - https://github.com/antirez/qwen-asr  (C inference implementation with MODEL.md)
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B  (model weights + config)

===========================================================================
SECTION 1: MEL FEATURE EXTRACTION
===========================================================================

The mel spectrogram is computed by WhisperFeatureExtractor (NOT a custom class).
Parameters from preprocessor_config.json:

  sample_rate   = 16000 Hz      (16 kHz mono audio)
  n_fft         = 400           (FFT window size = 25 ms at 16 kHz)
  hop_length    = 160           (hop = 10 ms at 16 kHz -> 100 frames/sec)
  n_mels        = 128           (mel bins, spanning 0-8000 Hz)
  chunk_length  = 30            (seconds, max audio length = 30s -> 3000 frames)
  n_samples     = 480000        (= 30s * 16000 Hz)
  nb_max_frames = 3000          (= 30s * 100 frames/s)

  fmin          = 0 Hz          (Whisper default)
  fmax          = 8000 Hz       (= sample_rate / 2)
  mel_norm      = "slaney"      (Slaney normalization, NOT htk)
  window        = "hann"        (Hann window)

Log-mel normalization (WhisperFeatureExtractor standard):
  1. Compute power mel spectrogram: |STFT|^2
  2. Apply mel filterbank
  3. Clamp to max_val / 1e8: mel = clip(mel, min=1e-10)  (or: max_val = mel.max(), clamp to max_val/1e8)
  4. Log10: log_mel = log10(mel)
  5. Clamp to 8 dB ceiling: log_mel = clip(log_mel, min=log_mel.max() - 8.0)
     NOTE: This is DYNAMIC (based on the spectrogram's own max), unlike Voxtral which uses a fixed global 1.5
  6. Normalize: log_mel = (log_mel + 4.0) / 4.0

Output shape: [1, 128, T] where T = ceil(audio_samples / hop_length)
  e.g. 1.6s audio -> T = ceil(25600 / 160) = 160 frames

===========================================================================
SECTION 2: CONV2D STEM ARCHITECTURE
===========================================================================

Three Conv2d layers provide 8x total downsampling in both time and frequency:

  Input:   [num_chunks, 1, 128, 100]  (1 channel, 128 freq bins, 100 time frames)

  conv2d1: Conv2d(in=1,   out=480, kernel=3x3, stride=2, padding=1) + GELU
  Output:  [num_chunks, 480, 64, 50]
           freq: ceil((128 + 2*1 - 3) / 2 + 1) = 64
           time: ceil((100 + 2*1 - 3) / 2 + 1) = 50

  conv2d2: Conv2d(in=480, out=480, kernel=3x3, stride=2, padding=1) + GELU
  Output:  [num_chunks, 480, 32, 25]

  conv2d3: Conv2d(in=480, out=480, kernel=3x3, stride=2, padding=1) + GELU
  Output:  [num_chunks, 480, 16, 13]
           freq: 128 -> 64 -> 32 -> 16  (8x reduction)
           time: 100 -> 50 -> 25 -> 13  (ceil(25/2+1)=13, ~8x reduction)

  Reshape: permute(0, 3, 1, 2).contiguous().view(B, T, C*F)
           [num_chunks, 480, 16, 13] -> [num_chunks, 13, 480*16] = [num_chunks, 13, 7680]

  conv_out: Linear(7680, 896, bias=False)
  Output:  [num_chunks, 13, 896]

So each 100-frame mel chunk -> 13 audio tokens of dimension 896.

For the 0.6B model, n_window=50, so chunk_size = n_window * 2 = 100 mel frames.
Each chunk covers 100 frames * 10ms/frame = 1.0 second of audio.
Each chunk produces 13 tokens.

Output token count formula:
  full_chunks = floor(T / 100)
  remainder   = T % 100
  tokens_from_remainder = ceil(ceil(ceil(remainder / 2) / 2) / 2)
                        = ((remainder - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1
  (simplified in code as):
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + full_chunks * 13

===========================================================================
SECTION 3: POSITIONAL EMBEDDINGS
===========================================================================

Type: Standard sinusoidal (not RoPE, not learned)
      Same formula as Attention Is All You Need (Vaswani et al. 2017).

Formula:
  PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
  PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

Buffer shape: [max_source_positions, d_model] = [1500, 896]

CRITICAL: Positions are applied PER CHUNK, not globally.
  - Each 100-frame chunk produces 13 tokens.
  - Those 13 tokens receive positions [0, 1, 2, ..., 12].
  - The next chunk's 13 tokens ALSO receive positions [0, 1, ..., 12].
  - There is NO global position information across chunks.
  - This is by design: the model is trained to be position-invariant across chunks,
    relying on the decoder's MRoPE for global temporal awareness.

Implementation:
  positional_embedding = self.positional_embedding.positional_embedding[:13, :].unsqueeze(0)
  padded_embed = padded_embed + positional_embedding
  # broadcast: [1, 13, 896] added to [num_chunks, 13, 896]

===========================================================================
SECTION 4: TRANSFORMER ENCODER
===========================================================================

Architecture: Pre-norm transformer (standard, not decoder-style)
Layers: 18 (0.6B model)
d_model: 896
Attention heads: 14
head_dim: 896 / 14 = 64
FFN dim: 3584 (4x d_model)
Activation: GELU
Normalization: LayerNorm WITH bias (unlike text decoder which uses bias-free RMSNorm)
Attention: Bidirectional (full attention within each window)

Windowed attention:
  - n_window_infer = 800 tokens (0.6B model)
  - window_aftercnn = max_tokens_in_padded_chunk * (n_window_infer / chunk_size_tokens)
                    = 13 * (800 / 100) = 13 * 8 = 104 tokens per attention window
  - Tokens only attend to others within the same cu_seqlens window
  - No cross-window attention (enables O(n) scaling with audio length)

===========================================================================
SECTION 5: OUTPUT PROJECTIONS
===========================================================================

After the transformer encoder stack:
  ln_post:  LayerNorm(896)
  proj1:    Linear(896, 896)
  act:      GELU
  proj2:    Linear(896, 1024)   <- output_dim for 0.6B

Final audio token dimension: 1024
These tokens are concatenated into the LLM input sequence at the position
of audio_token_id = 151676, between audio_start_token_id (151669) and
audio_end_token_id (151670).

===========================================================================
SECTION 6: LLM TEXT DECODER
===========================================================================

Uses Qwen3 decoder architecture (NOT the audio encoder's transformer):
  hidden_size: 1024
  num_hidden_layers: 28
  num_attention_heads: 16
  num_key_value_heads: 8  (GQA with ratio 2:1)
  intermediate_size: 3072
  vocab_size: 151936
  max_position_embeddings: 65536
  rms_norm_eps: 1e-6
  rope_theta: 1000000
  rope_scaling: MRoPE (Multi-dimensional RoPE, interleaved=True)
    mrope_section: [24, 20, 20]  (time, height, width dims for audio tokens)
  hidden_act: silu
  dtype: bfloat16

===========================================================================
SECTION 7: WHISPER FEATURE EXTRACTOR MEL FILTERBANK DETAILS
===========================================================================

The mel filterbank is NOT stored in preprocessor_config.json - it is computed
at runtime by WhisperFeatureExtractor using librosa's mel filterbank formula
with Slaney normalization.

Key parameters:
  - 128 triangular mel filters
  - Frequency range: 0 Hz to fmax = sample_rate/2 = 8000 Hz
  - Mel scale conversion: 2595 * log10(1 + f/700)  (standard Slaney/O'Shaughnessy mel)
  - Filter normalization: Slaney (area normalization, NOT HTK peak normalization)
  - Result: filterbank matrix of shape [128, n_fft//2 + 1] = [128, 201]

STFT:
  - Window: Hann window of length n_fft=400
  - FFT size: n_fft=400
  - Hop: hop_length=160
  - Magnitude spectrum: |STFT|^2 (power spectrum)
  - Frequency bins: n_fft//2 + 1 = 201

===========================================================================
SECTION 8: COMPLETE PIPELINE (PSEUDOCODE)
===========================================================================

def encode_audio(wav_16khz_mono: np.ndarray) -> torch.Tensor:
    # Step 1: Mel spectrogram
    mel = whisper_feature_extractor(wav_16khz_mono)
    # mel: [1, 128, T]  (T frames at 100 Hz)

    # Step 2: Pass to audio encoder
    feature_lens = torch.tensor([T])  # actual frame count (excluding padding)
    audio_tokens = audio_encoder(mel, feature_lens=feature_lens)
    # audio_tokens.last_hidden_state: [N_tokens, 1024]
    # where N_tokens = _get_feat_extract_output_lengths(T)

    # Step 3: Build LLM input sequence
    # [BOS] [audio_start] [audio_token x N_tokens] [audio_end] [task_prompt_tokens] [EOS]

    return audio_tokens.last_hidden_state
"""
