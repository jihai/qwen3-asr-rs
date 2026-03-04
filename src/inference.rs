use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use log::{debug, info};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::config::AsrConfig;
use crate::decoder::{compute_mrope_cos_sin, create_causal_mask, KvCache, TextDecoder};
use crate::encoder::AudioEncoder;
use crate::error::AsrError;
use crate::mel::{load_audio_wav, MelExtractor};

// Special token IDs
pub(crate) const IM_END_TOKEN_ID: i64 = 151645;
pub(crate) const ENDOFTEXT_TOKEN_ID: i64 = 151643;
// ASR-specific separator token (not in base Qwen3 tokenizer vocab, hence decodes to "")
pub(crate) const ASR_TEXT_SEP_TOKEN_ID: u32 = 151704;

pub(crate) const MEL_SAMPLE_RATE: u32 = 16000;
const N_FFT:           usize = 400; // Whisper-compatible FFT window (25ms @ 16kHz)
const HOP_LENGTH:      usize = 160; // Whisper-compatible hop size  (10ms @ 16kHz)

// Prompt structure token IDs (Qwen3 chat template)
pub(crate) const TOK_IM_START:  i64 = 151644; // <|im_start|>
pub(crate) const TOK_SYSTEM:    i64 = 8948;   // "system"
pub(crate) const TOK_NEWLINE:   i64 = 198;    // "\n"
pub(crate) const TOK_IM_END:    i64 = IM_END_TOKEN_ID; // 151645
pub(crate) const TOK_USER:      i64 = 872;    // "user"
pub(crate) const TOK_ASSISTANT: i64 = 77091;  // "assistant"

/// Options controlling the transcription behaviour.
///
/// Construct via [`TranscribeOptions::default()`] and then mutate the fields
/// you need to override:
///
/// ```
/// # use qwen3_asr::TranscribeOptions;
/// let mut opts = TranscribeOptions::default();
/// opts.language = Some("english".into());
/// ```
#[non_exhaustive]
pub struct TranscribeOptions {
    /// Force a specific language (e.g. `"english"`). `None` enables auto-detection.
    pub language: Option<String>,
    /// Maximum number of new tokens to generate. Default: 512.
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self { language: None, max_new_tokens: 512 }
    }
}

impl TranscribeOptions {
    /// Set the maximum number of new tokens to generate.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Force a specific language (e.g. `"english"`).
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

#[non_exhaustive]
pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

pub(crate) struct AsrInferenceInner {
    pub(crate) audio_encoder: AudioEncoder,
    pub(crate) text_decoder: TextDecoder,
    pub(crate) mel_extractor: MelExtractor,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) config: AsrConfig,
    pub(crate) device: Device,
}

// SAFETY: The raw pointers inside candle Metal tensors point to heap-allocated
// buffers managed via Arc, not thread-local storage. Transferring ownership to
// another thread is therefore safe. Concurrent access is prevented by the
// enclosing Mutex<AsrInferenceInner>.
unsafe impl Send for AsrInferenceInner {}

pub struct AsrInference {
    pub(crate) inner: Mutex<AsrInferenceInner>,
}
// AsrInference: Send + Sync automatically — Mutex<T>: Send+Sync when T: Send.

impl AsrInference {
    pub fn load(model_dir: &Path, device: Device) -> crate::Result<Self> {
        info!("Loading config...");
        let config = AsrConfig::from_file(&model_dir.join("config.json"))
            .context("load config")
            .map_err(AsrError::ModelLoad)?;

        info!("Loading weights (this may take a moment)...");
        let weights = load_weights(model_dir, &device)
            .context("load weights")
            .map_err(AsrError::ModelLoad)?;
        info!("Loaded {} weight tensors", weights.len());

        info!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))
            .map_err(AsrError::ModelLoad)?;

        info!("Model loaded successfully.");
        Self::build_engine(config, weights, tokenizer, device).map_err(AsrError::ModelLoad)
    }

    /// Download a model from HuggingFace Hub (if not already cached) and load it.
    ///
    /// `cache_dir` specifies where model files are stored persistently.
    /// A `.complete` marker file inside the model subdirectory indicates that
    /// all files have been downloaded successfully; subsequent calls skip the
    /// download entirely.
    ///
    /// Requires the `hub` feature flag.
    ///
    /// # Example
    /// ```no_run
    /// # use candle_core::Device;
    /// # use std::path::Path;
    /// let engine = qwen3_asr::AsrInference::from_pretrained(
    ///     "Qwen/Qwen3-ASR-0.6B",
    ///     Path::new("models/"),
    ///     Device::Cpu,
    /// )?;
    /// # Ok::<(), qwen3_asr::AsrError>(())
    /// ```
    #[cfg(feature = "hub")]
    pub fn from_pretrained(model_id: &str, cache_dir: &Path, device: Device) -> crate::Result<Self> {
        let model_dir = ensure_model_cached(model_id, cache_dir).map_err(AsrError::ModelLoad)?;
        Self::load(&model_dir, device)
    }

    fn build_engine(
        config: AsrConfig,
        weights: HashMap<String, Tensor>,
        tokenizer: tokenizers::Tokenizer,
        device: Device,
    ) -> anyhow::Result<Self> {
        info!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(
            &weights,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
            &device,
        )
        .context("load audio encoder")?;

        info!("Loading text decoder...");
        let text_decoder = TextDecoder::load(
            &weights,
            "thinker.model",
            &config.thinker_config.text_config,
        )
        .context("load text decoder")?;

        let mel_extractor = MelExtractor::new(
            N_FFT,
            HOP_LENGTH,
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
        );

        let inner = AsrInferenceInner { audio_encoder, text_decoder, mel_extractor, tokenizer, config, device };
        Ok(AsrInference { inner: Mutex::new(inner) })
    }

    /// Transcribe from a WAV file path.
    pub fn transcribe(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
    ) -> crate::Result<TranscribeResult> {
        info!("Loading audio: {}", audio_path);
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        info!("Audio: {} samples @ {}Hz", samples.len(), MEL_SAMPLE_RATE);
        let inner = self.inner.lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner.run_inference(&samples, &options).map_err(AsrError::Inference)
    }

    /// Transcribe directly from pre-loaded 16 kHz f32 samples.
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        options: TranscribeOptions,
    ) -> crate::Result<TranscribeResult> {
        let inner = self.inner.lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner.run_inference(samples, &options).map_err(AsrError::Inference)
    }
}

impl AsrInferenceInner {
    pub(crate) fn run_inference(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> anyhow::Result<TranscribeResult> {
        let audio_embeds = self.encode_audio(samples)?;
        let generated_ids = self.generate(
            &audio_embeds,
            options.language.as_deref(),
            None, // no prefix
            options.max_new_tokens,
        )?;
        self.decode_result(&generated_ids, options.language.as_deref())
    }

    /// Extract mel spectrogram and run the audio encoder.
    /// Returns audio embeddings [num_audio_tokens, output_dim].
    pub(crate) fn encode_audio(&self, samples: &[f32]) -> anyhow::Result<Tensor> {
        let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
        debug!("Mel: {}×{} frames", n_mels, n_frames);
        let mel = Tensor::from_vec(mel_data, (n_mels, n_frames), &self.device)?;
        let audio_embeds = self.audio_encoder.forward(&mel)?;
        info!("Audio tokens: {}", audio_embeds.dims()[0]);
        Ok(audio_embeds)
    }

    /// Run the full decoder pipeline: build prompt, prefill, generate tokens.
    ///
    /// `prefix_text`: optional text to prepend to the assistant turn (for streaming
    /// rollback). The prefix tokens are included in the prompt and the model
    /// generates continuation tokens after them.
    ///
    /// Returns the raw generated token IDs (not including prompt/prefix tokens).
    pub(crate) fn generate(
        &self,
        audio_embeds: &Tensor,
        language: Option<&str>,
        prefix_text: Option<&str>,
        max_new_tokens: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let num_audio_tokens = audio_embeds.dims()[0];

        // Build prompt token IDs (with optional prefix)
        let (input_ids, audio_start_pos) =
            self.build_prompt(num_audio_tokens, language, prefix_text)?;
        let seq_len = input_ids.len();

        // Build embeddings, inject audio at the audio pad positions
        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + num_audio_tokens..].to_vec();

        let before_t =
            Tensor::from_vec(before_ids, (audio_start_pos,), &self.device)?.to_dtype(DType::U32)?;
        let after_t = Tensor::from_vec(
            after_ids,
            (input_ids.len() - audio_start_pos - num_audio_tokens,),
            &self.device,
        )?
        .to_dtype(DType::U32)?;

        let before_emb = self.text_decoder.embed(&before_t)?;
        let after_emb = self.text_decoder.embed(&after_t)?;
        let audio_emb = audio_embeds.to_dtype(before_emb.dtype())?;

        let hidden_states =
            Tensor::cat(&[&before_emb, &audio_emb, &after_emb], 0)?.unsqueeze(0)?;

        // Precompute MRoPE cos/sin table
        let text_cfg = &self.config.thinker_config.text_config;
        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_table, sin_table) = compute_mrope_cos_sin(
            &full_ids,
            text_cfg.head_dim,
            text_cfg.rope_theta,
            &text_cfg.mrope_section(),
            text_cfg.mrope_interleaved(),
            &self.device,
        )?;

        let cos = cos_table.narrow(0, 0, seq_len)?;
        let sin = sin_table.narrow(0, 0, seq_len)?;

        // Prefill
        let mask = create_causal_mask(seq_len, 0, &self.device)?;
        let mut kv_cache = KvCache::new(text_cfg.num_hidden_layers);

        let logits = self.text_decoder.forward(
            &hidden_states,
            &cos,
            &sin,
            &mut kv_cache,
            Some(&mask),
        )?;

        // Autoregressive generation
        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        let mut next_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let mut current_pos = seq_len;

        for step_idx in 0..max_new_tokens {
            let next_token = next_logits.argmax(1)?.to_vec1::<u32>()?[0];

            if log::log_enabled!(log::Level::Debug) {
                let logits_f32 = next_logits.to_dtype(candle_core::DType::F32)?;
                let logits_vec = logits_f32.to_vec2::<f32>()?[0].clone();
                let mut indexed: Vec<(f32, u32)> = logits_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v, i as u32))
                    .collect();
                indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let top10: Vec<String> = indexed
                    .iter()
                    .take(10)
                    .map(|(score, tok)| format!("{}({:.2})", tok, score))
                    .collect();
                debug!(
                    "  step {:2}: top10 = {}  chosen={}",
                    step_idx,
                    top10.join(" "),
                    next_token
                );
            }

            if eos_ids.contains(&(next_token as i64)) {
                break;
            }

            generated_ids.push(next_token);

            let next_id_t =
                Tensor::from_vec(vec![next_token], (1,), &self.device)?;
            let next_emb = self.text_decoder.embed(&next_id_t)?.unsqueeze(0)?;

            let new_cos = cos_table.narrow(0, current_pos, 1)?;
            let new_sin = sin_table.narrow(0, current_pos, 1)?;

            let past_len = kv_cache.seq_len();
            let step_mask = create_causal_mask(1, past_len, &self.device)?;

            let step_logits = self.text_decoder.forward(
                &next_emb,
                &new_cos,
                &new_sin,
                &mut kv_cache,
                Some(&step_mask),
            )?;

            next_logits = step_logits.squeeze(1)?;
            current_pos += 1;
        }

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }

    /// Decode generated token IDs into a TranscribeResult.
    pub(crate) fn decode_result(
        &self,
        generated_ids: &[u32],
        language: Option<&str>,
    ) -> anyhow::Result<TranscribeResult> {
        let raw_text = self
            .tokenizer
            .decode(generated_ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {}", e))?;

        let (lang, text) = if language.is_some() {
            ("forced".to_string(), raw_text.trim().to_string())
        } else if let Some(sep_pos) =
            generated_ids.iter().position(|&id| id == ASR_TEXT_SEP_TOKEN_ID)
        {
            let lang_ids: Vec<u32> = generated_ids[..sep_pos].to_vec();
            let text_ids: Vec<u32> = generated_ids[sep_pos + 1..].to_vec();
            let lang_raw = self
                .tokenizer
                .decode(&lang_ids, true)
                .map_err(|e| anyhow::anyhow!("decode lang: {}", e))?;
            let text_raw = self
                .tokenizer
                .decode(&text_ids, true)
                .map_err(|e| anyhow::anyhow!("decode text: {}", e))?;
            let lang =
                lang_raw.strip_prefix("language ").unwrap_or(&lang_raw).trim().to_string();
            (lang, text_raw.trim().to_string())
        } else {
            parse_asr_output(&raw_text, false)
        };
        Ok(TranscribeResult { text, language: lang, raw_output: raw_text })
    }

    /// Encode text into token IDs using the tokenizer.
    #[allow(dead_code)]
    pub(crate) fn tokenizer_encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token IDs into text using the tokenizer.
    pub(crate) fn tokenizer_decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {}", e))
    }

    /// Build prompt token IDs with optional prefix text in the assistant turn.
    pub(crate) fn build_prompt(
        &self,
        num_audio_tokens: usize,
        language: Option<&str>,
        prefix_text: Option<&str>,
    ) -> anyhow::Result<(Vec<i64>, usize)> {
        let cfg = &self.config.thinker_config;
        let mut tokens: Vec<i64> = vec![
            TOK_IM_START,
            TOK_SYSTEM,
            TOK_NEWLINE,
            TOK_IM_END,
            TOK_NEWLINE,
            TOK_IM_START,
            TOK_USER,
            TOK_NEWLINE,
            cfg.audio_start_token_id,
        ];

        let audio_start_pos = tokens.len();
        tokens.extend(std::iter::repeat_n(cfg.audio_token_id, num_audio_tokens));

        tokens.extend_from_slice(&[
            cfg.audio_end_token_id,
            TOK_IM_END,
            TOK_NEWLINE,
            TOK_IM_START,
        ]);

        if let Some(lang) = language {
            tokens.push(TOK_ASSISTANT);
            tokens.push(TOK_NEWLINE);
            let prefix = format!("language {}", capitalize_first(lang));
            let enc = self
                .tokenizer
                .encode(prefix.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        } else {
            tokens.push(TOK_ASSISTANT);
            tokens.push(TOK_NEWLINE);
        }

        // Append prefix text tokens (for streaming rollback)
        if let Some(prefix) = prefix_text {
            if !prefix.is_empty() {
                let enc = self
                    .tokenizer
                    .encode(prefix, false)
                    .map_err(|e| anyhow::anyhow!("encode prefix: {}", e))?;
                tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
            }
        }

        Ok((tokens, audio_start_pos))
    }
}

fn parse_asr_output(raw: &str, language_forced: bool) -> (String, String) {
    if language_forced {
        return ("forced".to_string(), raw.trim().to_string());
    }
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix("language ") {
        if let Some(pos) = rest.find("<asr_text>") {
            let lang = rest[..pos].trim().to_string();
            let text = rest[pos + "<asr_text>".len()..].trim().to_string();
            return (lang, text);
        }
        // Find first non-alphabetic char to split lang from text
        let mut lang_end = rest.len();
        for (i, c) in rest.char_indices() {
            if c.is_whitespace() || !c.is_alphabetic() {
                lang_end = i;
                break;
            }
        }
        if lang_end > 0 && lang_end < rest.len() {
            let lang = rest[..lang_end].to_string();
            let text = rest[lang_end..].trim().to_string();
            return (lang, text);
        }
    }
    ("unknown".to_string(), raw.to_string())
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

// ─── Hub download helpers ─────────────────────────────────────────────────────

#[cfg(feature = "hub")]
fn hf_url(model_id: &str, filename: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", model_id, filename)
}

/// Make a GET request; returns `None` on 404, error on other failures.
#[cfg(feature = "hub")]
fn hf_try_get(url: &str) -> anyhow::Result<Option<reqwest::blocking::Response>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut b = client.get(url);
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        b = b.header("Authorization", format!("Bearer {}", tok));
    }
    let resp = b.send()?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} for {}", resp.status(), url);
    }
    Ok(Some(resp))
}

/// GET a URL and return the full body as bytes.
#[cfg(feature = "hub")]
fn hf_get_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    hf_try_get(url)?
        .ok_or_else(|| anyhow::anyhow!("404: {}", url))
        .and_then(|r| Ok(r.bytes()?.to_vec()))
}

/// Stream a URL to a file, printing progress to stderr.
#[cfg(feature = "hub")]
fn hf_stream_to_file(url: &str, path: &std::path::Path) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    info!("Downloading {}", url);
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut b = client.get(url);
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        b = b.header("Authorization", format!("Bearer {}", tok));
    }
    let mut resp = b.send()?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} for {}", resp.status(), url);
    }
    let mut file = std::fs::File::create(path)?;
    let mut downloaded = 0u64;
    let mut buf = [0u8; 65536];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
    }
    info!("Downloaded {:.1} MB", downloaded as f64 / 1_048_576.0);
    Ok(())
}

/// Ensure model files for `model_id` exist under `cache_dir` and return the
/// model directory path.  Downloads from HuggingFace only when needed.
///
/// Cache layout: `{cache_dir}/{model_id.replace('/', '--')}/`
/// A `.complete` marker file signals that all files are present. If the
/// directory exists but `.complete` is missing (interrupted download), the
/// directory is removed and the download restarts.
#[cfg(feature = "hub")]
fn ensure_model_cached(model_id: &str, cache_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let sanitized = model_id.replace('/', "--");
    let model_dir = cache_dir.join(&sanitized);
    let marker = model_dir.join(".complete");

    // Fast path: already downloaded.
    if marker.exists() {
        info!("Using cached model at {}", model_dir.display());
        return Ok(model_dir);
    }

    // Partial / interrupted download — remove and restart.
    if model_dir.exists() {
        info!("Removing incomplete download at {}", model_dir.display());
        std::fs::remove_dir_all(&model_dir)?;
    }

    info!("Downloading '{}' from HuggingFace to {}…", model_id, model_dir.display());
    std::fs::create_dir_all(&model_dir)?;

    // config.json
    let config_bytes = hf_get_bytes(&hf_url(model_id, "config.json"))
        .context("download config.json")?;
    std::fs::write(model_dir.join("config.json"), &config_bytes)?;

    // Weights: check for sharded index first.
    if let Some(resp) = hf_try_get(&hf_url(model_id, "model.safetensors.index.json"))? {
        let index_text = resp.text()?;
        std::fs::write(model_dir.join("model.safetensors.index.json"), &index_text)?;

        let index: serde_json::Value = serde_json::from_str(&index_text)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid model.safetensors.index.json"))?;
        let shards: std::collections::HashSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        for shard in &shards {
            hf_stream_to_file(&hf_url(model_id, shard), &model_dir.join(shard))
                .with_context(|| format!("download shard {}", shard))?;
        }
    } else {
        hf_stream_to_file(
            &hf_url(model_id, "model.safetensors"),
            &model_dir.join("model.safetensors"),
        )
        .context("download model.safetensors")?;
    }

    // Tokenizer: Qwen3-ASR ships tokenizer_config.json (with added_tokens_decoder)
    // but not tokenizer.json. Reconstruct from vocab.json + merges.txt + config.
    let tok_config = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "tokenizer_config.json"))
            .context("download tokenizer_config.json")?,
    )?;
    let vocab = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "vocab.json")).context("download vocab.json")?,
    )?;
    let merges = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "merges.txt")).context("download merges.txt")?,
    )?;
    let tok_json = build_qwen3_tokenizer_json(&vocab, &merges, &tok_config)?;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json)?;

    // Mark download as complete.
    std::fs::write(&marker, b"")?;
    info!("Download complete, cached at {}", model_dir.display());

    Ok(model_dir)
}

/// Build the Qwen3 tokenizer JSON from vocab.json, merges.txt, and tokenizer_config.json.
/// The added_tokens list is derived from tokenizer_config.json's added_tokens_decoder field,
/// so no special tokens need to be hardcoded here.
#[cfg(feature = "hub")]
fn build_qwen3_tokenizer_json(vocab: &str, merges: &str, tok_config: &str) -> anyhow::Result<Vec<u8>> {
    let vocab_val: serde_json::Value = serde_json::from_str(vocab)?;
    let merges_vec: Vec<&str> = merges
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .collect();

    // Build added_tokens from tokenizer_config.json's added_tokens_decoder.
    let tok_cfg: serde_json::Value = serde_json::from_str(tok_config)?;
    let mut added_tokens: Vec<serde_json::Value> = Vec::new();
    if let Some(decoder_map) = tok_cfg["added_tokens_decoder"].as_object() {
        let mut entries: Vec<(u64, &serde_json::Value)> = decoder_map
            .iter()
            .filter_map(|(k, v)| k.parse::<u64>().ok().map(|id| (id, v)))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        for (id, v) in &entries {
            added_tokens.push(serde_json::json!({
                "id": id,
                "content": v["content"],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": v["special"]
            }));
        }
    }
    let added_tokens = serde_json::Value::Array(added_tokens);

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }
            ]
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab_val,
            "merges": merges_vec
        }
    });

    serde_json::to_vec(&tokenizer_json).map_err(Into::into)
}

/// Load safetensors weights from a directory (single file or sharded).
fn load_weights(model_dir: &Path, device: &Device) -> anyhow::Result<HashMap<String, Tensor>> {
    // Check for sharded model
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid index.json"))?;

        let mut shard_files: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for v in weight_map.values() {
            if let Some(s) = v.as_str() {
                shard_files.insert(s.to_string());
            }
        }

        let mut all_weights = HashMap::new();
        for shard in shard_files {
            let shard_path = model_dir.join(&shard);
            let w = candle_core::safetensors::load(&shard_path, device)?;
            all_weights.extend(w);
        }
        return Ok(all_weights);
    }

    // Single file
    let model_path = model_dir.join("model.safetensors");
    candle_core::safetensors::load(&model_path, device).map_err(Into::into)
}
