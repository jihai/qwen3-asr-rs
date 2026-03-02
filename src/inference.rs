use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::config::AsrConfig;
use crate::decoder::{compute_mrope_cos_sin, create_causal_mask, KvCache, TextDecoder};
use crate::encoder::AudioEncoder;
use crate::mel::{load_audio_wav, MelExtractor};

// Special token IDs
const IM_END_TOKEN_ID: i64 = 151645;
const ENDOFTEXT_TOKEN_ID: i64 = 151643;
const AUDIO_PAD_TOKEN_ID: i64 = 151676;
// ASR-specific separator token (not in base Qwen3 tokenizer vocab, hence decodes to "")
const ASR_TEXT_SEP_TOKEN_ID: u32 = 151704;

const MEL_SAMPLE_RATE: u32 = 16000;

pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

pub struct AsrInference {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    mel_extractor: MelExtractor,
    tokenizer: tokenizers::Tokenizer,
    config: AsrConfig,
    device: Device,
}

impl AsrInference {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        eprintln!("Loading config...");
        let config =
            AsrConfig::from_file(&model_dir.join("config.json")).context("load config")?;

        eprintln!("Loading weights (this may take a moment)...");
        let weights = load_weights(model_dir, &device).context("load weights")?;
        eprintln!("Loaded {} weight tensors", weights.len());

        eprintln!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(
            &weights,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
            &device,
        )
        .context("load audio encoder")?;

        eprintln!("Loading text decoder...");
        let text_decoder = TextDecoder::load(
            &weights,
            "thinker.model",
            &config.thinker_config.text_config,
        )
        .context("load text decoder")?;

        eprintln!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))?;

        let mel_extractor = MelExtractor::new(
            400,
            160,
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
        );

        eprintln!("Model loaded successfully.");
        Ok(Self { audio_encoder, text_decoder, mel_extractor, tokenizer, config, device })
    }

    pub fn transcribe(&self, audio_path: &str, language: Option<&str>) -> Result<TranscribeResult> {
        // Step 1: Load audio
        eprintln!("Loading audio: {}", audio_path);
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        eprintln!("Audio: {} samples @ {}Hz", samples.len(), MEL_SAMPLE_RATE);

        // Step 2: Mel spectrogram
        let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(&samples)?;
        eprintln!("Mel: {}×{} frames", n_mels, n_frames);
        let mel = Tensor::from_vec(mel_data, (n_mels, n_frames), &self.device)?;

        // Step 3: Audio encoder
        let audio_embeds = self.audio_encoder.forward(&mel)?;
        let num_audio_tokens = audio_embeds.dims()[0];
        eprintln!("Audio tokens: {}", num_audio_tokens);

        // Step 4: Build prompt token IDs
        let (input_ids, audio_start_pos) = self.build_prompt(num_audio_tokens, language)?;
        let seq_len = input_ids.len();

        // Step 5: Build embeddings, inject audio at the audio pad positions
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
        // Keep audio embeddings in their native dtype (BF16) to match embed dtype.
        let audio_emb = audio_embeds.to_dtype(before_emb.dtype())?;

        let hidden_states =
            Tensor::cat(&[&before_emb, &audio_emb, &after_emb], 0)?.unsqueeze(0)?;
        // hidden_states: [1, seq_len, hidden]

        // Step 6: MRoPE position IDs (all 3 sections use the same linear positions)
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let position_ids: [Vec<i64>; 3] =
            [positions.clone(), positions.clone(), positions.clone()];

        let text_cfg = self.config.thinker_config.text_config.clone();
        let (cos, sin) = compute_mrope_cos_sin(
            &position_ids,
            text_cfg.head_dim,
            text_cfg.rope_theta,
            &text_cfg.mrope_section(),
            text_cfg.mrope_interleaved(),
            &self.device,
        )?;

        // Step 7: Prefill
        let mask = create_causal_mask(seq_len, 0, &self.device)?;
        let mut kv_cache = KvCache::new(text_cfg.num_hidden_layers);

        let logits = self.text_decoder.forward(
            &hidden_states,
            &cos,
            &sin,
            &mut kv_cache,
            Some(&mask),
        )?;

        // Step 8: Autoregressive generation
        let mut generated_ids: Vec<u32> = Vec::new();
        let max_new_tokens = 512;
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        // logits: [1, seq_len, vocab]
        let mut next_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [1, vocab]

        let mut current_pos = seq_len;

        let debug_sample = std::env::var("DEBUG_LOGITS").is_ok();
        let mut step_idx = 0usize;

        for _ in 0..max_new_tokens {
            let next_token = next_logits.argmax(1)?.to_vec1::<u32>()?[0];

            // Debug: print top-10 logits at each step
            if debug_sample {
                let logits_f32 = next_logits.to_dtype(candle_core::DType::F32)?;
                let logits_vec = logits_f32.to_vec2::<f32>()?[0].clone();
                let mut indexed: Vec<(f32, u32)> = logits_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v, i as u32))
                    .collect();
                indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                eprint!("  step {:2}: top10 =", step_idx);
                for (score, tok) in indexed.iter().take(10) {
                    eprint!(" {}({:.2})", tok, score);
                }
                eprintln!("  chosen={}", next_token);
            }
            step_idx += 1;

            if eos_ids.contains(&(next_token as i64)) {
                break;
            }

            generated_ids.push(next_token);

            // Embed next token
            let next_id_t =
                Tensor::from_vec(vec![next_token], (1,), &self.device)?;
            let next_emb = self.text_decoder.embed(&next_id_t)?.unsqueeze(0)?; // [1, 1, hidden]

            // MRoPE for single new token
            let new_pos: [Vec<i64>; 3] = [
                vec![current_pos as i64],
                vec![current_pos as i64],
                vec![current_pos as i64],
            ];
            let (new_cos, new_sin) = compute_mrope_cos_sin(
                &new_pos,
                text_cfg.head_dim,
                text_cfg.rope_theta,
                &text_cfg.mrope_section(),
                text_cfg.mrope_interleaved(),
                &self.device,
            )?;

            let past_len = kv_cache.seq_len();
            let step_mask = create_causal_mask(1, past_len, &self.device)?;

            let step_logits = self.text_decoder.forward(
                &next_emb,
                &new_cos,
                &new_sin,
                &mut kv_cache,
                Some(&step_mask),
            )?;

            next_logits = step_logits.squeeze(1)?; // [1, vocab]
            current_pos += 1;
        }

        // Step 9: Decode
        eprintln!("Generated {} tokens", generated_ids.len());
        let raw_text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {}", e))?;

        // Token 151704 is the ASR text separator in Qwen3-ASR but is absent from
        // the base Qwen3-0.6B tokenizer (decodes to ""). Split on it directly.
        let (lang, text) = if language.is_some() {
            ("forced".to_string(), raw_text.trim().to_string())
        } else if let Some(sep_pos) = generated_ids.iter().position(|&id| id == ASR_TEXT_SEP_TOKEN_ID) {
            let lang_ids: Vec<u32> = generated_ids[..sep_pos].to_vec();
            let text_ids: Vec<u32> = generated_ids[sep_pos + 1..].to_vec();
            let lang_raw = self.tokenizer.decode(&lang_ids, true)
                .map_err(|e| anyhow::anyhow!("decode lang: {}", e))?;
            let text_raw = self.tokenizer.decode(&text_ids, true)
                .map_err(|e| anyhow::anyhow!("decode text: {}", e))?;
            // lang_raw is like "language English" → strip prefix
            let lang = lang_raw.strip_prefix("language ").unwrap_or(&lang_raw).trim().to_string();
            (lang, text_raw.trim().to_string())
        } else {
            parse_asr_output(&raw_text, false)
        };
        Ok(TranscribeResult { text, language: lang, raw_output: raw_text })
    }

    fn build_prompt(
        &self,
        num_audio_tokens: usize,
        language: Option<&str>,
    ) -> Result<(Vec<i64>, usize)> {
        let mut tokens: Vec<i64> = vec![
            151644, // <|im_start|>
            8948,   // system
            198,    // \n
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
            872,    // user
            198,    // \n
            151669, // <|audio_start|>
        ];

        let audio_start_pos = tokens.len();
        for _ in 0..num_audio_tokens {
            tokens.push(AUDIO_PAD_TOKEN_ID);
        }

        tokens.extend_from_slice(&[
            151670, // <|audio_end|>
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
        ]);

        if let Some(lang) = language {
            tokens.push(77091); // assistant
            tokens.push(198);
            let prefix = format!("language {}", capitalize_first(lang));
            let enc = self
                .tokenizer
                .encode(prefix.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        } else {
            tokens.push(77091); // assistant
            tokens.push(198);
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

/// Load safetensors weights from a directory (single file or sharded).
fn load_weights(model_dir: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
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
