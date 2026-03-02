use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use std::collections::HashMap;

use crate::config::AudioEncoderConfig;

fn get_weight(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    weights
        .get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
}

fn get_weight_opt(weights: &HashMap<String, Tensor>, name: &str) -> Option<Tensor> {
    weights.get(name).cloned()
}

// ─── LayerNorm (with bias, used in audio encoder) ────────────────────────────

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, &format!("{}.weight", prefix))?,
            bias: get_weight(weights, &format!("{}.bias", prefix))?,
            eps,
        })
    }

    // Compute in F32 for numerical stability, cast result back to input dtype.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let last_dim = x_f32.rank() - 1;
        let mean = x_f32.mean_keepdim(last_dim)?;
        let diff = x_f32.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(last_dim)?;
        let std = (var + self.eps)?.sqrt()?;
        let normed = diff.broadcast_div(&std)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let b = self.bias.to_dtype(DType::F32)?;
        let result = normed.broadcast_mul(&w)?.broadcast_add(&b)?;
        result.to_dtype(orig_dtype).map_err(Into::into)
    }
}

// ─── Audio Encoder Self-Attention ─────────────────────────────────────────────

struct AudioAttention {
    q_proj_w: Tensor,
    q_proj_b: Tensor,
    k_proj_w: Tensor,
    k_proj_b: Tensor,
    v_proj_w: Tensor,
    v_proj_b: Tensor,
    out_proj_w: Tensor,
    out_proj_b: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let qp = format!("{}.q_proj", prefix);
        let kp = format!("{}.k_proj", prefix);
        let vp = format!("{}.v_proj", prefix);
        let op = format!("{}.out_proj", prefix);
        Ok(Self {
            q_proj_w: get_weight(weights, &format!("{}.weight", qp))?,
            q_proj_b: get_weight(weights, &format!("{}.bias", qp))?,
            k_proj_w: get_weight(weights, &format!("{}.weight", kp))?,
            k_proj_b: get_weight(weights, &format!("{}.bias", kp))?,
            v_proj_w: get_weight(weights, &format!("{}.weight", vp))?,
            v_proj_b: get_weight(weights, &format!("{}.bias", vp))?,
            out_proj_w: get_weight(weights, &format!("{}.weight", op))?,
            out_proj_b: get_weight(weights, &format!("{}.bias", op))?,
            num_heads,
            head_dim,
        })
    }

    // Use the weight's native dtype (BF16) for matmul to match the reference.
    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let dtype = w.dtype();
        let x_cast = x.to_dtype(dtype)?;
        let out_features = w.dims()[0];
        let dims = x_cast.dims().to_vec();
        let in_features = dims[dims.len() - 1];
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let out_2d = x_cast.reshape((batch, in_features))?.matmul(&w.t()?)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_features);
        Ok(out_2d.reshape(out_shape)?.broadcast_add(&b.to_dtype(dtype)?)?)
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self.linear(x, &self.q_proj_w, &self.q_proj_b)?
            .reshape((bsz, seq_len, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.linear(x, &self.k_proj_w, &self.k_proj_b)?
            .reshape((bsz, seq_len, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self.linear(x, &self.v_proj_w, &self.v_proj_b)?
            .reshape((bsz, seq_len, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (hd as f64).sqrt();
        let mut attn: Tensor = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / scale))?;

        if let Some(m) = mask {
            attn = attn.broadcast_add(m)?;
        }

        // Softmax in F32 for numerical stability, cast back to compute dtype.
        let attn_dtype = attn.dtype();
        let attn = softmax(&attn.to_dtype(DType::F32)?, 3)?.to_dtype(attn_dtype)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((bsz, seq_len, nh * hd))?;
        Ok(self.linear(&out, &self.out_proj_w, &self.out_proj_b)?)
    }
}

// ─── Audio Encoder FFN ────────────────────────────────────────────────────────

struct AudioFfn {
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

impl AudioFfn {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1_w: get_weight(weights, &format!("{}.fc1.weight", prefix))?,
            fc1_b: get_weight(weights, &format!("{}.fc1.bias", prefix))?,
            fc2_w: get_weight(weights, &format!("{}.fc2.weight", prefix))?,
            fc2_b: get_weight(weights, &format!("{}.fc2.bias", prefix))?,
        })
    }

    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let dtype = w.dtype();
        let x_cast = x.to_dtype(dtype)?;
        let out_features = w.dims()[0];
        let dims = x_cast.dims().to_vec();
        let in_features = dims[dims.len() - 1];
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let out_2d = x_cast.reshape((batch, in_features))?.matmul(&w.t()?)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_features);
        Ok(out_2d.reshape(out_shape)?.broadcast_add(&b.to_dtype(dtype)?)?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.linear(x, &self.fc1_w, &self.fc1_b)?;
        // GELU in F32 for safety, cast back to compute dtype.
        let dtype = hidden.dtype();
        let hidden = hidden.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(dtype)?;
        Ok(self.linear(&hidden, &self.fc2_w, &self.fc2_b)?)
    }
}

// ─── Audio Encoder Layer ──────────────────────────────────────────────────────

struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: AudioAttention,
    final_layer_norm: LayerNorm,
    ffn: AudioFfn,
}

impl AudioEncoderLayer {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: LayerNorm::load(
                weights,
                &format!("{}.self_attn_layer_norm", prefix),
                1e-5,
            )?,
            self_attn: AudioAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_heads,
                d_model,
            )?,
            final_layer_norm: LayerNorm::load(
                weights,
                &format!("{}.final_layer_norm", prefix),
                1e-5,
            )?,
            ffn: AudioFfn::load(weights, prefix)?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm + self-attention + residual
        let h = self.self_attn_layer_norm.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = (x + &h)?;

        // Pre-norm + FFN + residual
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        (&x + &h).map_err(Into::into)
    }
}

// ─── Conv2d wrapper ───────────────────────────────────────────────────────────

struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, &format!("{}.weight", prefix))?,
            bias: get_weight_opt(weights, &format!("{}.bias", prefix)),
            stride,
            padding,
        })
    }

    // Use the weight's native dtype (BF16) to match the reference implementation.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.weight.dtype();
        let w = &self.weight;
        let x_cast = x.to_dtype(dtype)?;
        let out = x_cast.conv2d(w, self.padding, self.stride, 1, 1)?;
        if let Some(b) = &self.bias {
            let (_, c, _, _) = out.dims4()?;
            let b_view = b.to_dtype(dtype)?.reshape((1, c, 1, 1))?;
            Ok(out.broadcast_add(&b_view)?)
        } else {
            Ok(out)
        }
    }
}

// ─── Linear wrapper ───────────────────────────────────────────────────────────

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, &format!("{}.weight", prefix))?,
            bias: get_weight_opt(weights, &format!("{}.bias", prefix)),
        })
    }

    // Use the weight's native dtype (BF16) for matmul to match the reference.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.weight.dtype();
        let x_cast = x.to_dtype(dtype)?;
        let out_features = self.weight.dims()[0];
        let dims = x_cast.dims().to_vec();
        let in_features = dims[dims.len() - 1];
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x_cast.reshape((batch, in_features))?;
        let out_2d = x_2d.matmul(&self.weight.t()?)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_features);
        let out = out_2d.reshape(out_shape)?;
        if let Some(b) = &self.bias {
            Ok(out.broadcast_add(&b.to_dtype(dtype)?)?)
        } else {
            Ok(out)
        }
    }
}

// ─── Sinusoidal positional embedding ─────────────────────────────────────────

fn create_sinusoidal_embedding(max_len: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = dim / 2;
    let log_timescale = (10000.0f64).ln() / (half_dim as f64 - 1.0);

    let mut embeddings = vec![0.0f32; max_len * dim];
    for pos in 0..max_len {
        for i in 0..half_dim {
            let inv_ts = (-(i as f64) * log_timescale).exp();
            let angle = pos as f64 * inv_ts;
            embeddings[pos * dim + i] = angle.sin() as f32;
            embeddings[pos * dim + half_dim + i] = angle.cos() as f32;
        }
    }

    Tensor::from_vec(embeddings, (max_len, dim), device).map_err(Into::into)
}

// ─── Audio Encoder ────────────────────────────────────────────────────────────

pub struct AudioEncoder {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    positional_embedding: Tensor,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let conv2d1 = Conv2d::load(weights, &format!("{}.conv2d1", prefix), 2, 1)?;
        let conv2d2 = Conv2d::load(weights, &format!("{}.conv2d2", prefix), 2, 1)?;
        let conv2d3 = Conv2d::load(weights, &format!("{}.conv2d3", prefix), 2, 1)?;
        let conv_out = Linear::load(weights, &format!("{}.conv_out", prefix))?;

        let mut layers = Vec::new();
        for i in 0..config.encoder_layers {
            let layer = AudioEncoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.encoder_attention_heads,
                config.d_model,
            )?;
            layers.push(layer);
        }

        let ln_post = LayerNorm::load(weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = Linear::load(weights, &format!("{}.proj1", prefix))?;
        let proj2 = Linear::load(weights, &format!("{}.proj2", prefix))?;

        let positional_embedding =
            create_sinusoidal_embedding(config.max_source_positions, config.d_model, device)?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            positional_embedding,
            layers,
            ln_post,
            proj1,
            proj2,
            config: config.clone(),
        })
    }

    /// Encode mel spectrogram [num_mel_bins, num_frames] → [num_tokens, output_dim]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let num_frames = mel.dims()[1];
        // Logical chunk = n_window * 2 = 100 mel frames (matches official windowed attention).
        let chunk_size = self.config.n_window * 2;

        let num_full = num_frames / chunk_size;
        let tail = num_frames % chunk_size;
        let num_chunks = num_full + if tail > 0 { 1 } else { 0 };

        // Collect chunks as F32 (conv2d runs in F32).
        let mut chunk_mels: Vec<Tensor> = Vec::with_capacity(num_chunks);
        let mut chunk_valid_tokens: Vec<usize> = Vec::with_capacity(num_chunks);

        for i in 0..num_full {
            let start = i * chunk_size;
            let chunk = mel.narrow(1, start, chunk_size)?.unsqueeze(0)?;
            chunk_mels.push(chunk);
            chunk_valid_tokens.push(Self::feat_extract_output_length(chunk_size));
        }

        if tail > 0 {
            let start = num_full * chunk_size;
            let tail_mel = mel.narrow(1, start, tail)?;
            let pad_frames = chunk_size - tail;
            let device = mel.device();
            // Pad with zeros in F32 (mel is F32); will be cast to BF16 with the batch.
            let pad = Tensor::zeros((mel.dims()[0], pad_frames), DType::F32, device)?;
            let padded = Tensor::cat(&[&tail_mel.to_dtype(DType::F32)?, &pad], 1)?.unsqueeze(0)?;
            chunk_mels.push(padded);
            chunk_valid_tokens.push(Self::feat_extract_output_length(tail));
        }

        // Stack chunks and cast to BF16 (weight's native dtype) for the full conv stem.
        // batched: [num_chunks, 1, mel_bins, chunk_size]
        let refs: Vec<&Tensor> = chunk_mels.iter().collect();
        let compute_dtype = self.conv2d1.weight.dtype();
        let batched = Tensor::cat(&refs, 0)?.unsqueeze(1)?.to_dtype(compute_dtype)?;

        // Conv2d in BF16, GELU in F32 for numerical safety, cast back to BF16.
        let x = self.conv2d1.forward(&batched)?;
        let x = { let d = x.dtype(); x.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(d)? };
        let x = self.conv2d2.forward(&x)?;
        let x = { let d = x.dtype(); x.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(d)? };
        let x = self.conv2d3.forward(&x)?;
        let x = { let d = x.dtype(); x.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(d)? };

        // Reshape: [b, c, f, t] -> [b, t, c*f]
        let (b, c, f, t) = x.dims4()?;
        let reshaped = x.permute((0, 3, 1, 2))?.contiguous()?.reshape((b, t, c * f))?;

        // Linear projection (F32 in → BF16 out, since conv_out weights are BF16).
        let conv_out = self.conv_out.forward(&reshaped)?;

        // Add positional embedding, cast to match conv_out's dtype (BF16).
        let pos_emb = self.positional_embedding
            .narrow(0, 0, t)?
            .unsqueeze(0)?
            .to_dtype(conv_out.dtype())?;
        let conv_out = conv_out.broadcast_add(&pos_emb)?;
        // conv_out: [num_chunks, tokens_per_chunk, d_model], dtype = BF16

        // Collect valid tokens from all chunks and concatenate.
        // The attention window (n_window_infer / chunk_size = 800/100 = 8 chunks = 104 tokens)
        // covers all tokens for short audio, so use full attention across all tokens.
        let mut all_valid: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for (idx, &valid) in chunk_valid_tokens.iter().enumerate() {
            let chunk_tokens = conv_out.i(idx)?.narrow(0, 0, valid)?;
            all_valid.push(chunk_tokens);
        }
        // Concatenate: [total_tokens, d_model]
        let refs: Vec<&Tensor> = all_valid.iter().collect();
        let hidden = Tensor::cat(&refs, 0)?;

        // Add batch dim: [1, total_tokens, d_model]
        let mut hidden = hidden.unsqueeze(0)?;

        // Transformer encoder with full attention (all tokens attend to each other).
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None)?;
        }

        // Output projection: LN → Linear → GELU → Linear
        // (hidden already has batch dim [1, total_tokens, d_model])
        let hidden = self.ln_post.forward(&hidden)?;
        let proj1_out = self.proj1.forward(&hidden)?;
        // GELU in F32 for safety, cast back to BF16.
        let proj1_dtype = proj1_out.dtype();
        let proj1_out = proj1_out.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(proj1_dtype)?;
        let hidden = self.proj2.forward(&proj1_out)?;

        // Remove batch dim: [num_tokens, output_dim]
        hidden.squeeze(0).map_err(Into::into)
    }

    fn feat_extract_output_length(input_frames: usize) -> usize {
        let after_conv = |len: usize| -> usize { (len - 1) / 2 + 1 };
        after_conv(after_conv(after_conv(input_frames)))
    }
}
