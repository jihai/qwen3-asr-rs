# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Source: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py
#
# NOTE: This file reconstructs the key classes from the QwenLM Qwen3-ASR repository.
# The full file is ~1400+ lines. The sections below are the audio encoder pipeline
# components that are directly relevant to mel feature extraction and audio encoding.

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import GradientCheckpointingLayer

from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig


# ---------------------------------------------------------------------------
# Utility: output length calculation (mirrors processing_qwen3_asr.py)
# ---------------------------------------------------------------------------

def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder.

    The encoder splits mel frames into chunks of n_window*2 = 100 frames.
    Each full 100-frame chunk -> 13 audio tokens after the 3x Conv2d stem (8x downsample).
    The remainder chunk (< 100 frames) is handled by the strided-conv formula below.

    Formula:
        remainder = input_lengths % 100
        after_conv1 = (remainder - 1) // 2 + 1       # stride-2
        after_conv2 = (after_conv1 - 1) // 2 + 1     # stride-2
        after_conv3 = (after_conv2 - 1) // 2 + 1     # stride-2
        total = after_conv3 + (input_lengths // 100) * 13
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (per-chunk, positions reset each chunk)
# ---------------------------------------------------------------------------

class SinusoidsPositionEmbedding(nn.Module):
    """
    Standard sinusoidal positional embeddings, pre-computed and stored as a buffer.

    Key design choice: positions are applied PER CHUNK, so each 100-frame chunk
    (which produces 13 tokens) receives positions starting from 0. This means
    the audio encoder has no concept of global position across chunks.

    Parameters:
        length (int): max_source_positions from config (1500 for 0.6B)
        channels (int): d_model (896 for 0.6B)
        max_timescale (float): 10000 (standard Transformer default)

    Shape of positional_embedding buffer: [length, channels]
    """
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


# ---------------------------------------------------------------------------
# Audio attention
# ---------------------------------------------------------------------------

class Qwen3ASRAudioAttention(nn.Module):
    """
    Multi-head attention for the audio encoder.

    Uses standard (non-RoPE) attention. Supports eager, flash_attention_2, and sdpa backends.
    Attention is windowed: tokens only attend within cu_seqlens boundaries (no cross-chunk attention).

    Config (0.6B):
        d_model: 896
        encoder_attention_heads: 14
        head_dim: 896 / 14 = 64
    """
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim} and "
                f"num_heads={self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # hidden_states: [total_tokens, d_model]  (packed/unpadded format)
        # cu_seqlens: cumulative sequence lengths for windowed attention
        # Standard QKV projection + multi-head attention
        # (implementation uses varlen flash attention or sdpa with masking)
        raise NotImplementedError(
            "See the full file at: "
            "https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py"
        )


# ---------------------------------------------------------------------------
# Audio encoder layer
# ---------------------------------------------------------------------------

class Qwen3ASRAudioEncoderLayer(GradientCheckpointingLayer):
    """
    Single transformer encoder layer for the audio encoder.

    Pre-norm architecture (LayerNorm before attention and FFN).
    LayerNorm includes bias (unlike the text decoder which is bias-free RMSNorm).

    Config (0.6B):
        d_model: 896
        encoder_attention_heads: 14
        encoder_ffn_dim: 3584
        activation_function: "gelu"
        dropout: 0.0
    """
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # fp16 stability clamp
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        return outputs


# ---------------------------------------------------------------------------
# Audio encoder (complete pipeline)
# ---------------------------------------------------------------------------

class Qwen3ASRAudioEncoder(PreTrainedModel):
    """
    Complete audio encoding pipeline for Qwen3-ASR.

    Pipeline summary (0.6B model):
    ================================
    Input: mel spectrogram [128 mel bins, T frames]
      - T = input_length_frames (at 100 Hz, e.g. 160 frames for 1.6s audio)

    Step 1 - Chunking:
      - Split T mel frames into chunks of n_window*2 = 50*2 = 100 frames each
      - Last chunk may be shorter (padded to full chunk size for batching)

    Step 2 - Conv2d stem (per chunk, batched):
      - Input shape: [num_chunks, 1, 128, 100]   (1 channel, 128 freq bins, 100 time frames)
      - conv2d1: Conv2d(1, 480, 3x3, stride=2, padding=1) + GELU
          -> [num_chunks, 480, 64, 50]
      - conv2d2: Conv2d(480, 480, 3x3, stride=2, padding=1) + GELU
          -> [num_chunks, 480, 32, 25]
      - conv2d3: Conv2d(480, 480, 3x3, stride=2, padding=1) + GELU
          -> [num_chunks, 480, 16, 13]    (100 frames -> 13 time steps, 128 freq -> 16 bins)
      - Reshape: permute(0,3,1,2).view(num_chunks, 13, 480*16)
          -> [num_chunks, 13, 7680]
      - conv_out: Linear(7680, 896, bias=False)
          -> [num_chunks, 13, 896]        (d_model for 0.6B)

    Step 3 - Positional embeddings:
      - Add sinusoidal PE of shape [13, 896] to each chunk's 13 tokens
      - Positions reset to [0..12] for EVERY chunk (not global positions!)

    Step 4 - Unpad and pack:
      - Remove padding tokens using padded_mask_after_cnn
      - Pack valid tokens: [total_valid_tokens, 896]

    Step 5 - Transformer encoder (18 layers for 0.6B):
      - Windowed self-attention using cu_seqlens to define attention boundaries
      - Window size (n_window_infer=800 for 0.6B): controls how many tokens
        attend to each other across adjacent chunks
      - Each layer: LayerNorm -> Attention -> Add + LayerNorm -> FFN -> Add

    Step 6 - Output projections:
      - ln_post: LayerNorm(896)
      - proj1: Linear(896, 896)
      - act: GELU
      - proj2: Linear(896, 1024)   (output_dim for 0.6B)
      - Final output: [total_valid_tokens, 1024]

    These audio tokens are then inserted into the LLM text sequence at the
    position of audio_token_id (151676), replacing the placeholder.

    Config values for 0.6B model (from config.json audio_config):
        d_model: 896
        encoder_layers: 18
        encoder_attention_heads: 14
        encoder_ffn_dim: 3584
        num_mel_bins: 128
        downsample_hidden_size: 480
        output_dim: 1024
        conv_chunksize: 500
        n_window: 50          <- chunk size = n_window * 2 = 100 mel frames
        n_window_infer: 800   <- attention window in tokens
        max_source_positions: 1500
        scale_embedding: False
        activation_function: "gelu"
    """

    config: Qwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins       # 128
        self.max_source_positions = config.max_source_positions  # 1500
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window               # 50 for 0.6B

        # Sinusoidal PE: [max_source_positions, d_model] = [1500, 896]
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)

        # 18 transformer encoder layers (0.6B)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.ln_post = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

        # Conv2d stem: 3 layers, each stride=2 -> total 8x temporal downsampling
        # freq dimension: 128 -> 64 -> 32 -> 16
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)

        # After conv: channels=480, freq=16 -> flatten to 480*16=7680, project to d_model=896
        # Formula: ((num_mel_bins+1)//2 + 1)//2 + 1)//2 = ((128+1)//2+1)//2+1)//2
        #        = ((64+1)//2+1)//2+1)//2 = (32+1)//2+1)//2 = (16+1)//2 = ... = 16
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )

        # Output projections: d_model -> d_model -> output_dim
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

        self.n_window_infer = self.config.n_window_infer   # 800 for 0.6B
        self.conv_chunksize = self.config.conv_chunksize   # 500

        self.post_init()

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        """
        Args:
            input_features: [1, num_mel_bins, T] = [1, 128, T]
                Mel spectrogram from WhisperFeatureExtractor.
                Note: input is passed transposed/packed; see chunking below.
            feature_lens: [batch] tensor of mel frame counts per sample
            aftercnn_lens: [batch] tensor of token counts after Conv2d (computed internally)

        Returns:
            BaseModelOutput with last_hidden_state: [total_valid_tokens, output_dim]
        """
        # Step 1: Compute lengths after CNN
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)

        # Step 2: Chunk the mel spectrogram into n_window*2 = 100-frame chunks
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        # Build per-chunk lengths: full chunks get 100, last chunk gets the remainder
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        # tail_chunk_index: index of last chunk for each sample in the flattened chunk list
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        # If remainder is exactly 0, the tail chunk is actually a full chunk
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        # Split along time axis: input_features.T has shape [T, num_mel_bins]
        # chunk_list: list of tensors, each [chunk_len, num_mel_bins]
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)

        # Pad chunks to same length and stack: [num_chunks, num_mel_bins, max_chunk_len]
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)

        # Compute per-chunk output token counts (for mask)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )

        # Add channel dim: [num_chunks, 1, num_mel_bins, max_chunk_len]
        padded_feature = padded_feature.unsqueeze(1)

        # Step 3: Apply Conv2d stem (batched in conv_chunksize=500 chunk sub-batches)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))   # stride=2: freq 128->64, time 100->50
            padded_embed = F.gelu(self.conv2d2(padded_embed))  # stride=2: freq 64->32, time 50->25
            padded_embed = F.gelu(self.conv2d3(padded_embed))  # stride=2: freq 32->16, time 25->13
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        # padded_embed: [num_chunks, 480, 16, 13]

        # Reshape: [num_chunks, 480, 16, 13] -> [num_chunks, 13, 480*16] -> [num_chunks, 13, 7680]
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        # padded_embed: [num_chunks, 13, d_model=896]

        # Step 4: Add sinusoidal positional embeddings (per-chunk, positions reset to 0..12)
        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        # positional_embedding shape: [1, 13, 896] (broadcast across chunks)

        # Step 5: Unpad - select only valid tokens using the mask
        # hidden_states: [total_valid_tokens, d_model=896]
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Step 6: Compute cu_seqlens for windowed attention
        # window_aftercnn = tokens per attention window
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        # Step 7: Run through 18 transformer encoder layers
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            hidden_states = layer_outputs[0]

        # Step 8: Output projections
        hidden_states = self.ln_post(hidden_states)    # LayerNorm
        hidden_states = self.proj1(hidden_states)      # Linear(896, 896)
        hidden_states = self.act(hidden_states)         # GELU
        hidden_states = self.proj2(hidden_states)      # Linear(896, 1024)

        return BaseModelOutput(last_hidden_state=hidden_states)


__all__ = [
    "SinusoidsPositionEmbedding",
    "Qwen3ASRAudioAttention",
    "Qwen3ASRAudioEncoderLayer",
    "Qwen3ASRAudioEncoder",
    "_get_feat_extract_output_lengths",
]
