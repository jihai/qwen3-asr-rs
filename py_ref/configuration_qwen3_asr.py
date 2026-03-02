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
# Source: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py

from transformers import PretrainedConfig


class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRAudioEncoder`]. It is used to instantiate a
    Qwen3-ASR audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.

    e.g. [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Qwen3ASRProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        n_window (`int`, *optional*, defaults to 100):
            The chunk for conv and flash attn in AudioEncoder.
            NOTE: For the 0.6B model, this is 50 (not 100 as the default).
        output_dim (`int`, *optional*, defaults to 3584):
            The output dimension of AudioEncoder.
            NOTE: For the 0.6B model, this is 1024.
        n_window_infer (`int`, *optional*, defaults to 400):
            Window size for windowed attention during inference.
            NOTE: For the 0.6B model, this is 800.
        conv_chunksize (`int`, *optional*, defaults to 500):
            Chunk size for batched Conv2d processing.
        downsample_hidden_size (`int`, *optional*, defaults to 480):
            Number of channels in the Conv2d stem layers.

    Example:

    ```python
    >>> from transformers import Qwen3ASRAudioEncoderConfig, Qwen3ASRAudioEncoder

    >>> # Initializing a Qwen3ASRAudioEncoderConfig
    >>> configuration = Qwen3ASRAudioEncoderConfig()

    >>> # Initializing a Qwen3ASRAudioEncoder (with random weights)
    >>> model = Qwen3ASRAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    # 0.6B model config.json values (audio_config section):
    # d_model: 896
    # encoder_layers: 18
    # encoder_attention_heads: 14
    # encoder_ffn_dim: 3584
    # num_mel_bins: 128
    # downsample_hidden_size: 480
    # output_dim: 1024
    # conv_chunksize: 500
    # n_window: 50
    # n_window_infer: 800
    # max_source_positions: 1500
    """

    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


__all__ = ["Qwen3ASRAudioEncoderConfig"]
