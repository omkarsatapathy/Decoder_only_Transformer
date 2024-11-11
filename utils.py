# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttentionOutput:
    """Container for attention output"""
    attention_output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class LayerNorm(nn.Module):
    """Layer normalization with optional bias"""

    def __init__(self, d_model: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        if self.beta is None:
            return self.gamma * normalized
        return self.gamma * normalized + self.beta


class MultiHeadAttention(nn.Module):
    """Improved Multi-head attention with proper initialization and scaling"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.model.num_head
        self.d_model = config.model.d_model
        self.head_dim = self.d_model // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # Single linear layer for all projections
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Initialize weights
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False
    ) -> AttentionOutput:
        batch_size, seq_len, _ = x.shape

        # Calculate Q, K, V projections simultaneously
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attention_output = self.out_proj(attention_output)

        if need_weights:
            return AttentionOutput(attention_output, attention_weights)
        return AttentionOutput(attention_output)


class FeedForward(nn.Module):
    """Improved feed-forward network with GELU activation and proper initialization"""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.model.d_model, config.model.ffn_hidden)
        self.fc2 = nn.Linear(config.model.ffn_hidden, config.model.d_model)
        self.dropout = nn.Dropout(config.model.drop_prob)
        self.activation = nn.GELU()

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """Improved Transformer encoder layer with Pre-LN architecture"""

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        # Layer normalization layers (Pre-LN architecture)
        self.norm1 = LayerNorm(config.model.d_model)
        self.norm2 = LayerNorm(config.model.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.model.drop_prob)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN architecture
        # Layer 1: Multi-head attention
        residual = x
        x = self.norm1(x)
        attention_output = self.attention(x, attention_mask)
        x = residual + self.dropout(attention_output.attention_output)

        # Layer 2: Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x


class Encoder(nn.Module):
    """Improved Transformer encoder with proper embedding scaling and positional encoding"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding with proper scaling
        self.token_embedding = nn.Embedding(config.model.vocab_size, config.model.d_model)
        self.embed_scale = config.model.d_model ** 0.5

        # Positional encoding
        self.pos_embedding = self._create_positional_embedding()

        # Dropout
        self.dropout = nn.Dropout(config.model.drop_prob)

        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.model.num_layers)
        ])

        # Layer normalization
        self.norm = LayerNorm(config.model.d_model)

        # Classifier
        self.classifier = nn.Linear(config.model.d_model, config.model.num_classes)

        # Initialize weights
        self._init_weights()

    def _create_positional_embedding(self) -> torch.Tensor:
        """Create sinusoidal positional embeddings"""
        max_seq_len = self.config.model.max_seq_len
        d_model = self.config.model.d_model

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get sequence length and create position ids
        seq_len = x.size(1)

        # Embed tokens and positions
        x = self.token_embedding(x) * self.embed_scale
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Apply final layer normalization
        x = self.norm(x)

        # Pool and classify
        # Using mean pooling instead of just taking the first token
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

    def get_attention_weights(
            self,
            x: torch.Tensor,
            layer_idx: int = -1
    ) -> torch.Tensor:
        """Get attention weights for visualization"""
        assert 0 <= layer_idx < len(self.layers), "Invalid layer index"

        # Embed tokens and positions
        x = self.token_embedding(x) * self.embed_scale
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        # Apply encoder layers until the requested layer
        for i, layer in enumerate(self.layers):
            if i == layer_idx:
                # Get attention weights from the requested layer
                attention_output = layer.attention(
                    layer.norm1(x),
                    need_weights=True
                )
                return attention_output.attention_weights
            x = layer(x)

        return None