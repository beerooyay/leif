"""
leif model architecture: graph-structured attention over lexia

leif-nano specification:
- 6 transformer layers
- 4 attention heads
- d_model = 256
- separate embeddings for token, sender, receiver, conduit, time
- graph-structured attention via lexical mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mask import build_lexical_mask_fast, RelationalMasking


class GraphStructuredAttention(nn.Module):
    """
    multi-head attention with lexical mask.
    
    attention is computed as:
        attn(Q, K, V) = softmax((QK^T / sqrt(d)) * G) V
    
    where G is the lexical mask (binary, derived from relational coordinates).
    positions with G[i,j] = 0 receive -inf before softmax, zeroing their weight.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) lexical mask, 1 = attend, 0 = block
            
        returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # reshape for multi-head: (batch, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # attention scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # apply lexical mask
        if mask is not None:
            # expand mask for heads: (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)
            # set masked positions to -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # reshape back: (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output


class TransformerBlock(nn.Module):
    """standard transformer block with graph-structured attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = GraphStructuredAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm architecture
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.ff(self.norm2(x))
        return x


class LeifModel(nn.Module):
    """
    leif: lexia-native language model with graph-structured attention.
    
    input: sequences of lexia (sender, receiver, conduit, time, token)
    output: next-token predictions
    
    the lexical mask is constructed from relational coordinates and
    applied to attention, implementing graph-structured attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        n_agents: int = 16,
        n_conduits: int = 4,
        dropout: float = 0.1,
        mask_config: Optional[RelationalMasking] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_config = mask_config or RelationalMasking()
        
        # embedding dimensions (scale with d_model)
        # allocate ~55% to token, rest to relational coordinates
        d_sender = max(8, d_model // 8)
        d_receiver = max(8, d_model // 8)
        d_conduit = max(4, d_model // 16)
        d_time = max(8, d_model // 8)
        d_token = d_model - d_sender - d_receiver - d_conduit - d_time
        
        # embeddings
        self.token_embed = nn.Embedding(vocab_size, d_token)
        self.sender_embed = nn.Embedding(n_agents, d_sender)
        self.receiver_embed = nn.Embedding(n_agents, d_receiver)
        self.conduit_embed = nn.Embedding(n_conduits, d_conduit)
        self.time_embed = nn.Embedding(max_seq_len, d_time)
        
        # projection to d_model (in case dimensions don't sum exactly)
        self.input_proj = nn.Linear(d_token + d_sender + d_receiver + d_conduit + d_time, d_model)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conduits: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        args:
            tokens: (batch, seq_len) token ids
            senders: (batch, seq_len) sender agent ids
            receivers: (batch, seq_len) receiver agent ids
            conduits: (batch, seq_len) conduit ids
            positions: (batch, seq_len) position indices (optional, defaults to 0..seq_len-1)
            
        returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # default positions
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # embed all components
        tok_emb = self.token_embed(tokens)
        send_emb = self.sender_embed(senders)
        recv_emb = self.receiver_embed(receivers)
        cond_emb = self.conduit_embed(conduits)
        time_emb = self.time_embed(positions)
        
        # concatenate and project
        x = torch.cat([tok_emb, send_emb, recv_emb, cond_emb, time_emb], dim=-1)
        x = self.input_proj(x)
        
        # build lexical mask from relational coordinates
        mask = build_lexical_mask_fast(senders, receivers, self.mask_config)
        
        # transformer blocks with graph-structured attention
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    def get_attention_density(self, senders: torch.Tensor, receivers: torch.Tensor) -> float:
        """compute attention density for given relational coordinates"""
        mask = build_lexical_mask_fast(senders, receivers, self.mask_config)
        seq_len = mask.shape[-1]
        causal_entries = seq_len * (seq_len + 1) / 2
        nonzero = mask.sum().item()
        total = mask.shape[0] * causal_entries
        return nonzero / total


class BaselineTransformer(nn.Module):
    """
    standard transformer baseline with dense attention.
    
    same architecture as leif but:
    - no relational embeddings (sender, receiver, conduit)
    - dense causal attention (no lexical mask)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # embeddings (token + position only)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # manual positional embedding to avoid MPS bug
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        **kwargs,  # ignore relational coordinates
    ) -> torch.Tensor:
        """
        args:
            tokens: (batch, seq_len) token ids
            positions: (batch, seq_len) position indices
            
        returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # embed
        token_emb = self.token_embed(tokens)
        pos_emb = self.pos_embed[positions.long()]  # manual lookup
        x = token_emb + pos_emb
        
        # causal mask (dense, lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
