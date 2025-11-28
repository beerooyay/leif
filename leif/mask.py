"""
lexical mask construction for graph-structured attention

the lexical mask G is a binary matrix where G[i,j] = 1 iff position i
should attend to position j based on relational structure:
- same sender (continuity of speaker)
- direct address (receiver matches sender)
- temporal proximity (recent context window)
"""

import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RelationalMasking:
    """configuration for lexical mask construction"""
    
    same_sender: bool = True
    direct_address: bool = True
    temporal_window: int = 8
    include_self: bool = True


def build_lexical_mask(
    senders: torch.Tensor,
    receivers: torch.Tensor,
    timestamps: torch.Tensor,
    config: Optional[RelationalMasking] = None,
) -> torch.Tensor:
    """
    construct the lexical mask from relational coordinates.
    
    args:
        senders: (batch, seq_len) tensor of sender ids
        receivers: (batch, seq_len) tensor of receiver ids
        timestamps: (batch, seq_len) tensor of timestamps
        config: masking configuration
        
    returns:
        mask: (batch, seq_len, seq_len) binary tensor
              mask[b, i, j] = 1 iff position i can attend to position j
    """
    if config is None:
        config = RelationalMasking()
    
    batch_size, seq_len = senders.shape
    device = senders.device
    
    # start with causal mask (can only attend to past)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
    
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1):  # causal: j <= i
                should_attend = False
                
                # self attention
                if config.include_self and i == j:
                    should_attend = True
                
                # same sender: what did i say before?
                if config.same_sender:
                    if senders[b, i] == senders[b, j]:
                        should_attend = True
                
                # direct address: who is talking to me?
                if config.direct_address:
                    # position j's sender is position i's receiver
                    # (the person who sent j is who i is responding to)
                    if senders[b, j] == receivers[b, i]:
                        should_attend = True
                    # position j was addressed to position i's sender
                    if receivers[b, j] == senders[b, i]:
                        should_attend = True
                
                # temporal window: recent context
                if config.temporal_window > 0:
                    if i - j <= config.temporal_window:
                        should_attend = True
                
                mask[b, i, j] = float(should_attend)
    
    return mask


def build_lexical_mask_fast(
    senders: torch.Tensor,
    receivers: torch.Tensor,
    config: Optional[RelationalMasking] = None,
) -> torch.Tensor:
    """
    vectorized lexical mask construction (faster for large batches).
    
    args:
        senders: (batch, seq_len) tensor of sender ids
        receivers: (batch, seq_len) tensor of receiver ids
        config: masking configuration
        
    returns:
        mask: (batch, seq_len, seq_len) binary tensor
    """
    if config is None:
        config = RelationalMasking()
    
    batch_size, seq_len = senders.shape
    device = senders.device
    
    # expand for pairwise comparison: (batch, seq_len, 1) vs (batch, 1, seq_len)
    senders_i = senders.unsqueeze(2)  # (batch, seq_len, 1) - query positions
    senders_j = senders.unsqueeze(1)  # (batch, 1, seq_len) - key positions
    receivers_i = receivers.unsqueeze(2)
    receivers_j = receivers.unsqueeze(1)
    
    # causal mask
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    # initialize mask
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
    
    # self attention
    if config.include_self:
        mask = mask | torch.eye(seq_len, device=device, dtype=torch.bool)
    
    # same sender
    if config.same_sender:
        same_sender = (senders_i == senders_j)
        mask = mask | same_sender
    
    # direct address
    if config.direct_address:
        # j's sender is i's receiver (i is responding to j)
        direct_1 = (senders_j == receivers_i)
        # j was addressed to i's sender
        direct_2 = (receivers_j == senders_i)
        mask = mask | direct_1 | direct_2
    
    # temporal window
    if config.temporal_window > 0:
        positions = torch.arange(seq_len, device=device)
        pos_i = positions.unsqueeze(1)  # (seq_len, 1)
        pos_j = positions.unsqueeze(0)  # (1, seq_len)
        temporal = (pos_i - pos_j <= config.temporal_window) & (pos_i >= pos_j)
        mask = mask | temporal
    
    # apply causal constraint
    mask = mask & causal
    
    return mask.float()


def compute_attention_density(mask: torch.Tensor) -> float:
    """compute the fraction of non-zero entries in the mask"""
    seq_len = mask.shape[-1]
    # only count lower triangle (causal)
    causal_entries = seq_len * (seq_len + 1) / 2
    nonzero = mask.sum().item()
    total = mask.shape[0] * causal_entries
    return nonzero / total
