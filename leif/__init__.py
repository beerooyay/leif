"""
leif: lexia-native language modeling with graph-structured attention
"""

from .model import LeifModel, BaselineTransformer
from .mask import build_lexical_mask, build_lexical_mask_fast, RelationalMasking
from .data import LexiaDataset, Lexia, generate_synthetic_dialogue

__version__ = "0.1.0"
__all__ = [
    "LeifModel",
    "BaselineTransformer", 
    "build_lexical_mask",
    "RelationalMasking",
    "LexiaDataset",
    "Lexia",
    "generate_synthetic_dialogue",
]
