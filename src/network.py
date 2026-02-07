""""
    AlphaDev Neural Network
    
    Autoregressive Policy Head: op -> rd -> rs1 -> rs2 -> rs3
    Each sub-decision is conditioned on the previous ones.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


# ─── Architecture Componets ───────────────────────────────────
# Must match the C++ environemt

NUM_OPS = 5
NUM_REGS = 8
MAX_STEPS = 20
OBS_REG_SIZE = 48 # 6 test cases × 8 regs
OBS_PROG_SIZE = 100 # 20 steps × 5 components
OBS_META_SIZE = 1
OBS_TOTAL_SIZE = OBS_REG_SIZE + OBS_PROG_SIZE + OBS_META_SIZE  # 149

# Network dimensions
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 3
NUM_HEADS   = 4
COMP_EMBED  = 24 # Embedding dim per instruction component

class TransformerBlock(nn.Module):
    """
        - Description:
            Pre-norm Transformer Encoder Block
    """


    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Self attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads = self.num_heads,
            qkv_features = self.embed_dim,
            deterministic = True,
        )(x, x)

        x = x + residual

        # Feed forward
        residual = x
        x = nn.LayerNorm()(x)
        mlp_dim = int(self.embed_dim * self.mlp_ratio)
        x = nn.Dense(mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = x + residual

        return x