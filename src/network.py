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
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 4
COMP_EMBED = 24 # Embedding dim per instruction component

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
    

class ProgramEncoder(nn.Module):
    """
        - Description:
            Encodes the instruction history using learned embeddings + transformer block
        
        - Input:
            program section of obs (i.e., observation), shape (batch, MAX_STEPS * 5)
            Values are 0-padded integers: op in [0, 5], regs in [0, 8]
            where 0 is the padding and 1 to N are the real values

        - Outputs:
            (batch, EMBED_DIM) latent vector
    """

    embed_dim: int = EMBED_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS

    @nn.compact
    def __call__(self, prog_flat: jnp.ndarray) -> jnp.ndarray:
        batch = prog_flat.shape[0]
        # Reshape to (batch, MAX_STEPS, 5)
        prog = prog_flat.reshape(batch, MAX_STEPS, 5).astype(jnp.int32)

        # Embed each component: 0 is padding token
        # op: vocab size 6 (0=pad, 1..5=ops)
        # regs: vocab size 9 (0=pad, 1..8=regs)
        emb_op  = nn.Embed(num_embeddings=NUM_OPS + 1, features=COMP_EMBED)(prog[:, :, 0])
        reg_emb = nn.Embed(num_embeddings=NUM_REGS + 1, features=COMP_EMBED)
        emb_rd  = reg_emb(prog[:, :, 1])
        emb_rs1 = reg_emb(prog[:, :, 2])
        emb_rs2 = reg_emb(prog[:, :, 3])
        emb_rs3 = reg_emb(prog[:, :, 4])

        # Concatenate → (batch, MAX_STEPS, 5 * COMP_EMBED)
        step_tokens = jnp.concatenate([emb_op, emb_rd, emb_rs1, emb_rs2, emb_rs3], axis=-1)

        # Project to model dimension
        x = nn.Dense(self.embed_dim)(step_tokens)  # (batch, MAX_STEPS, embed_dim)

        # Learned positional encoding
        pos = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, MAX_STEPS, self.embed_dim)
        )
        x = x + pos

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
            )(x)

        # Final norm + mean pool over sequence
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)  # (batch, embed_dim)

        return x