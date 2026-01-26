import jax
import jax.numpy as jnp
from flax import linen as nn

# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 4
DROPOUT_RATE = 0.1

# Architecture Constants
NUM_OPS = 5     # ADD, SUB, AND, SLT, CMOV
NUM_REGS = 8    # x0..x7
MAX_STEPS = 10  # From assembly_env.hpp
NUM_TEST_CASES = 6 

class TransformerBlock(nn.Module):
    """
        A single Transformer Encoder Block.
    """
    embed_dim: int
    num_heads: int
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Attention
        norm1 = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadAttention(
            num_heads=self.num_heads, 
            qkv_features=self.embed_dim
        )(norm1, norm1) # Self-Attention
        x = x + attn_out
        
        # MLP
        norm2 = nn.LayerNorm()(x)
        mlp_out = nn.Dense(self.embed_dim * 4)(norm2)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.embed_dim)(mlp_out)
        x = x + mlp_out
        
        return x

class AssemblyNetwork(nn.Module):
    """
        The AlphaDev Agent.
        Input: Observation Vector [98]
        Output: 
            - Policy: Tuple of 5 logits (Op, Rd, Rs1, Rs2, Rs3)
            - Value: Scalar
    """
    
    @nn.compact
    def __call__(self, obs, training: bool = False):
        
        # 1. Input Parsing
        # obs shape: (Batch, 98)
        # Registers: First 48 floats (6 universes * 8 regs)
        # History: Next 50 floats (10 steps * 5 components)
        batch_size = obs.shape[0]
        
        regs_flat = obs[:, :48]
        history_flat = obs[:, 48:]
        
        # Reshape history to (Batch, 10, 5)
        # The 5 components are: [Op, Rd, Rs1, Rs2, Rs3]
        history = history_flat.reshape(batch_size, MAX_STEPS, 5)
        
        # 2. History Embedding (The Sequence)
        # We need to handle the -1 padding. 
        # Idea: Shift everything +1 so (-1 becomes 0).
        # 0 is now Padding/None. 
        # Ops: 1...5, Regs: 1...8
        history_idx = history.astype(jnp.int32) + 1
        
        # Embed each component separately
        # Opcode Embedding (Size 6: 0=Pad, 1..5=Ops)
        emb_op = nn.Embed(num_embeddings=6, features=32)(history_idx[:, :, 0])
        
        # Register Embeddings (Size 9: 0=Pad, 1..8=Regs)
        # We share embeddings for Rd, Rs1, Rs2, Rs3 to learn "Registerness"
        reg_embedder = nn.Embed(num_embeddings=9, features=32)
        emb_rd  = reg_embedder(history_idx[:, :, 1])
        emb_rs1 = reg_embedder(history_idx[:, :, 2])
        emb_rs2 = reg_embedder(history_idx[:, :, 3])
        emb_rs3 = reg_embedder(history_idx[:, :, 4])
        
        # Concatenate component embeddings per step -> (Batch, 10, 160)
        step_embeds = jnp.concatenate([emb_op, emb_rd, emb_rs1, emb_rs2, emb_rs3], axis=-1)
        
        # Project to main model dimension
        x_seq = nn.Dense(EMBED_DIM)(step_embeds)
        
        # Add Positional Encodings (Learned)
        pos_embedding = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, MAX_STEPS, EMBED_DIM))
        x_seq = x_seq + pos_embedding
        
        # Transformer Layers
        for _ in range(NUM_LAYERS):
            x_seq = TransformerBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)(x_seq, training=training)
            
        # Global Pooling of the sequence (just flatten or take mean)
        # Since we use positional embeddings, flattening preserves order info.
        x_seq_flat = x_seq.reshape(batch_size, -1) 
        
        # 3. Register Embedding (The Context)
        # Simple MLP to digest the 48 register values
        x_regs = nn.Dense(128)(regs_flat)
        x_regs = nn.gelu(x_regs)
        x_regs = nn.Dense(128)(x_regs)
        x_regs = nn.gelu(x_regs)
        
        # 4. Fusion & Heads
        # Combine Sequence Knowledge + Current Machine State
        x_combined = jnp.concatenate([x_seq_flat, x_regs], axis=-1)
        
        # Deep backbone
        latent = nn.Dense(HIDDEN_DIM)(x_combined)
        latent = nn.gelu(latent)
        latent = nn.Dense(HIDDEN_DIM)(latent)
        latent = nn.gelu(latent)
        
        # Policy Heads (Actor)
        # We output logits for each component independently
        logits_op  = nn.Dense(NUM_OPS)(latent)   # 5 options
        logits_rd  = nn.Dense(NUM_REGS)(latent)  # 8 options
        logits_rs1 = nn.Dense(NUM_REGS)(latent)
        logits_rs2 = nn.Dense(NUM_REGS)(latent)
        logits_rs3 = nn.Dense(NUM_REGS)(latent)
        
        # Value Head (Critic)
        value = nn.Dense(1)(latent)
        
        return (logits_op, logits_rd, logits_rs1, logits_rs2, logits_rs3), value