"""
    - Central Configuration for AlphaDev.
    - All hyperparameters live here.
"""

from dataclasses import dataclass, field
from typing import Tuple
import jax.numpy as jnp



@dataclass(frozen=True)
class InstructionConfig:
    """
        ## Define the assembly ISA to the agent
    """

    # R0, R1, R2, R3
    num_registers: int = 4
    
    # mem[0] to mem[7]
    num_memory_slots: int = 8

    # Not using immediates for sorting
    max_immediate_value: int = 0


@dataclass(frozen=True)
class EnvironmentConfig:
    """
        ## Environment Configuration
    """

    # Number of elements to sort
    sort_size: int = 3

    # Maximum instructions before termination
    max_program_length: int = 20

    num_registers: int = 4
    num_memory_slots: int = 8

    # ─── Reward shaping ────────────────────────────────────────
    correctness_reward: float = 1.0
    latency_penalty_per_instruction: float = 0.1
    incorrect_penalty: float = -1.0

    # ─── Test Sequencing ───────────────────────────────────────
    # For sort_3: we use 3! = 6 total combinations
    # For sort_4: we use 4! = 24 total combinations
    use_exhaustive_test: bool = True


@dataclass(frozen=True)
class NNConfig:
    """
        ## Neural Network configuration
    """

    # ─── State Embedding ───────────────────────────────────────
    instruction_embed_dim: int = 64
    register_embed_dim: int = 32
    memory_embed_dim: int = 32
    flags_embed_dim: int = 16

    # ─── Transformer Torso ─────────────────────────────────────
    num_transformer_layers: int = 4
    num_attention_heads: int = 4
    transformer_dim: int = 128
    mlp_dim: int = 256
    dropout_rate: float = 0.1


    # ─── Heads ─────────────────────────────────────────────────
    policy_head_hidden: int = 128
    value_head_hidden: int = 128

    # ─── Datatype for mixed precision ──────────────────────────
    dtype: jnp.dtype = jnp.float32


@dataclass(frozen=True)
class MCTSConfig:
    """
        ## Monte-Carlo Search Tree configuration
    """

    # Simulations per move
    num_simulations: int = 200

    # Exploration Constant
    c_puct: float = 2.5

    # ─── Alpha ─────────────────────────────────────────────────
    dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.25

    # ─── Temperature ───────────────────────────────────────────
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 50

    # ─── Virtual Loss ──────────────────────────────────────────
    # used for multiple hardware accelerators 
    # to run parallel MCTS 
    virtual_loss: float = 3.0

    # Discount Factor
    discount: float = 0.997


@dataclass(frozen=True)
class TrainingConfig:
    """
        ## Training loop configuration
    """

    # ─── Core ──────────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 500
    lr_decay_steps: int = 50000
    max_training_steps: int = 100000

    # ─── Replay Buffer ─────────────────────────────────────────
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000

    # ─── Self Play ─────────────────────────────────────────────
    num_self_play_games_per_step: int = 4

    # ─── Checkpointing ─────────────────────────────────────────
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 50

    # ─── Loss + Policy ─────────────────────────────────────────
    policy_loss_weights: float = 1.0
    value_loss_weights: float = 1.0

    # ─── Seeds ─────────────────────────────────────────────────
    seed: int = 51


@dataclass(frozen=True)
class AlphaDevConfig:
    """
        ## Top level configuration combining all the configurations
    """

    instruction: InstructionConfig = field(default_factory=InstructionConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NNConfig = field(default_factory=NNConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

