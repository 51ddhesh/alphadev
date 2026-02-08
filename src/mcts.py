"""
    Monte-Carlo Tree Search for AlphaDev

    Uses env.clone() for state management
    Use Dirichlet noise at root for exploration
"""

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable

# Global Constants: Must match C++ environment
NUM_OPS = 5
NUM_REGS = 8

# ─── MCTS Hyperparameters ──────────────────────────────
C_PUCT = 2.5 
TOP_K_ACTIONS = 30
DIRICHLET_ALPHA = 0.3 
DIRICHLET_FRAC = 0.25 


@dataclass(frozen=True) # Make the instance of class immutable
class Action:
    """
        - Description:
            An assembled instruction as an action for the Monte-Carlo Tree Search
    """

    op: int
    rd: int
    rs1: int
    rs2: int
    rs3: int

    def __repr__(self) -> str:
        ops = ["ADD", "SUB", "AND", "SLT", "CMOV"]
        name = ops[self.op] if self.op < len(ops) else f"OP{self.op}"
        return f"{name} x{self.rd}, x{self.rs1}, x{self.rs2}, x{self.rs3}"


class Node:
    """
        - Description:
            A single node of the Monte-Carlo Tree Search
    """

    # Optimize the memory usage and restrict attribute creation for class instances
    __slot__ = [
        'parent', 'action_from_parent', 'children',
        'visit_count', 'value_sum', 'prior',
        'is_terminal', 'terminal_reward', 'env_snapshot',
    ]


    def __init__(
        self,
        parent: Optional['Node'] = None,
        action_from_parent: Optional[Action] = None,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[Action, 'Node'] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.is_terminal: bool = False
        self.terminal_reward: float = 0.0
        self.env_snapshot = None 

    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        
        return self.value_sum / self.visit_count
    

    def ucb_score(self, parent_visits: int) -> float:
        """
            - Description:
                Calculate the best C_PUCT value
        """

        exploration = C_PUCT * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration
    

    
