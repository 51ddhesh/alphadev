"""
    ## Monte-Carlo Tree Search Node

    - Each Node represents a state in the search tree.
    - Nodes track visit counts, value estimates and prior probabilities.
"""

from typing import Dict, Optional, List
import numpy as np
import math


class MCTSNode:
    """
        ## A Node in the Monte-Carlo Tree Search

        ## Attributes
            - `state`: The environment state at this node.
            - `parent`: Parent node (`None` for root).
            - `action`: Action taken from parent to reach this node.
            - `prior`: Prior probability from the neural network
            - `children`: Dict mapping action_id -> child MCTSNode.
            - `visit_counts`: Number of times this node has been visited.
            - `value_sum`: Sum of values from all visits.
            - `reward`: Immediate reward for reaching this node.
            - `is_terminal`: Whether this is the terminal state.
    """

    def __init__(
            self, 
            state, 
            parent: Optional["MCTSNode"] = None, 
            action: Optional[int] = None, 
            prior: float = 0.0
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = Dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.reward: float = 0.0

        self.is_terminal: bool = False
        self.is_expanded: bool = False


    @property
    def value(self) -> float:
        """
            ## Mean value estimate
        """

        if self.visit_count == 0:
            return 0.0
        
        return self.value_sum / self.visit_count
