"""
    ## Monte-Carlo Tree Search Node

    - Each Node represents a state in the search tree.
    - Nodes track visit counts, value estimates and prior probabilities.
"""

from typing import Dict, Optional
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


    def expand(
            self,
            actions_priors: np.ndarray,
            env
    ):
        """
            ## Expand *this node by creating children for all valid actions

            ## Args
                - `action_priors`: (num_actions, ) prior probabilities from neural network.
                - `env`: The environment (for stepping)
        """

        if self.is_expanded:
            return
        
        self.is_expanded = True

        if self.state.done:
            self.is_terminal = True
            return
        
        for action_id in range(len(actions_priors)):
            prior = float(actions_priors[action_id])

            # Only create children for non-zero priors
            if prior > 0:

                # Lazy exapnsion
                child = MCTSNode(
                    state=None,
                    parent=self,
                    action=action_id,
                    prior=prior,
                )

                self.children[action_id] = child

    

    def select_child(self, c_puct: float) -> 'MCTSNode':
        """
            ## Select the child with highest UCB (upper confidence bound) score.

            UCB = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
            
            where:
                - Q(s, a): mean value of the child
                - P(s, a): prior probability of child
                - N(s): visit count of parent
                - N(s, a): visit count of child
        """

        best_score = -float('inf')
        best_child = None

        sqrt_parent = math.sqrt(self.visit_count)

        for child in self.children.values():
            q_value = child.value

            exploration = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)

            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
    

    def backpropogate(self, value: float, discount: float = 0.997):
        """
            ## Backpropogate a value up the tree

            The value is discounted as we go up (further from leaf -> less certain).
        """

        node = self
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = node.reward + discount * current_value
            node = node.parent


    def add_noise(self, noise: np.ndarray, fraction: float):
        """
            ## Add Dirichlet noise to prior probabilities at the root.

            This encourages exploration while the game is played.
        """

        if not self.children:
            return
        
        actions = list(self.children.keys())

        for i, action_id in enumerate(actions):
            child = self.children[action_id]
            child.prior = (
                (1 - fraction) * child.prior
                + fraction * noise[i]
            )


    def get_action_distribution(self, temperature: float = 1.0) -> np.ndarray:
        """
            ## Get the action probability distribution based on visit counts

            ## Args:
                - `temperature`: Controls exploration.
                    `temperature == 1.0`: proportional to visits
                    `temperature -> 0`: greedy exploration

            ## Returns: 
                `(num_actions, )` probability distribution
        """

        if not self.children:
            return np.array([])
        
        # Determine the array size by finding the max action id
        max_actions = max(self.children.keys()) + 1
        visits = np.zeros(max_actions)

        for action_id, child in self.children.items():
            visits[action_id] = child.visit_count


        # Greedy
        if temperature == 0:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
            return probs
        
        # Apply temperature
        if np.sum(visits) == 0:
            # Uniform if no visits
            probs = np.ones_like(visits) / len(visits)

        else:
            visits_temps = visits ** (1.0 / temperature)
            probs = visits_temps / np.sum(visits_temps)

        return probs