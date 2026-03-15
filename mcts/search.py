"""
    ## Monte-Carlo Tree Search search implementation 

    ## The Algorithm

        1. SELECT: Traverse the tree using UCB until an unexpanded node is reached
        2. EXPAND: Use neural network to get prior probabilities
        3. EVALUATE: Use neural network value head for value estimate
        4. BACKPROPOGATE: Update visit counts and value estimates up the tree
"""

import numpy as np
from typing import Optional, Tuple, Callable
import jax
import jax.numpy as jnp

from .node import MCTSNode


class MCTS:
    """
        ## Monte-Carlo Tree Search with Neural Network Guidance
    """

    def __init__(
            self,
            env,
            network_fn: Callable,
            state_encoder: Callable,
            num_simulations: int = 200,
            c_puct: float = 2.5,
            dirichlet_alpha: float = 0.3,
            root_noise_fraction: float = 0.25,
            discount: float = 0.997
    ):
        self.env = env
        self.network_fn = network_fn
        self.state_encoder = state_encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_fraction = root_noise_fraction
        self.discount = discount


    def _evalutate(self, state) -> Tuple[np.ndarray, float]:
        """
            ## Evalutate a state using the neural network

            ## Returns:
                `(policy_priors, value)`: where `policy_prioirs` is a probability_distribution over actions
        """

        # Encode the state
        obs = self.state_encoder(state)

        # Add batch dim
        obs_batch = {k: v[None, ...] for k, v in obs.items()}

        # Convert to JAX arrays
        obs_jax = {k: jnp.array(v) for k, v in obs_batch.items()}

        # Forward pass
        policy_logits, value = self.network_fn(obs_jax)

        # Convert to numpy 
        # Remove the batch dim
        policy_logits = np.array(policy_logits[0]) 
        value = float(value[0])

        # Softmax for priors
        # Ensure we have the right number of actions
        policy_logits = policy_logits[:self.env.num_actions]
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        priors = exp_logits / np.sum(exp_logits)


        return priors, value
    

    def search(self, root_state, add_noise: bool = True) -> MCTSNode:
        """
            ## Run MCTS from the given root state.

            ## Args:
                - `root_state`: The current environment state.
                - `add_noise`: Whether to add Dirichlet Noise at root.
        
            ## Returns
                The root MCTSNode after the search
        """

        # Create the root node
        root = MCTSNode(state=root_state)

        # Evaluate root
        priors, root_value = self._evalutate(root_state)
        root.expand(priors, self.env)

        # Add Dirichlet noise at the root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            ) 

            root.add_noise(noise, self.root_noise_fraction)

        for _ in range(self.num_simulations):
            node = root

            """
                Step 1: SELECT
            """

            while node.is_expanded and not node.is_terminal:
                if not node.children:
                    break

                node = node.select_child(self.c_puct)

            # If node hasn't been materialzed (lazy exapnsion)
            if node.state is None and node.parent is not None:
                result = self.env.step_from_state(
                    node.parent.state, node.action
                )

                node.state = result.state
                node.reward = result.reward

                if result.is_done:
                    node.is_expanded = True
                    node.is_terminal = True

                    value = result.reward
                    node.backpropogate(value, self.discount)
                    continue
            
            """
                Step 2 and 3: EXPAND and EVALUATE
            """

            if not node.is_terminal and node.state is not None:
                priors, value = self._evalutate(node.state)
                node.expand(priors, self.env)

            else:
                value = node.reward

            
            """
                Step 4: BACKPROPOGATE
            """

            node.backpropogate(value, self.discount)

        return root

    

    def get_action_and_stats(
            self, 
            root: MCTSNode,
            temperature: float = 1.0
    ) -> Tuple[int, np.ndarray, float]:
        """
            ## Select an action from the search results


            ## Returns:
                (action, action_probs, root_value)
        """

        action_probs = root.get_action_distribution(temperature)

        if temperature == 0:
            action = int(np.argmax(action_probs))
        
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))

        
        return action, action_probs, root.value