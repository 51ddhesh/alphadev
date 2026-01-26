import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'build'))

sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp

from src.agent import AssemblyNetwork
from src.mcts import Node, mcts_search, Action

def test_mcts_step():
    print("Initializing Model...")
    model = AssemblyNetwork()
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 98))
    params = model.init(key, dummy_input)
    
    print("Initializing MCTS Root...")
    root = Node()
    
    print("Running MCTS (20 simulations)...")
    root = mcts_search(root, model, params, num_simulations=20)
    
    print(f"\nRoot Visits: {root.visit_count}")
    print(f"Children Expanded: {len(root.children)}")
    
    if len(root.children) == 0:
        print("FAILURE: No children expanded.")
        return

    # Print best action
    best_action = max(root.children.items(), key=lambda item: item[1].visit_count)
    print(f"Most visited action: {best_action[0]}")
    print(f"Visits: {best_action[1].visit_count}, Value: {best_action[1].value:.4f}")
    
    print("\nSUCCESS: MCTS loop is functional.")

if __name__ == "__main__":
    test_mcts_step()