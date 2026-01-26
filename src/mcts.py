import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import alphadev 

TOP_K_ACTIONS = 10 # Consider the top 10 moves given by the network
C_PUCT = 2.0  # Exploration constant

@dataclass(frozen=True)
class Action:
    op: int
    rd: int
    rs1: int
    rs2: int
    rs3: int
    
    def __repr__(self):
        ops = ["ADD", "SUB", "AND", "SLT", "CMOV"]
        name = ops[self.op] if self.op < len(ops) else "UNK"
        return f"{name} x{self.rd}, x{self.rs1}, x{self.rs2}, x{self.rs3}"
    

class Node:
    def __init__(self, parent=None, action_from_parent=None, prior=0.0):
        self.parent = parent
        self.action_from_parent = action_from_parent # The action that led to this node
        
        self.children: Dict[Action, Node] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        
        self.is_terminal = False
        self.reward = 0.0

    @property
    def value(self):
        return self.value_sum / (self.visit_count + 1e-6)
    
    # The AlphaZero UCB Formula
    def ucb_score(self, total_parent_visits):
        u = C_PUCT * self.prior * np.sqrt(total_parent_visits) / (1 + self.visit_count)
        return self.value + u
    


def decode_top_k_actions(logits_tuple, k=TOP_K_ACTIONS) -> List[Tuple[Action, float]]:
    l_op, l_rd, l_rs1, l_rs2, l_rs3 = logits_tuple

    # Convert to probabilities on CPU
    p_op = np.array(jax.nn.softmax(l_op)[0])   # shape (5,)
    p_rd = np.array(jax.nn.softmax(l_rd)[0])   # shape (8,)
    p_rs1 = np.array(jax.nn.softmax(l_rs1)[0])
    p_rs2 = np.array(jax.nn.softmax(l_rs2)[0])
    p_rs3 = np.array(jax.nn.softmax(l_rs3)[0])

    actions = []
    seen = set()


    # 1. Always include the Argmax (Best Guess)
    best_act = Action(
        np.argmax(p_op), np.argmax(p_rd), np.argmax(p_rs1), 
        np.argmax(p_rs2), np.argmax(p_rs3)
    )


    joint_p = p_op[best_act.op] * p_rd[best_act.rd] * p_rs1[best_act.rs1] * \
              p_rs2[best_act.rs2] * p_rs3[best_act.rs3]
    
    actions.append((best_act, joint_p))
    seen.add(best_act)
    
    # 2. Sample others
    attempts = 0
    while len(actions) < k and attempts < k * 5:
        attempts += 1
        op = np.random.choice(len(p_op), p=p_op)
        rd = np.random.choice(len(p_rd), p=p_rd)
        rs1 = np.random.choice(len(p_rs1), p=p_rs1)
        rs2 = np.random.choice(len(p_rs2), p=p_rs2)
        rs3 = np.random.choice(len(p_rs3), p=p_rs3)

        act = Action(op, rd, rs1, rs2, rs3)
        if act in seen:
            continue

        joint_p = p_op[op] * p_rd[rd] * p_rs1[rs1] * p_rs2[rs2] * p_rs3[rs3]
        actions.append((act, joint_p))
        seen.add(act)

    return actions


def mcts_search(root: Node, model, params, num_simulations=20):
    for _ in range(num_simulations):
        node = root
        
        # Replay Environment to current state
        # (Optimized: In a real system we would keep env copies, 
        # but for depth=10 replay is instant)
        env = alphadev.AssemblyEnv()
        env.reset()
        
        # Collect path from root
        path = []
        curr = node
        while curr.parent is not None:
            path.append(curr.action_from_parent)
            curr = curr.parent
        
        # Apply path (Reverse because we walked up)
        for act in reversed(path):
            env.step(act.op, act.rd, act.rs1, act.rs2, act.rs3)
            
        # 2. Selection
        # Walk down tree until we find a node that is NOT expanded or Terminal
        while len(node.children) > 0 and not node.is_terminal:
            # Pick child with highest UCB
            best_score = -float('inf')
            best_child = None
            
            for act, child in node.children.items():
                score = child.ucb_score(node.visit_count)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
            # Step the environment
            act = node.action_from_parent
            env.step(act.op, act.rd, act.rs1, act.rs2, act.rs3)

        # 3. Expansion & Evaluation
        if not node.is_terminal:
            # Check if this state is actually terminal in the game (Max steps or Sorted)
            # We already stepped the env, so we check status
            # Note: The 'env.step' returns done, but we need to check strictly here.
            
            if env.is_sorted():
                node.is_terminal = True
                node.reward = 10.0 - (0.1 * len(path)) # Win!
                value = node.reward
            elif len(path) >= 10: # MAX STEPS
                node.is_terminal = True
                node.reward = -10.0 # Lose
                value = node.reward
            else:
                # Not terminal, ask the Neural Network
                obs = np.array(env.observe())
                # Add batch dim (1, 98)
                obs_batch = jnp.expand_dims(obs, axis=0)
                
                logits, value_est = model.apply(params, obs_batch)
                value = float(value_est[0][0])
                
                # Expand children
                possible_actions = decode_top_k_actions(logits)
                for act, prior in possible_actions:
                    node.children[act] = Node(parent=node, action_from_parent=act, prior=prior)
        else:
            value = node.reward

        # backprop
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            
    return root
