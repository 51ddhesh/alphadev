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
    

def _decode_top_k_actions(
    logits_tuple: Tuple,
    k: int = TOP_K_ACTIONS,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[Action, float]]:
    """
        - Description:
            Get the top K actions from the factored logits

        - Strategy:
            1. Always include the argmax (greedy)
            2. Sample the remaining (k - 1) actions proportional to the product of per-component probabilities
    """

    l_op, l_rd, l_rs1, l_rs2, l_rs3 = logits_tuple

    # Squeeze the batch dim and convert to numpy
    p_op = np.asarray(jax.nn.softmax(l_op[0]))
    p_rd = np.asarray(jax.nn.softmax(l_rd[0]))
    p_rs1 = np.asarray(jax.nn.softmax(l_rs1[0]))
    p_rs2 = np.asarray(jax.nn.softmax(l_rs2[0]))
    p_rs3 = np.asarray(jax.nn.softmax(l_rs3[0]))

    if rng is None:
        rng = np.random.default_rng


    actions: List[Tuple[Action, float]] = []
    seen = set()


    # Greedy action
    greedy = Action(
        int(np.argmax(p_op)),
        int(np.argmax(p_rd)),
        int(np.argmax(p_rs1)),
        int(np.argmax(p_rs2)),
        int(np.argmax(p_rs3)),
    )

    # Find the joint probability
    # This is the probability of selecting a specific joint action during the simulation phase
    # Possiblly, this is the best heuristic
    joint_p = float(
        p_op[greedy.op] * p_rd[greedy.rd] * p_rs1[greedy.rs1] * p_rs2[greedy.rs2] * p_rs3[greedy.rs3]
    )

    actions.append((greedy, joint_p))
    seen.add(greedy)

    # Sample the remaining k - 1 actions

    max_attempts = k * 10
    attempts = 0

    while len(actions) < k and attempts < max_attempts:
        attempts += 1
        op = int(rng.choice(len(p_op), p = p_op))
        rd = int(rng.choice(len(p_rd), p = p_rd))
        rs1 = int(rng.choice(len(p_rs1), p = p_rs1))
        rs2 = int(rng.choice(len(p_rs2), p = p_rs2))
        rs3 = int(rng.choice(len(p_rs3), p = p_rs3))

        act = Action(op, rd, rs1, rs2, rs3)
        if act in seen:
            continue

        jp = float(
            p_op[op] * p_rd[rd] * p_rs1[rs1] * p_rs2[rs2] * p_rs3[rs3] 
        )

        actions.append((act, jp))
        seen.add(act)

    return actions


def _add_dirichlet_noise(node: Node, rng: np.random.Generator) -> None:
    """
        - Description:
            Add noise to the root priors to promote exploration
    """

    if not node.children:
        return
    
    actions = list(node.children.keys)
    noise = rng.dirichlet([DIRICHLET_ALPHA] * len(actions))

    for i, act in enumerate(actions):
        child = node.children[act]
        child.prior = (
            (1.0 - DIRICHLET_FRAC) * child.prior + DIRICHLET_FRAC * noise[i]
        )


def _select(node: Node) -> Node:
    """
        - Description:
            Walk down the tree and use Upper Confidence Bound (UCB) until a terminal is hit 
    """

    while node.children and not node.is_terminal:
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        node = best_child


    return node


def _expand(node: Node, env, inference_fn: Callable, params, rng: np.random.Generator) -> float:
    """
        - Description
            This function expands the leaf nodes
            1. Runs the network to get the policy and value
            2. Create children from the Top-K actions
            3. Return the values estimate
    """

    # Get the observation from the environment snapshot
    obs = np.array(env.observe(), dtype=np.float32)
    obs_batch = jnp.expand_dims(jnp.array(obs), axis = 0)

    logits, value_est = inference_fn(logits, TOP_K_ACTIONS, rng)
    value = float(value_est[0, 0])

    # Get the top-K actions
    candidate_actions = _decode_top_k_actions(logits, TOP_K_ACTIONS, rng)

    # CReate the child nodes
    for act, prior in candidate_actions:
        if act not in node.children:
            node.children[act] = Node(
                parent=node,
                action_from_parent=act,
                prior=prior
            )

    node.env_snapshot = env

    return value

def _backpropogate(node: Node, value: float) -> None:
    """
        - Description:  
            Propogate the value estimate up to the root
    """

    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        node = node.parent


# ─── The Core Monte-Carlo Search ──────────────────────────────
def mcts_search(
    root_env,
    inference_fn: Callable,
    params,
    num_simulations: int = 50,
    add_noise: bool = True,
    seed: int = 0
) -> Node:
    """
        - Description:
            Run the Monte-Carlo Tree Search from the given position or the env state.

        - Input:
            root_env: AssemblyEnv (from C++ bindings).
            inference_fn: JIT Compiled function which takes in (params, obs) and returns (policy, value).
            params: network parameters.
            num_simulations: the number of MCTS simulations to run.
            add_noise: the Dirichlet parameters to add noise (promotes expoloration) at root.
            seed: Random Number Generator seed value.

        - Outputs:
            Node: Root Node with updated visit counts and values  
    """

    rng = np.random.default_rng(seed=seed);
    root = Node()

    # Expand the root
    root_clone = root_env.clone()

    if root_clone.is_sorted():
        root.is_terminal = True
        root.terminal_reward = root_clone.reward()
        root.visit_count = 1
        root.value_sum = root.terminal_reward
        return root
    
    if root_clone.is_done():
        root.is_terminal = True
        root.terminal_reward = root_clone.reward()
        root.visit_count = 1
        root.value_sum = root.terminal_reward
        return root

    _expand(root, root_env, inference_fn, params, rng)

    if add_noise:
        _add_dirichlet_noise(root, rng)

    
    """
        Start running the simulations
    """

    for _ in range(num_simulations):
        # Selection
        leaf = _select(root)
        if leaf.is_terminal:        
            _backpropogate(leaf, leaf.terminal_reward)

        # Get env for the node
        ancestors = []
        cursor = leaf
        while cursor.env_snapshot is None and cursor.parent is not None:
            ancestors.append(cursor.action_from_parent)
            cursor = cursor.parent

        if cursor.env_snapshot is None:
            _backpropogate(leaf, 0.0)

        leaf_env = cursor.env_snapshot.clone()

        for act in reversed(ancestors):
            try:
                leaf_env.step(act.op, act.rd, act.rs1, act.rs2, act.rs3)
            except RuntimeError:
                leaf.is_terminal = True
                leaf.terminal_reward = leaf_env.reward()
                _backpropogate(leaf, leaf.terminal_reward)
                break
        
        else:
            # Check terminal conditions
            if leaf_env.is_sorted():
                leaf.is_terminal = True
                leaf.terminal_reward = leaf_env.reward()
                _backpropogate(leaf, leaf.terminal_reward)
                continue

            if leaf_env.is_done():
                leaf.is_terminal = True
                leaf.terminal_reward = leaf_env.reward()
                
        # expansion
        value = _expand(leaf, leaf_env, inference_fn, rng)
        _backpropogate(leaf, value)

    return root


def get_mcts_policy(root: Node) -> Tuple:
    """
        - Description:
            Extract improved policy from MCTS visit counts.
            Returns 5 probability distributions (marginalized)

            (p_op, p_rd, p_rs1, p_rs2, p_rs3)
    """

    counts_op = np.zeros(NUM_OPS, dtype=np.float32)
    counts_rd = np.zeros(NUM_REGS, dtype=np.float32)
    counts_rs1 = np.zeros(NUM_REGS, dtype=np.float32)
    counts_rs2 = np.zeros(NUM_REGS, dtype=np.float32)
    counts_rs3 = np.zeros(NUM_REGS, dtype=np.float32)

    total = 0

    for action, child in root.children.items():
        v = child.visit_count
        if v == 0:
            continue
    
        total += v
        counts_op[action.op] += v
        counts_rd[action.rd] += v
        counts_rs1[action.rs1] += v
        counts_rs2[action.rs2] += v
        counts_rs3[action.rs3] += v

    if total == 0:
        return (
            np.ones(NUM_OPS, dtype=np.float32) / NUM_OPS,
            np.ones(NUM_REGS, dtype=np.float32) / NUM_REGS,
            np.ones(NUM_REGS, dtype=np.float32) / NUM_REGS,
            np.ones(NUM_REGS, dtype=np.float32) / NUM_REGS,
            np.ones(NUM_REGS, dtype=np.float32) / NUM_REGS,
        )
    
    return (
        counts_op  / total,
        counts_rd  / total,
        counts_rs1 / total,
        counts_rs2 / total,
        counts_rs3 / total,
    )

def select_action(root: Node, temperature: float = 1.0) -> Action:
    """
        - Description:
            Select an action from the root based on visit counts

            temperature = 0: always pick most visited (exploited)
            temperature < 0: sample proportional to viist_count ^ (1 / temp)
    """
    if not root.children:
        raise ValueError("Cannot select action as the root has no children")
    
    actions = list(root.children.keys())
    visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

    if temperature < 1e-6:
        idx = int(np.argmax(visits))

    else:
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        idx = int(np.random.choice(len(actions), p = probs))

    return actions[idx]