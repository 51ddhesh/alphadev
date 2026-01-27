import sys
import os

# 1. Add current directory (project root) so we can import 'src.agent'
sys.path.append(os.getcwd())
# 2. Add build directory so we can import 'alphadev' (C++ module)
sys.path.append(os.path.join(os.getcwd(), 'build'))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import pickle
from functools import partial

# Import our modules
from src.agent import AssemblyNetwork, NUM_OPS, NUM_REGS
from src.mcts import Node, mcts_search, Action
import alphadev

# # --- Hyperparameters ---
# LEARNING_RATE = 1e-4
# BATCH_SIZE = 16 
# NUM_ITERATIONS = 200    # Increased for long run
# GAMES_PER_ITER = 10     
# SIMS_PER_MOVE = 50      
# BUFFER_SIZE = 2000      

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 
NUM_ITERATIONS = 50     # Reduced from 200
GAMES_PER_ITER = 5      # Reduced from 10 (Less waiting)
SIMS_PER_MOVE = 20      # Reduced from 50 (Faster thinking, still smart enough for Sort3)
BUFFER_SIZE = 2000

class TrainState(train_state.TrainState):
    """Custom State to hold batch statistics if needed."""
    pass

def create_train_state(rng, learning_rate):
    """Initialize the model and optimizer."""
    model = AssemblyNetwork()
    dummy_obs = jnp.zeros((1, 98))
    params = model.init(rng, dummy_obs)
    
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_obs, batch_target_policies, batch_target_values):
    """Performs one step of Gradient Descent."""
    
    def loss_fn(params):
        # 1. Forward Pass
        (l_op, l_rd, l_rs1, l_rs2, l_rs3), v_pred = state.apply_fn(params, batch_obs)
        
        # 2. Value Loss (MSE)
        value_loss = jnp.mean((v_pred - batch_target_values) ** 2)
        
        # 3. Policy Loss (Cross Entropy)
        def ce_loss(logits, targets):
            log_probs = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))

        loss_op = ce_loss(l_op, batch_target_policies[0])
        loss_rd = ce_loss(l_rd, batch_target_policies[1])
        loss_src = (ce_loss(l_rs1, batch_target_policies[2]) + 
                    ce_loss(l_rs2, batch_target_policies[3]) + 
                    ce_loss(l_rs3, batch_target_policies[4]))
        
        policy_loss = loss_op + loss_rd + loss_src
        total_loss = value_loss + policy_loss
        return total_loss, (value_loss, policy_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (v_loss, p_loss)), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, v_loss, p_loss

def marginalize_mcts_policy(root: Node):
    """Converts MCTS children visit counts into 5 separate probability distributions."""
    counts_op = np.zeros(NUM_OPS)
    counts_rd = np.zeros(NUM_REGS)
    counts_rs1 = np.zeros(NUM_REGS)
    counts_rs2 = np.zeros(NUM_REGS)
    counts_rs3 = np.zeros(NUM_REGS)
    
    total_visits = 0
    for action, child in root.children.items():
        v = child.visit_count
        if v == 0: continue
        total_visits += v
        counts_op[action.op] += v
        counts_rd[action.rd] += v
        counts_rs1[action.rs1] += v
        counts_rs2[action.rs2] += v
        counts_rs3[action.rs3] += v
        
    if total_visits == 0:
        return (np.ones(NUM_OPS)/NUM_OPS, np.ones(NUM_REGS)/NUM_REGS, 
                np.ones(NUM_REGS)/NUM_REGS, np.ones(NUM_REGS)/NUM_REGS, 
                np.ones(NUM_REGS)/NUM_REGS)
                
    return (counts_op/total_visits, counts_rd/total_visits, counts_rs1/total_visits, 
            counts_rs2/total_visits, counts_rs3/total_visits)

# def self_play_game(state, sims_per_move=SIMS_PER_MOVE):
#     """Plays one full game. Returns data and reward."""
#     env = alphadev.AssemblyEnv()
#     env.reset()
    
#     trajectory = []
#     action_history = [] 
    
#     step = 0
#     while True:
#         root = Node()
#         # Run MCTS
#         root = mcts_search(root, state.apply_fn, state.params, num_simulations=sims_per_move)
        
#         # Store Data
#         obs = np.array(env.observe())
#         mcts_policy = marginalize_mcts_policy(root)
#         trajectory.append({'obs': obs, 'policy': mcts_policy})
        
#         # Select Action
#         if not root.children:
#             # Should not happen with sims=50, but safety break
#             break
            
#         best_action = max(root.children.items(), key=lambda i: i[1].visit_count)[0]
#         action_history.append(best_action)
        
#         # Execute
#         is_done = env.step(best_action.op, best_action.rd, best_action.rs1, best_action.rs2, best_action.rs3)
        
#         # Check Win/Loss
#         if env.is_sorted():
#             reward = 10.0 - (0.1 * len(trajectory))
            
#             # --- VICTORY PRINT ---
#             print("\n" + "="*60)
#             print(f"🌟 SOLUTION DISCOVERED! (Length: {len(action_history)}, Score: {reward:.2f})")
#             print("="*60)
#             for i, act in enumerate(action_history):
#                 print(f"{i+1:02d}: {act}")
#             print("="*60 + "\n")
#             # ---------------------
#             break
            
#         elif is_done: 
#             reward = -10.0
#             break
        
#         step += 1
    
#     # Backpropagate reward
#     data_points = []
#     for step_data in trajectory:
#         data_points.append((step_data['obs'], step_data['policy'], reward))
        
#     return data_points, reward


def self_play_game(state, sims_per_move=SIMS_PER_MOVE):
    """Plays one full game. Returns data and reward."""
    env = alphadev.AssemblyEnv()
    env.reset()
    
    trajectory = []
    action_history = [] 
    
    # --- OPTIMIZATION START ---
    # We compile the network forward pass into an XLA kernel.
    # This removes Python overhead from the inner MCTS loop.
    @jax.jit
    def fast_inference(params, obs):
        return state.apply_fn(params, obs)
    # --- OPTIMIZATION END ---

    step = 0
    while True:
        root = Node()
        
        # Pass the JIT-compiled 'fast_inference' instead of 'state.apply_fn'
        root = mcts_search(root, fast_inference, state.params, num_simulations=sims_per_move)
        
        # Store Data
        obs = np.array(env.observe())
        mcts_policy = marginalize_mcts_policy(root)
        trajectory.append({'obs': obs, 'policy': mcts_policy})
        
        # Select Action
        if not root.children:
            break
            
        best_action = max(root.children.items(), key=lambda i: i[1].visit_count)[0]
        action_history.append(best_action)
        
        # Execute
        is_done = env.step(best_action.op, best_action.rd, best_action.rs1, best_action.rs2, best_action.rs3)
        
        # Check Win/Loss
        if env.is_sorted():
            reward = 10.0 - (0.1 * len(trajectory))
            
            print("\n" + "="*60)
            print(f"🌟 SOLUTION DISCOVERED! (Length: {len(action_history)}, Score: {reward:.2f})")
            print("="*60)
            for i, act in enumerate(action_history):
                print(f"{i+1:02d}: {act}")
            print("="*60 + "\n")
            break
            
        elif is_done: 
            reward = -10.0
            break
        
        step += 1
    
    data_points = []
    for step_data in trajectory:
        data_points.append((step_data['obs'], step_data['policy'], reward))
        
    return data_points, reward

def run_training():
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    print("Initializing Agent on GPU...")
    state = create_train_state(init_rng, LEARNING_RATE)
    
    replay_buffer = []
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")
        
        # A. Self-Play
        print(f"Generating {GAMES_PER_ITER} games...")
        iter_rewards = []
        for g in range(GAMES_PER_ITER):
            new_data, reward = self_play_game(state)
            replay_buffer.extend(new_data)
            iter_rewards.append(reward)
            if (g+1) % 5 == 0:
                print(f"  Game {g+1}: Reward {reward:.1f}")
            
        avg_reward = sum(iter_rewards) / len(iter_rewards)
        print(f"  Avg Reward: {avg_reward:.2f}")
        
        # Trim Buffer
        if len(replay_buffer) > BUFFER_SIZE:
            replay_buffer = replay_buffer[-BUFFER_SIZE:]
            
        # B. Training
        if len(replay_buffer) < BATCH_SIZE:
            print("  Buffer filling...")
            continue
            
        print("Training Network...")
        indices = np.arange(len(replay_buffer))
        np.random.shuffle(indices)
        
        num_batches = len(replay_buffer) // BATCH_SIZE
        total_loss = 0
        
        for b in range(num_batches):
            batch_idx = indices[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            
            b_obs = []
            b_p_op, b_p_rd, b_p_rs1, b_p_rs2, b_p_rs3 = [], [], [], [], []
            b_val = []
            
            for i in batch_idx:
                obs, (p_op, p_rd, p_rs1, p_rs2, p_rs3), val = replay_buffer[i]
                b_obs.append(obs)
                b_p_op.append(p_op); b_p_rd.append(p_rd)
                b_p_rs1.append(p_rs1); b_p_rs2.append(p_rs2); b_p_rs3.append(p_rs3)
                b_val.append([val])
                
            b_obs = jnp.array(b_obs)
            b_vals = jnp.array(b_val)
            b_pol = [jnp.array(x) for x in [b_p_op, b_p_rd, b_p_rs1, b_p_rs2, b_p_rs3]]
            
            state, loss, v_loss, p_loss = train_step(state, b_obs, b_pol, b_vals)
            total_loss += loss
            
        print(f"  Loss: {total_loss/num_batches:.4f}")
        
        if (iteration + 1) % 10 == 0:
             with open(f"checkpoint_{iteration}.pkl", "wb") as f:
                 pickle.dump(jax.device_get(state.params), f)
                 print("  Checkpoint saved.")

if __name__ == "__main__":
    run_training()