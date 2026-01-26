import sys
import os

# Ensure we can find the src module
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
from src.agent import AssemblyNetwork

def test_network_forward():
    # Print device to confirm GPU usage
    print(f"JAX Device: {jax.devices()[0]}")
    
    # Initialize Model
    model = AssemblyNetwork()
    key = jax.random.PRNGKey(0)
    
    # Fake Input (Batch Size 2, 98 features)
    dummy_input = jnp.zeros((2, 98))
    
    # Initialize Parameters
    params = model.init(key, dummy_input)
    print("Model initialized successfully.")
    
    # Forward Pass
    (l_op, l_rd, l_rs1, l_rs2, l_rs3), value = model.apply(params, dummy_input)
    
    print("\n--- Output Shapes (Batch=2) ---")
    print(f"Opcode Logits: {l_op.shape} (Expected 2, 5)")
    print(f"Rd Logits:     {l_rd.shape} (Expected 2, 8)")
    print(f"Value:         {value.shape} (Expected 2, 1)")
    
    assert l_op.shape == (2, 5)
    assert l_rd.shape == (2, 8)
    assert value.shape == (2, 1)
    
    print("\nSUCCESS: Network architecture is valid.")

if __name__ == "__main__":
    test_network_forward()