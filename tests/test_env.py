import sys
import os

# Point to the build directory so we can find the .so file
# (Adjust 'build' if you named it something else)
sys.path.append(os.path.join(os.getcwd(), 'build'))

import alphadev

def test_arithmetic():
    env = alphadev.AssemblyEnv()
    env.reset()
    
    # OpCode 0: ADD
    # Instruction: ADD x4, x1, x2
    # Logic: x4 = x1 + x2
    env.step(0, 4, 1, 2, 0)
    
    obs = env.observe()
    
    # Let's verify Universe 0 (Input: 1, 2, 3)
    # x1 (idx 1) = 1
    # x2 (idx 2) = 2
    # x4 (idx 4) should be 1 + 2 = 3
    val_univ0 = obs[4]
    
    # Let's verify Universe 1 (Input: 1, 3, 2)
    # x1 (idx 9) = 1
    # x2 (idx 10) = 3
    # x4 (idx 12) should be 1 + 3 = 4
    # Note: Universe 1 starts at index 8. x4 is 8+4=12.
    val_univ1 = obs[12]
    
    print(f"Universe 0 Result: {val_univ0} (Expected 3.0)")
    print(f"Universe 1 Result: {val_univ1} (Expected 4.0)")

    if val_univ0 == 3.0 and val_univ1 == 4.0:
        print(" - SUCCESS: ADD Instruction verified across parallel universes.")
    else:
        print(" - FAILURE: Arithmetic failed.")

def test_cmov():
    env = alphadev.AssemblyEnv()
    env.reset()
    
    # We want to test CMOV: rd = (rs3 != 0) ? rs1 : rs2
    # OpCode 4: CMOV (Assuming 0-indexed enum: ADD, SUB, AND, SLT, CMOV)
    # Let's verify the Enum value first
    OP_CMOV = 4 
    
    # Setup: Set x4 = 10, x5 = 20
    # Since we don't have immediate loads, we rely on existing values or math.
    # Let's just use x1 and x2 as sources.
    
    # Instruction: CMOV x6, x1, x2, x1
    # Condition: x1 (rs3). Since x1 is never 0 in our inputs, 
    # the result should ALWAYS be x1.
    env.step(OP_CMOV, 6, 1, 2, 1)
    
    obs = env.observe()
    # Universe 0: x1=1, x2=2. Condition=1 (True). Result x6 should be 1.
    val = obs[6]
    
    print(f"CMOV Result (Cond True): {val} (Expected 1.0)")
    
    if val == 1.0:
        print(" - SUCESS: CMOV verified.")
    else:
        print(" - FAILURE: CMOV failed.")

if __name__ == "__main__":
    test_arithmetic()
    test_cmov()