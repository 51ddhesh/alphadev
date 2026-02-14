# FILE: tests/test_env.py
"""Tests for the C++ assembly environment."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'build'))

import alphadev_env


def test_initial_state():
    """Environment starts with correct permutations."""
    env = alphadev_env.AssemblyEnv()
    obs = env.observe()

    assert len(obs) == alphadev_env.OBS_SIZE, f"Obs size: {len(obs)} != {alphadev_env.OBS_SIZE}"
    assert not env.is_sorted()
    assert env.num_steps() == 0
    assert not env.is_done()

    # First test case: {1, 2, 3} → already sorted
    # Check normalized register values: x1=1/4, x2=2/4, x3=3/4
    assert obs[1] == 1.0 / 4.0, f"x1 perm0: {obs[1]}"
    assert obs[2] == 2.0 / 4.0, f"x2 perm0: {obs[2]}"
    assert obs[3] == 3.0 / 4.0, f"x3 perm0: {obs[3]}"

    print(" test_initial_state passed")


def test_add_instruction():
    """ADD executes correctly across all test cases."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    # ADD x4, x1, x2 → x4 = x1 + x2
    env.step(0, 4, 1, 2, 0)

    obs = env.observe()

    # Perm 0: x1=1, x2=2 → x4 = 3, normalized = 3/4
    assert obs[4] == 3.0 / 4.0, f"Perm 0 x4: {obs[4]}"

    # Perm 1: x1=1, x2=3 → x4 = 4, normalized = 4/4
    assert obs[8 + 4] == 4.0 / 4.0, f"Perm 1 x4: {obs[8 + 4]}"

    assert env.num_steps() == 1
    print(" test_add_instruction passed")


def test_slt_instruction():
    """SLT (Set Less Than) works correctly."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    # Perm 0: x1=1, x2=2 → SLT x4, x1, x2 → x4 = 1 (true)
    # Perm 2: x1=2, x2=1 → SLT x4, x1, x2 → x4 = 0 (false)
    env.step(3, 4, 1, 2, 0)
    obs = env.observe()

    perm0_x4 = obs[4]   # 1/4 = 0.25
    perm2_x4 = obs[20]  # 0/4 = 0.0

    assert perm0_x4 == 0.25, f"Perm 0 SLT: {perm0_x4}"
    assert perm2_x4 == 0.0,  f"Perm 2 SLT: {perm2_x4}"

    print(" test_slt_instruction passed")


def test_cmov_instruction():
    """CMOV (Conditional Move) works correctly."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    # First set x4 = 1 via SLT (for perm 0 where x1=1 < x2=2)
    env.step(3, 4, 1, 2, 0)  # SLT x4, x1, x2

    # CMOV x5, x1, x2, x4 → if x4 != 0: x5 = x1 else x5 = x2
    # Perm 0: x4=1 (nonzero) → x5 = x1 = 1
    env.step(4, 5, 1, 2, 4)

    obs = env.observe()
    perm0_x5 = obs[5] * 4.0  # Denormalize
    assert perm0_x5 == 1.0, f"CMOV result: {perm0_x5}"

    print(" test_cmov_instruction passed")


def test_x0_hardwired():
    """Writing to x0 should be silently ignored."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    # ADD x0, x1, x2 → should NOT change x0
    env.step(0, 0, 1, 2, 0)

    obs = env.observe()
    for t in range(6):
        assert obs[t * 8] == 0.0, f"x0 in case {t} changed!"

    print(" test_x0_hardwired passed")


def test_clone():
    """Clone produces an independent copy."""
    env = alphadev_env.AssemblyEnv()
    env.reset()
    env.step(0, 4, 1, 2, 0)  # Modify original

    clone = env.clone()
    assert clone.num_steps() == 1

    # Modify clone, original should be unaffected
    clone.step(0, 5, 1, 3, 0)
    assert clone.num_steps() == 2
    assert env.num_steps() == 1

    print(" test_clone passed")


def test_sorting_program():
    """The known 9-instruction sorting program works."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    program = [
        (3, 4, 2, 1, 0),
        (4, 5, 2, 1, 4),
        (4, 6, 1, 2, 4),
        (3, 4, 3, 6, 0),
        (4, 2, 3, 6, 4),
        (4, 3, 6, 3, 4),
        (3, 4, 2, 5, 0),
        (4, 1, 2, 5, 4),
        (4, 2, 5, 2, 4),
    ]

    for op, rd, rs1, rs2, rs3 in program:
        env.step(op, rd, rs1, rs2, rs3)

    assert env.is_sorted(), f"Program did not sort!\n{env.dump_regs()}"
    assert env.correctness() == 1.0
    assert env.reward() > 1.0  # Solved bonus

    print(f" test_sorting_program passed (reward={env.reward():.3f})")


def test_observation_size():
    """Observation vector has correct size at all steps."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    for i in range(10):
        obs = env.observe()
        assert len(obs) == alphadev_env.OBS_SIZE, f"Step {i}: obs size {len(obs)}"
        env.step(0, 4, 1, 2, 0)

    print(" test_observation_size passed")


def test_max_steps():
    """Environment correctly reports done after MAX_STEPS."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    for i in range(alphadev_env.MAX_STEPS - 1):
        done = env.step(0, 4, 1, 2, 0)
        assert not done, f"Reported done at step {i + 1}"

    done = env.step(0, 4, 1, 2, 0)
    assert done, "Should be done at MAX_STEPS"
    assert env.is_done()

    print(" test_max_steps passed")


def test_range_validation():
    """Out-of-range arguments raise exceptions."""
    env = alphadev_env.AssemblyEnv()
    env.reset()

    try:
        env.step(5, 0, 0, 0, 0)  # op=5 is out of range
        assert False, "Should have raised"
    except RuntimeError:
        pass

    try:
        env.step(0, 8, 0, 0, 0)  # rd=8 is out of range
        assert False, "Should have raised"
    except RuntimeError:
        pass

    print(" test_range_validation passed")


def test_reward_range():
    """Rewards are in expected ranges."""
    # Unsolved
    env = alphadev_env.AssemblyEnv()
    env.reset()
    r = env.reward()
    assert -0.5 <= r <= 0.0, f"Unsolved reward {r} out of range"

    # Solved
    env2 = alphadev_env.AssemblyEnv()
    env2.reset()
    for op, rd, rs1, rs2, rs3 in [
        (3, 4, 2, 1, 0), (4, 5, 2, 1, 4), (4, 6, 1, 2, 4),
        (3, 4, 3, 6, 0), (4, 2, 3, 6, 4), (4, 3, 6, 3, 4),
        (3, 4, 2, 5, 0), (4, 1, 2, 5, 4), (4, 2, 5, 2, 4),
    ]:
        env2.step(op, rd, rs1, rs2, rs3)
    r2 = env2.reward()
    assert 1.0 <= r2 <= 2.0, f"Solved reward {r2} out of range"

    print(f" test_reward_range passed (unsolved={r:.3f}, solved={r2:.3f})")


if __name__ == '__main__':
    test_initial_state()
    test_add_instruction()
    test_slt_instruction()
    test_cmov_instruction()
    test_x0_hardwired()
    test_clone()
    test_sorting_program()
    test_observation_size()
    test_max_steps()
    test_range_validation()
    test_reward_range()
    print("\n All environment tests run!")