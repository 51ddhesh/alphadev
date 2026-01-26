#include "assembly_env.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

AssemblyEnv::AssemblyEnv() {
    // Resize the registers to hold 6 different permutations of 8 registers each
    // flatten it into a single continuous sequence for SIMD optimizations later
    registers_.resize(NUM_TEST_CASES * NUM_REGS);
    reset();
}

void AssemblyEnv::reset() {
    history_.clear();
    steps_taken_ = 0;


    const int inputs[6][3] = {
        {1, 2, 3},
        {1, 3, 2},
        {2, 1, 3},
        {2, 3, 1},
        {3, 1, 2},
        {3, 2, 1}
    };

    for (int i = 0; i < NUM_TEST_CASES; i++) {
        int base_index = i * NUM_REGS;

        // x0 = 0
        registers_[base_index + 0] = 0;

        // load the values in the registers (x1, x2, x3)
        registers_[base_index + 1] = inputs[i][0];
        registers_[base_index + 2] = inputs[i][1];
        registers_[base_index + 3] = inputs[i][2];
    
        // set x4, x5, x6, x7 to zero
        for (int r = 4; r < NUM_REGS; r++) {
            registers_[base_index + r] = 0;
        }
    }
}

bool AssemblyEnv::step(int op, int rd, int rs1, int rs2, int rs3) {
    history_.push_back({op, rd, rs1, rs2, rs3});
    steps_taken_++;

    if (rd == 0) return steps_taken_ >= MAX_STEPS;
    
    for (int i = 0; i < NUM_TEST_CASES; i++) {
        int base_index = i * NUM_REGS;

        int val1 = registers_[base_index + rs1];
        int val2 = registers_[base_index + rs2];
        int val3 = registers_[base_index + rs3];
        
        int result = 0;

        switch (op) {
            case OP_ADD: // ADD
                result = val1 + val2;
                break;
            
            case OP_SUB: // SUB
                result = val1 - val2;
                break;

            case OP_AND: // AND
                result = val1 & val2;
                break;

            case OP_SLT: // SLT (Set Less Than)
                result = (val1 < val2) ? 1 : 0;
                break;

            case OP_CMOV: // CMOV (Conditional Move)
                // if val3 is NOT zero, take val1 else val2
                result = (val3 != 0) ? val1 : val2;
                break;

            default:
                // unknown condition, -> NOP (no operation)
                result = registers_[base_index + rd];
                break;
        }

        // write the result to the permutation's register
        registers_[base_index + rd] = result;
        
        // redundant safety check for x0 = 0
        registers_[base_index + 0] = 0;
    }

    return steps_taken_ >= MAX_STEPS;
}

bool AssemblyEnv::is_sorted() const {
    for (int i = 0; i < NUM_TEST_CASES; i++) {
        int base_index = i * NUM_REGS;
        int x1 = registers_[base_index + 1];
        int x2 = registers_[base_index + 2];
        int x3 = registers_[base_index + 3];
        
        if (!(x1 <= x2 && x2 <= x3)) return false;
    }

    return true;
}

float AssemblyEnv::get_score() const {
    if (is_sorted()) { 
        // Model wins
        return 10.0f - (0.1f * steps_taken_);
    }

    else return -100.0f;
}


std::vector<float> AssemblyEnv::observe() const {
    std::vector<float> obs;
    obs.reserve((NUM_TEST_CASES * NUM_REGS) + (MAX_STEPS * 5));

    for (int val : registers_) {
        obs.push_back(static_cast<float>(val));
    }


    for (int i = 0; i < MAX_STEPS; i++) {
        if (i < history_.size()) {
            const auto& instr = history_[i];
            obs.push_back(static_cast<float>(instr.op));
            obs.push_back(static_cast<float>(instr.rd));
            obs.push_back(static_cast<float>(instr.rs1));
            obs.push_back(static_cast<float>(instr.rs2));
            obs.push_back(static_cast<float>(instr.rs3));
        }

        else {
            // if the program is shorter than MAX_STEPS
            // fill the remaining observation with -1
            // This will tell the model that nothing has happened here yet
            obs.push_back(-1.0f); // op
            obs.push_back(-1.0f); // rd
            obs.push_back(-1.0f); // rs1
            obs.push_back(-1.0f); // rs2
            obs.push_back(-1.0f); // rs3
        }
    }
    
    return obs;
}


