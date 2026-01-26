#pragma once

#include <vector>
#include <array>
#include <cstdint>

// The Allowed Moves (The ISA)
enum OpCode {
    OP_MOV  = 0, // rd = rs1
    OP_ADD  = 1, // rd = rs1 + rs2
    OP_SUB  = 2, // rd = rs1 - rs2
    OP_AND  = 3, // rd = rs1 & rs2
    OP_SLT  = 4, // rd = (rs1 < rs2) ? 1 : 0
    OP_CMOV = 5, // rd = (rs3 != 0) ? rs1 : rs2
    OP_COUNT = 6
};

// A single instruction in the program history
struct Instruction {
    int op;
    int rd;
    int rs1;
    int rs2;
    int rs3;
};


class AssemblyEnv {
private:
    std::vector<int32_t> registers_;
    std::vector<Instruction> history_;

    const int NUM_TEST_CASES = 6;
    const int NUM_REGS = 8;
    const int MAX_STEPS = 10;

    int steps_taken_ = 0;

public:
    AssemblyEnv();

    // clear the history and load a new set of permutations in the register
    void reset();

    // makes the moves, returns true if MAX_STEPS reached
    bool step(int op, int rd, int rs1, int rs2, int rs3);

    // calculates the score
    // if x1 <= x2 <= x3 -> returns the instruction count (correct)
    // -100 if incorrect
    float get_score() const;

    bool is_sorted() const;
};
