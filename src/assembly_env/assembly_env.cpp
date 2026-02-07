#include "assembly_env.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <format>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace alphadev {

    // Instruction helpers
    
    static constexpr std::array<const char*, NUM_OPS> OP_NAMES = {
        "ADD", "SUB", "AND", "SLT", "CMOV"
    };

    std::string Instruction::to_string() const {
        if (op < 0 || op >= NUM_OPS) {
            return std::format("UNKNOWN(op={})", static_cast<int>(op));
        }

        return std::format("{} x{}, x{}, x{}, x{}", 
            OP_NAMES[static_cast<std::size_t>(op)],
            static_cast<int>(rd),
            static_cast<int>(rs1),
            static_cast<int>(rs2),
            static_cast<int>(rs3)
        );
    }

    // Constructor/Reset
    
    void AssemblyEnv::load_permutations() {
        static constexpr std::array<std::array<int, 3>, NUM_TEST_CASES> PERMS = {{
            {1, 2, 3},
            {1, 3, 2},
            {2, 1, 3},
            {2, 3, 1},
            {3, 1, 2},
            {3, 2, 1}
        }};

        registers_.fill(0);
        
        for (int t = 0; t < NUM_TEST_CASES; t++) {
            reg(t, 1) = PERMS[static_cast<std::size_t>(t)][0];
            reg(t, 2) = PERMS[static_cast<std::size_t>(t)][1];
            reg(t, 3) = PERMS[static_cast<std::size_t>(t)][2];
        
            // x0, x4, x5, x6, x7 is already set to zero by registers_.fill(0) 
        }
    }

    void AssemblyEnv::reset() {
        history_.clear();
        steps_taken_ = 0;
        load_permutations();
    }

    AssemblyEnv::AssemblyEnv() {
        history_.reserve(MAX_STEPS);
        reset();
    }


    // Step 

    bool AssemblyEnv::step(int op, int rd, int rs1, int rs2, int rs3) {
        // Validate ranges
        if (op < 0 || op >= NUM_OPS)
            throw std::out_of_range(std::format("op={} out of [0,{})", op, NUM_OPS));
        if (rd < 0 || rd >= NUM_REGS)
            throw std::out_of_range(std::format("rd={} out of [0,{})", rd, NUM_REGS));
        if (rs1 < 0 || rs1 >= NUM_REGS)
            throw std::out_of_range(std::format("rs1={} out of [0,{})", rs1, NUM_REGS));
        if (rs2 < 0 || rs2 >= NUM_REGS)
            throw std::out_of_range(std::format("rs2={} out of [0,{})", rs2, NUM_REGS));
        if (rs3 < 0 || rs3 >= NUM_REGS)
            throw std::out_of_range(std::format("rs3={} out of [0,{})", rs3, NUM_REGS));

        if (steps_taken_ >= MAX_STEPS)
            throw std::runtime_error("Environment already at MAX_STEPS");
    

        // Record
        history_.push_back({
            static_cast<std::int8_t>(op),
            static_cast<std::int8_t>(rd),
            static_cast<std::int8_t>(rs1),
            static_cast<std::int8_t>(rs2),
            static_cast<std::int8_t>(rs3)
        });
        
        steps_taken_++;

        if (rd == 0) return steps_taken_ >= MAX_STEPS;
    
    
        for (int t = 0; t < NUM_TEST_CASES; ++t) {
            std::int32_t val1 = reg(t, rs1);
            std::int32_t val2 = reg(t, rs2);
            std::int32_t val3 = reg(t, rs3);

            std::int32_t result = 0;

            switch (static_cast<OpCode>(op)) {
                case OpCode::ADD:
                    result = val1 + val2;
                    break;
                case OpCode::SUB:
                    result = val1 - val2;
                    break;
                case OpCode::AND:
                    result = val1 & val2;
                    break;
                case OpCode::SLT:
                    result = (val1 < val2) ? 1 : 0;
                    break;
                case OpCode::CMOV:
                    result = (val3 != 0) ? val1 : val2;
                    break;
                default:
                    // Should never reach here due to range check above
                    result = reg(t, rd);
                    break;
            }

            reg(t, rd) = result;
            // x0 stays zero (rd==0 was handled above, but a sanity-check)
        }

        return steps_taken_ >= MAX_STEPS;
    }

    // Check for sorted

    bool AssemblyEnv::is_sorted() const {
        for (int t = 0; t < NUM_TEST_CASES; t++) {
            if (reg(t, 1) != 1 || reg(t, 2) != 2 || reg(t, 3) != 3) {
                return false;
            }
        }

        return true;
    }

    // Correctness

    float AssemblyEnv::correctness() const {
        int correct = 0;
        constexpr int total = NUM_TEST_CASES * 3;

        for (int i = 0; i < NUM_TEST_CASES; i++) {
            if (reg(i, 1) == 1) correct++;
            if (reg(i, 2) == 2) correct++;
            if (reg(i, 3) == 3) correct++;
        }

        return static_cast<float>(correct) / static_cast<float>(total);
    }

    float AssemblyEnv::reward() const {
        float correct = correctness();

        if (correct == 1.0f) {
            // fully sorted -> reward in [1.0, 2.0]
            // bonus for fewer steps
            float step_bonus = static_cast<float>(MAX_STEPS - steps_taken_) / static_cast<float>(MAX_STEPS);
            return 1.0f + step_bonus;
        }

        // not solved: reward in [-0.5, 0)
        return -0.5f + 0.5f * correct;
    }

    // Observation

    std::vector<float> AssemblyEnv::observe() const {
        std::vector<float> obs;
        obs.reserve(OBS_TOTAL_SIZE);

        // Section 0: Normalized register values
        for (int t = 0; t < NUM_TEST_CASES; ++t) {
            for (int r = 0; r < NUM_REGS; ++r) {
                obs.push_back(static_cast<float>(reg(t, r)) / REG_NORM);
            }
        }

        // Section 1: Program history as integer indices (0-padded for unused slots)
        // op:  [0, NUM_OPS-1],  pad = 0  → we store op+1 so padding is 0 and real ops are 1..5
        //  regs: [0, NUM_REGS-1], pad = 0  → we store reg+1 so padding is 0 and real regs are 1..8
        for (int i = 0; i < MAX_STEPS; ++i) {
            if (i < static_cast<int>(history_.size())) {
                const auto& instr = history_[static_cast<std::size_t>(i)];
                obs.push_back(static_cast<float>(instr.op  + 1));
                obs.push_back(static_cast<float>(instr.rd  + 1));
                obs.push_back(static_cast<float>(instr.rs1 + 1));
                obs.push_back(static_cast<float>(instr.rs2 + 1));
                obs.push_back(static_cast<float>(instr.rs3 + 1));
            } else {
                // Padding: 0 for all components
                for (int j = 0; j < 5; ++j) {
                    obs.push_back(0.0f);
                }
            }
        }

        // Section 2: Progress indicator
        obs.push_back(static_cast<float>(steps_taken_) / static_cast<float>(MAX_STEPS));

        assert(static_cast<int>(obs.size()) == OBS_TOTAL_SIZE);
        return obs;
    }

    // Clone

    std::unique_ptr<AssemblyEnv> AssemblyEnv::clone() const {
        auto copy = std::make_unique<AssemblyEnv>();
        copy -> registers_ = registers_;
        copy -> history_ = history_;
        copy -> steps_taken_ = steps_taken_;
    
        return copy;
    }

    // Debug util

    std::string AssemblyEnv::dump_regs() const {
        std::ostringstream ss;
        for (int t = 0; t < NUM_TEST_CASES; ++t) {
            ss << std::format("  Case {}: [", t);
            for (int r = 0; r < NUM_REGS; ++r) {
                if (r > 0) ss << ", ";
                ss << std::format("x{}={}", r, reg(t, r));
            }
            ss << "]\n";
        }
        return ss.str();
    }

} // namespace alphadev

