#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <compare>

namespace alphadev {
    
    // ─── Instruction Set Definition ──────────────────────
    
    enum class OpCode : std::int8_t {
        ADD = 0, // rd = rs1 + rs2
        SUB = 1, // rd = rs1 - rs2
        AND = 2, // rd = rs1 & rs2
        SLT = 3, // rd = (rs1 < rs2) ? 1 : 0
        CMOV = 4, // rd = (rs3 != 0) ? rs1 : rs2
        COUNT = 5
    }; 
    
    inline constexpr int NUM_OPS = static_cast<int>(OpCode::COUNT);
    inline constexpr int NUM_REGS = 8; // x0, x1, x2, ..., x7
    
    // ─── Environment Constants ───────────────────────────
    // Sort-3: all 3! = 6 permutations of {1, 2, 3}
    
    inline constexpr int NUM_TEST_CASES = 6;
    inline constexpr int MAX_STEPS = 20;
    
    // Observation Layout
    //   Section 0 : registers   NUM_TEST_CASES * NUM_REGS  = 48 floats (normalized)
    //   Section 1 : program     MAX_STEPS * 5              = 100 ints  (0-padded)
    //   Section 2 : meta        1 float (step_count / MAX_STEPS)
    
    inline constexpr int OBS_REG_SIZE = NUM_TEST_CASES * NUM_REGS; // 48
    inline constexpr int OBS_PROG_SIZE = MAX_STEPS * 5; // 100
    inline constexpr int OBS_META_SIZE = 1; // 1
    inline constexpr int OBS_TOTAL_SIZE = OBS_REG_SIZE + OBS_PROG_SIZE + OBS_META_SIZE; // 149
    
    // Normalization constant: register values are divided by this
    inline constexpr float REG_NORM = 4.0f;
    
    
    // ─── Instructions ────────────────────────────────────
    
    struct Instruction {
        std::int8_t op;
        std::int8_t rd;
        std::int8_t rs1;
        std::int8_t rs2;
        std::int8_t rs3;
        
        [[nodiscard]] auto operator<=>(const Instruction&) const = default;
        [[nodiscard]] std::string to_string() const;
    };
    
    // ─── Assembly Environment ───────────────────────────
    
    class AssemblyEnv {
        public:
        
        AssemblyEnv();
        
        // Core 

        void reset();
        bool step(int op, int rd, int rs1, int rs2, int rs3);
        
        // Queries
        
        [[nodiscard]] bool is_done() const { return steps_taken_ >= MAX_STEPS; } 
        [[nodiscard]] int num_steps() const { return steps_taken_; }
        [[nodiscard]] bool is_sorted() const;
        [[nodiscard]] float correctness() const;
        [[nodiscard]] float reward() const;
        
        // Observation
        
        [[nodiscard]] std::vector<float> observe() const;
        [[nodiscard]] static constexpr int obs_size() { return OBS_TOTAL_SIZE; }
        
        // Deep Copy
        
        [[nodiscard]] std::unique_ptr<AssemblyEnv> clone() const;
        
        // Inspection
        
        [[nodiscard]] std::span<const Instruction> program() const { return history_; }
        [[nodiscard]] std::string dump_regs() const;
        
        private:
        
        // Registers: Flat array [test_case][register]
        std::array<std::int32_t, NUM_TEST_CASES * NUM_REGS> registers_ {};
        
        // Program built so far
        std::vector<Instruction> history_;
        
        int steps_taken_ = 0;
        
        // Internal Helpers
        
        void load_permutations();
        
        [[nodiscard]] std::int32_t& reg(int test_case, int reg_index) {
            return registers_[static_cast<std::size_t>(test_case * NUM_REGS + reg_index)];
        }
        
        [[nodiscard]] const std::int32_t& reg(int test_case, int reg_index) const {
            return registers_[static_cast<std::size_t>(test_case * NUM_REGS + reg_index)];
        }
    };
    
} // namespace alphadev