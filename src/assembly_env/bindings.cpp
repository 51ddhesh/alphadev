#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "assembly_env.hpp"

namespace py = pybind11;
using namespace alphadev;

PYBIND11_MODULE(alphadev_env, m) {
    m.doc() = "AlphaDev Assembly Environment (Sort-3)";

    // ─── Constants ─────────────────────────────────
    m.attr("NUM_OPS") = NUM_OPS;
    m.attr("NUM_REGS") = NUM_REGS;
    m.attr("NUM_TEST_CASES") = NUM_TEST_CASES;
    m.attr("MAX_STEPS") = MAX_STEPS;
    m.attr("OBS_SIZE") = OBS_TOTAL_SIZE;
    
    
    // ─── OpCode Enum ───────────────────────────────
    py::enum_<OpCode>(m, "OpCode")
    .value("ADD",  OpCode::ADD)
    .value("SUB",  OpCode::SUB)
    .value("AND",  OpCode::AND)
    .value("SLT",  OpCode::SLT)
    .value("CMOV", OpCode::CMOV)
    .export_values();
    

    // ─── Instruction ───────────────────────────────
    py::class_<Instruction>(m, "Instruction")
    .def_readonly("op",  &Instruction::op)
    .def_readonly("rd",  &Instruction::rd)
    .def_readonly("rs1", &Instruction::rs1)
    .def_readonly("rs2", &Instruction::rs2)
    .def_readonly("rs3", &Instruction::rs3)
    .def("__repr__", &Instruction::to_string);


    // ─── Environment ───────────────────────────────
    py::class_<AssemblyEnv>(m, "AssemblyEnv")
        .def(py::init<>())

        .def("reset", &AssemblyEnv::reset,
             "Reset environment to initial state with all permutations loaded")

        .def("step", &AssemblyEnv::step,
             py::arg("op"), py::arg("rd"),
             py::arg("rs1"), py::arg("rs2"), py::arg("rs3"),
             "Execute one instruction. Returns True if MAX_STEPS reached.")

        .def("is_sorted", &AssemblyEnv::is_sorted,
             "Check if x1<=x2<=x3 across all test cases")

        .def("is_done", &AssemblyEnv::is_done,
             "True if step count >= MAX_STEPS")

        .def("num_steps", &AssemblyEnv::num_steps,
             "Number of instructions executed so far")

        .def("correctness", &AssemblyEnv::correctness,
             "Fraction of output positions that are correct [0, 1]")

        .def("reward", &AssemblyEnv::reward,
             "Current reward signal")

        .def("observe", &AssemblyEnv::observe,
             "Get flat observation vector of size OBS_SIZE")

        .def_static("obs_size", &AssemblyEnv::obs_size,
             "Size of the observation vector")

        .def("clone", &AssemblyEnv::clone,
             "Deep copy of the environment")

        .def("program", [](const AssemblyEnv& env) {
                std::vector<Instruction> prog;
                auto span = env.program();
                prog.assign(span.begin(), span.end());
                return prog;
             },
             "Get the current program as a list of Instructions")

        .def("dump_regs", &AssemblyEnv::dump_regs,
             "Debug string of all register states")

        .def("__repr__", [](const AssemblyEnv& env) {
            return std::format("<AssemblyEnv steps={}/{} sorted={} correctness={:.1%}>",
                env.num_steps(), MAX_STEPS, env.is_sorted(), env.correctness());
        });
}

