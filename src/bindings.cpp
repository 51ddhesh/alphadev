#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "assembly_env.hpp"

namespace py = pybind11;

PYBIND11_MODULE(alphadev, m) {
    m.doc() = "RISC-V Subset of AlphaDev styled Assembly Environment";

    py::enum_<OpCode>(m, "OpCode")
        .value("ADD", OP_ADD)
        .value("SUB", OP_SUB)
        .value("AND", OP_AND)
        .value("SLT", OP_SLT)
        .value("CMOV", OP_CMOV)
        .export_values();

    py::class_<AssemblyEnv>(m, "AssemblyEnv")
        .def(py::init<>()) // constructor
        .def("reset", &AssemblyEnv::reset, "Reset the permutations")

        // return true if steps completed
        .def("step", &AssemblyEnv::step, 
            py::arg("op"), py::arg("rd"), py::arg("rs1"), py::arg("rs2"), py::arg("rs3"),
            "Executes one instruction. Returns true if max steps reached")

        .def("get_score", &AssemblyEnv::get_score, "Get the current reward")
        
        .def("is_sorted", &AssemblyEnv::is_sorted, "Check if all cases are sorted")

        .def("observe", &AssemblyEnv::observe, "Get flat observation vector");
}
