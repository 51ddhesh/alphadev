"""
    ## Program representation and execution

    "A program is an ordered sequence of assembly instructions."
    This module handles running a complete program on inputs.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import copy

from .instructions import Instruction, InstructionSet
from .cpu_state import CPUState, execute_instruction

@dataclass
class Program:
    """
        ## Represents an assembly program as a sequqnce of instructions
    """

    instructions: List[Instruction] = field(default_factory=list)


    def append(self, instruction: Instruction):
        """
            ## Add an instruction to the end of the program
        """

        self.instructions.append(instruction)


    def pop(self) -> Instruction:
        """
            ## Remove and return the last instruction
        """

        return self.instructions.pop()
    
    @property
    def length(self) -> int:
        return len(self.instructions)
    
    def copy(self) -> 'Program':
        """
            ## Deep copy of the program
        """

        return Program(instructions=list(self.instructions))
    

    def __repr__(self):
        lines = [f"    {i}: {instr}" for i, instr in enumerate(self.instructions)]
        return "Program(\n" + "\n".join(lines) + "\n)"


def execute_program(
    program: Program,
    initial_state: CPUState,
) -> CPUState:
    """
        ## Executes a complete program starting from the given CPU state.

        ## Returns the final CPU state after all instructions have executed.
    """

    state = initial_state.copy()

    for instruction in program.instructions:
        state = execute_instruction(state, instruction)

    return state



def run_sort_program(
    program: Program,
    input_values: List[int],
    num_registers: int = 4,
    num_memory_slots: int = 8,
) -> List[Optional[int]]:
    """
        ## Run a sorting program on a given input.
        
        1. Loads input values into memory starting at index 0
        2. Executes the program
        3. Returns the memory contents (the sorted output)
            
        ## Args:
            - program: The assembly program to execute
            - input_values: Values to sort
            - num_registers: Number of CPU registers
            - num_memory_slots: Number of memory slots
        
        ## Returns:
            List of output values from memory[0..sort_size-1]
    """
    state = CPUState(
        num_registers=num_registers,
        num_memory_slots=num_memory_slots
    )
    state.load_input(input_values)
    
    final_state = execute_program(program, state)
    return final_state.get_output(len(input_values))


def program_to_action_sequence(
    program: Program, 
    instruction_set: InstructionSet
) -> List[int]:
    """
        ## Convert a program to a sequence of action IDs.
    """

    return [
        instruction_set.instruction_to_action(instr) 
        for instr in program.instructions
    ]