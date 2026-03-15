"""
    ## Simulates CPU state: registers, memory, and flags.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import copy

from .instructions import (
    Opcode, Operand, OperandType, Instruction
)


@dataclass
class Flags:
    """
        ## CPU flags register, set by CMP instruction.
        
        CMP op1, op2 computes (op1 - op2) and sets:
        zero: True if op1 == op2
        sign: True if op1 < op2 (result is negative)
        overflow: Always False for our integer arithmetic
        
        These flags determine CMOV behavior:
            - CMOVG:  not zero AND not sign (op1 > op2)
            - CMOVL:  sign (op1 < op2)
            - CMOVGE: not sign (op1 >= op2)  
            - CMOVLE: zero OR sign (op1 <= op2)
    """

    zero: bool = False
    sign: bool = False
    valid: bool = False    # Whether flags have been set by a CMP

    def copy(self) -> 'Flags':
        return Flags(
            zero=self.zero,
            sign=self.sign,
            valid=self.valid
        )


class CPUState:
    """
        ## Complete CPU state for the assembly environment.
    
        State consists of:
            - registers: List of integer values (R0, R1, R2, R3)
            - memory: List of integer values (mem[0] through mem[N-1])  
            - flags: Comparison flags set by CMP
    """
    
    def __init__(self, num_registers: int = 4, num_memory_slots: int = 8):
        self.num_registers = num_registers
        self.num_memory_slots = num_memory_slots
        self.registers: List[Optional[int]] = [None] * num_registers
        self.memory: List[Optional[int]] = [None] * num_memory_slots
        self.flags = Flags()
    
    def copy(self) -> 'CPUState':
        """
            ## Deep copy the CPU state.
        """

        new_state = CPUState(self.num_registers, self.num_memory_slots)
        new_state.registers = list(self.registers)
        new_state.memory = list(self.memory)
        new_state.flags = self.flags.copy()
        return new_state
    
    
    def read_operand(self, operand: Operand) -> Optional[int]:
        """
            ## Read value from a register or memory location.
        """

        if operand.type == OperandType.REGISTER:
            if operand.index >= self.num_registers:
                raise IndexError(
                    f"Register R{operand.index} out of range "
                    f"(max R{self.num_registers - 1})"
                )
            return self.registers[operand.index]
        
        else:
            if operand.index >= self.num_memory_slots:
                raise IndexError(
                    f"Memory slot {operand.index} out of range "
                    f"(max {self.num_memory_slots - 1})"
                )
            return self.memory[operand.index]
    
    
    def write_operand(self, operand: Operand, value: Optional[int]):
        """
            ## Write value to a register or memory location.
        """

        if operand.type == OperandType.REGISTER:
            if operand.index >= self.num_registers:
                raise IndexError(
                    f"Register R{operand.index} out of range"
                )
            self.registers[operand.index] = value

        else:
            if operand.index >= self.num_memory_slots:
                raise IndexError(
                    f"Memory slot {operand.index} out of range"
                )
            self.memory[operand.index] = value
    
    def load_input(self, values: List[int], start_memory_index: int = 0):
        """
            ## Load input values into memory for sorting.
        
            For sort_size=3 with values [3,1,2]:
            mem[0] = 3, mem[1] = 1, mem[2] = 2
        """

        for i, val in enumerate(values):
            if start_memory_index + i >= self.num_memory_slots:
                raise IndexError(
                    f"Cannot load {len(values)} values starting at "
                    f"memory index {start_memory_index}"
                )
            self.memory[start_memory_index + i] = val
    
    def get_output(self, sort_size: int, start_memory_index: int = 0) -> List[Optional[int]]:
        """
            ## Read output values from memory after program execution.
        """
        return [
            self.memory[start_memory_index + i] 
            for i in range(sort_size)
        ]
    
    def __repr__(self):
        reg_str = ", ".join(
            f"R{i}={v}" for i, v in enumerate(self.registers)
        )

        mem_str = ", ".join(
            f"[{i}]={v}" for i, v in enumerate(self.memory) 
            if v is not None
        )

        flag_str = (
            f"Z={self.flags.zero}, S={self.flags.sign}" 
            if self.flags.valid else "unset"
        )

        return f"CPU({reg_str} | {mem_str} | flags: {flag_str})"

    def __eq__(self, other):
        if not isinstance(other, CPUState):
            return False
        
        return (
            self.registers == other.registers and
            self.memory == other.memory and
            self.flags.zero == other.flags.zero and
            self.flags.sign == other.flags.sign and
            self.flags.valid == other.flags.valid
        )


def execute_instruction(state: CPUState, instruction: Instruction) -> CPUState:
    """
        ## Execute a single instruction, returning the new CPU state.
    
        ## Returns 
            - a NEW CPUState (functional style for safety).
    """
    new_state = state.copy()
    opcode = instruction.opcode
    op1 = instruction.operand1
    op2 = instruction.operand2
    
    if opcode == Opcode.MOV:
        # MOV dst, src: dst = src
        src_val = new_state.read_operand(op2)
        # Read FIRST, then write (handles case where op1 == op2 safely,
        # though we filter those out in InstructionSet)
        new_state.write_operand(op1, src_val)
    
    elif opcode == Opcode.CMP:
        # CMP op1, op2: compute op1 - op2, set flags
        
        val1 = new_state.read_operand(op1)
        val2 = new_state.read_operand(op2)
        
        if val1 is not None and val2 is not None:
            diff = val1 - val2
            new_state.flags.zero = (diff == 0)
            new_state.flags.sign = (diff < 0)
            new_state.flags.valid = True
        else:

            # Comparing with uninitialized values: flags become invalid
            new_state.flags.valid = False
    
    elif opcode in (Opcode.CMOVG, Opcode.CMOVL, Opcode.CMOVGE, Opcode.CMOVLE):
        # CMOV* dst, src: conditionally move src to dst based on flags
        if not new_state.flags.valid:
            # If flags are not valid, CMOV is a no-op
            # (This prevents the agent from using CMOV without CMP)
            pass
        
        else:
            should_move = False
            
            if opcode == Opcode.CMOVG:
                # Move if greater: not zero AND not sign
                should_move = (not new_state.flags.zero and 
                              not new_state.flags.sign)
            elif opcode == Opcode.CMOVL:
                # Move if less: sign is set
                should_move = new_state.flags.sign
            elif opcode == Opcode.CMOVGE:
                # Move if greater or equal: not sign
                should_move = not new_state.flags.sign
            elif opcode == Opcode.CMOVLE:
                # Move if less or equal: zero OR sign
                should_move = (new_state.flags.zero or 
                              new_state.flags.sign)
            
            if should_move:
                src_val = new_state.read_operand(op2)
                new_state.write_operand(op1, src_val)
    
    return new_state