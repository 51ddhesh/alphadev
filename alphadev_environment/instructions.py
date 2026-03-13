"""
    Assembly Instruction Set for AlphaDev

    - Implements a simplified but realistic x86 -esque instruction set focused on 
      the operations needed for sorting networks (MOV, CMOV, CMP)
    
    - In the AlphaDev paper, the agent operates on a RISC-like subset of assembly
      to keep the action space tractable while still being expressive enough to 
      discover novel sorting algorithms.

    - Instruction Encoding
        - Each instruction is a tuple (opcode, operand1, operand2)
          where operands can be registers or memory locations.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple
import itertools

class OpCode(IntEnum):
    """
        - Assembly opcodes available to the agent.

        MOV: Move data between registers/memory
        CMP: Compare two values, set flags
        CMOVG: Conditional move if greater (flags-based)
        CMOVL: Conditional move if less (flags-based)
        CMOVGE: Conditional move if greater or equal
        CMOVLE: Conditional move if less or equal 
    """

    MOV = 0
    CMP = 1
    
    # Conditional move if greater
    CMOVG = 2    
    
    # Conditional move if less
    CMOVL = 3    
    
    # Conditional move if greater or equal
    CMOVGE = 4   
    
    # Conditional move if less or equal
    CMOVLE = 5   


    @staticmethod
    def num_opcodes() -> int:   
        return 6
    

class OperandType(IntEnum):
    """
        - Type of operand
    """

    REGISTER = 0
    MEMORY = 1


@dataclass(frozen=True)
class Operand:
    """
        ## Represents an instruction operand.

        - type: REGISTER or MEMORY
        - index: which register (0-3) or memory slot (0-7)
    """

    type: OperandType
    index: int

    def __repr__(self):
        if self.type == OperandType.REGISTER:
            return f"R{self.index}"
        
        else:
            return f"mem[{self.index}]"
        

class Instruction(NamedTuple):
    """
        ## A single assembly instruction
    """

    opcode: OpCode
    operand1: Operand
    operand2: Operand

    def __repr__(self):
        return f"{self.opcode.name} {self.operand1}, {self.operand2}"
    

class InstructionSet:
    """
        ## Manages the full set of valid instructions (the action space).

        - The action space is the Cartesian product of:
            opcodes * valid_operand1 * valid_operand2

        - With constraints:
            - MOV: dst can be mem/reg, src can be mem/reg (but not mem-to-men on x86)
            - CMP: both operands must be readable (reg or mem, at most one mem)
            - CMOV*: dst must be a register, src can be reg/mem
    """

    num_registers: int
    num_memory_slots: int

    def __init__(self, num_registers: int = 4, num_memory_slots: int = 8):
        self.num_registers = num_registers
        self.num_memory_slots = num_memory_slots

        self._registers = [
            Operand(OperandType.REGISTER, i) for i in range(num_registers)
        ]

        self._memory = [
            Operand(OperandType.MEMORY, i) for i in range(num_memory_slots)
        ]

        self._all_operands = self._registers + self._memory

        # Build actions table: action_id -> Instructions
        self._actions: List[Instruction] = []
        self._action_to_id: dict = {}
        self._build_action_space()

    
    def _build_action_space(self):
        """
            ## Build the complete action space with architectural constraints
        """

        actions = []

        for opcode in OpCode:
            if opcode == OpCode.MOV:
                # MOV dst, src: dst=reg|mem, src=region
                # Constraint: no mem-to-mem (x86 limitation)

                for dst in self._all_operands:
                    for src in self._all_operands:
                        if dst == src:
                            continue # MOV R0, R0 is a NOP, skip

                        if (dst.type == OperandType.MEMORY and src.type == OperandType.MEMORY):
                            continue

                        actions.append(Instruction(opcode, dst, src))

            elif opcode == OpCode.CMP:
                # CMP op1, op2: compares op1 - op2, sets flags
                # At most one memory operand

                for op1 in self._all_operands:
                    for op2 in self._all_operands:
                        if op1 == op2:
                            continue # CMP R0, R0 always equal, useless

                        if (op1.type == OperandType.MEMORY and op2.type == OperandType.MEMORY):
                            continue

                        actions.append(Instruction(opcode, op1, op2))

            elif opcode in (OpCode.CMOVG, OpCode.CMOVL, OpCode.CMOVGE, OpCode.CMOVLE):
                # CMOV dst, src: dst must be a register, src can be reg|mem

                for dst in self._registers:
                    for src in self._all_operands:
                        if dst == src:
                            continue

                        actions.append(Instruction(opcode, dst, src))

        self._actions = actions
        self._action_to_id = {
            instr: idx for idx, instr in enumerate(actions)
        }
    
    @property
    def num_actions(self) -> int:
        """
            ## Returns the total number of valid actions
        """

        return len(self._actions)
    

    def action_to_instruction(self, action_id: int) -> Instruction:
        """
            ## Convert action ID to instructions
        """

        if action_id < 0 or action_id >= len(self._actions):
            raise ValueError(
                f"Action ID {action_id} out of range [0, {len(self._actions)})"
            )
        
        return self._actions[action_id]
    

    def instruction_to_action(self, instruction: Instruction) -> int:
        """
            ## Convert instruction to action ID
        """

        if instruction not in self._action_to_id:
            raise ValueError(
                f"Invalid Instruction: {instruction}"
            )
        
        return self._action_to_id[instruction]
    

    def get_all_instructions(self) -> List[Instruction]:
        """
            ## Returns all valid instructions
        """

        return list(self._actions)
    
    
    def encode_instructions(self, instruction: Instruction) -> Tuple[int, int, int, int, int]:
        """
            ## Encode instructions as a fixed size tuple
            - Returns a tuple for neural network input.
            - Return:
                `(opcode, op1_type, op1_index, op2_type, op2_index)`
        """

        return (
            int(instruction.opcode),
            int(instruction.operand1.type),
            instruction.operand1.index,
            int(instruction.operand2.type),
            instruction.operand2.index,
        )

    def decode_instructions(self, encoded: Tuple[int, int, int, int, int]) -> Instruction:
        """
            ## Decode a fixed size tuple back to an instruction
        """

        opcode = OpCode(encoded[0])
        op1 = Operand(OperandType(encoded[1]), encoded[2])
        op2 = Operand(OperandType(encoded[3]), encoded[4])  

        return Instruction(opcode, op1, op2)