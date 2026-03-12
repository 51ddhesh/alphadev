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
    

class Operand(IntEnum):
    """
        - Type of operand
    """

    REGISTER = 0
    MEMORY = 1