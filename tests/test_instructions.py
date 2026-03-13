"""
    - Tests for the Instructions module
"""

import pytest

from alphadev_environment.instructions import (
    OpCode, Operand, OperandType, Instruction, InstructionSet
)


class TestOperand:
    def test_register_operand(self):
        op = Operand(OperandType.REGISTER, 0)

        assert op.type == OperandType.REGISTER
        assert op.index == 0
        assert repr(op) == "R0"

    def test_memory_operand(self):
        op = Operand(OperandType.MEMORY, 3)

        assert op.type == OperandType.MEMORY
        assert op.index == 3
        assert repr(op) == "mem[3]"

    def test_operand_equality(self):
        op1 = Operand(OperandType.REGISTER, 0)
        op2 = Operand(OperandType.REGISTER, 0)
        op3 = Operand(OperandType.REGISTER, 1)

        assert op1 == op2
        assert op2 != op3

    def test_operand_immutability(self):
        op = Operand(OperandType.REGISTER, 0)
        with pytest.raises(AttributeError):
            op.index = 1


class TestInstruction:
    def test_instruction_creation(self):
        instr = Instruction(
            OpCode.MOV,
            Operand(OperandType.REGISTER, 0),
            Operand(OperandType.MEMORY, 0)
        )

        assert instr.opcode == OpCode.MOV
        assert repr(instr) == "MOV R0, mem[0]"


    def test_cmp_instruction(self):
        instr = Instruction(
            OpCode.CMP,
            Operand(OperandType.REGISTER, 0),
            Operand(OperandType.REGISTER, 1),
        )

        assert instr.opcode == OpCode.CMP
        assert repr(instr) == "CMP R0, R1"


class TestInstructionSet:
    def setup_method(self):
        self.iset = InstructionSet(num_registers=4, num_memory_slots=8)
    
    def test_action_space_nonempty(self):
        assert self.iset.num_actions > 0
    
    def test_no_self_mov(self):
        """
            - MOV R0, R0 should not be in the action space.
        """
        
        for instr in self.iset.get_all_instructions():
            if instr.opcode == OpCode.MOV:
                assert instr.operand1 != instr.operand2, f"Self-MOV found: {instr}"
    
    def test_no_mem_to_mem_mov(self):
        """
            - MOV mem[x], mem[y] should not be in the action space.
        """
        
        for instr in self.iset.get_all_instructions():
            if instr.opcode == OpCode.MOV:
                assert not (
                    instr.operand1.type == OperandType.MEMORY and
                    instr.operand2.type == OperandType.MEMORY
                ), f"Mem-to-mem MOV found: {instr}"
    
    def test_cmov_dst_is_register(self):
        """
            - CMOV* instructions must have register as destination.
        """
        
        cmov_opcodes = {OpCode.CMOVG, OpCode.CMOVL, OpCode.CMOVGE, OpCode.CMOVLE}
        for instr in self.iset.get_all_instructions():
            if instr.opcode in cmov_opcodes:
                assert instr.operand1.type == OperandType.REGISTER, f"CMOV with non-register dst: {instr}"
    
    def test_no_self_cmp(self):
        """
            - CMP R0, R0 should not be in the action space.
        """
        
        for instr in self.iset.get_all_instructions():
            if instr.opcode == OpCode.CMP:
                assert instr.operand1 != instr.operand2
    
    def test_roundtrip_conversion(self):
        """
            - action_id -> instruction -> action_id should be identity.
        """
        
        for action_id in range(self.iset.num_actions):
            instr = self.iset.action_to_instruction(action_id)
            recovered_id = self.iset.instruction_to_action(instr)
            assert recovered_id == action_id, f"Roundtrip failed for action {action_id}: {instr}"
    
    def test_encode_decode_roundtrip(self):
        """
            - encode -> decode should be identity.
        """

        for instr in self.iset.get_all_instructions():
            encoded = self.iset.encode_instructions(instr)
            decoded = self.iset.decode_instructions(encoded)
            assert decoded == instr, f"Encode/decode failed for {instr}"
    
    def test_action_out_of_range(self):
        with pytest.raises(ValueError):
            self.iset.action_to_instruction(-1)
        with pytest.raises(ValueError):
            self.iset.action_to_instruction(self.iset.num_actions)
    
    def test_unique_actions(self):
        """
            - All actions should be unique.
        """

        instructions = self.iset.get_all_instructions()
        assert len(instructions) == len(set(instructions))
    
    def test_small_instruction_set(self):
        """
            - Test with minimal registers/memory.
        """
        iset = InstructionSet(num_registers=2, num_memory_slots=2)
        assert iset.num_actions > 0
        
        # Should have MOV, CMP, CMOV variants
        opcodes_present = set(
            instr.opcode for instr in iset.get_all_instructions()
        )
        assert OpCode.MOV in opcodes_present
        assert OpCode.CMP in opcodes_present
        assert OpCode.CMOVG in opcodes_present

