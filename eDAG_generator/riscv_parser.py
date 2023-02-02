from typing import List, Optional, Dict
import re
from instruction_parser import InstructionParser
from eDAG import Vertex, OpType


class RiscvParser(InstructionParser):
    """
    An eDAG parser which only targets the RISC-V assembly trace
    """

    load_instructions = { "lb", "lh", "lw", "ld", "li", "lhu", "lbu", "lui" }
    store_instructions = { "sb", "sh", "sw", "sd" }
    mv_instructions = { "mv", "sext.w" }
    add_instructions = { "addi", "addw", "addiw", "add" }
    sub_instructions = { "sub", "subw" }
    mul_instructions = { "mul", "mulw" }
    comp_and_set_instructions = { "slti", "sltiu", "slt", "sltu" }
    bit_op_instructions = \
        { "xori", "ori", "andi", "slli", "srli", "srai", "sll", "xor", "srl", 
        "sra", "or", "and", "srliw", "sraiw" }
    uncond_jump_instructions = { "j", "jal", "jalr" }
    cond_jump_2ops_instructions = { "bgez", "blez" }
    cond_jump_3ops_instructions = \
        { "beq", "bne", "bge", "blt", "bltu", "bgeu", "ble"}
    ret_instructions = { "ret", "uret", "sret", "mret", "tail" }

    reg_offset_pattern = re.compile(r"-?\d+\((\w+\d?)\)")

    # =========== Constructor ===========
    def __init__(self) -> None:
        super().__init__()


    def _get_offset_reg(self, operand: str) -> str:
        """
        A private helper function that returns the offset register used
        in register-offset addressing mode. Uses regex to parse the given
        operand string.
        """
        matches = RiscvParser.reg_offset_pattern.match(operand)
        if matches is None:
            raise ValueError(f"[ERROR] Invalid operand for register-offset addressing mode: {operand}")
        offset_reg = matches.group(1)
        return offset_reg

    def generate_vertex(self, id: int, instruction: str,
                        operands: List[str], cpu: Optional[int] = None,
                        insn_addr: Optional[str] = None) -> Vertex:
        """
        A function that converts the given assembly instruction
        and its corresponding operands into an eDAG vertex.
        """
        # TODO Not the most efficient way to do this, probably 
        # needs to be improved
        target = None
        op_type = None
        dependencies = set()

        # Iterates through all types of instructions and checks to which
        # group of instructions it belongs
        if instruction in RiscvParser.load_instructions:
            assert(len(operands) == 2)
            # Target is always the first operand
            target = operands[0]
            # Letter 'i' in the instruction indicates that
            # it is not accessing memory
            if 'i' not in instruction:
                # The addressing mode is register-offset
                dependencies.add(self._get_offset_reg(operands[1]))
                dependencies.add(operands[1])
                op_type = OpType.LOAD_MEM
            else:
                op_type = OpType.LOAD_IMM

        elif instruction in RiscvParser.store_instructions:
            assert(len(operands) == 2)
            # Target of a `load` instruction is always the second operand
            target = operands[1]
            dependencies.add(operands[0])
            dependencies.add(self._get_offset_reg(operands[1]))
            op_type = OpType.STORE_MEM

        elif instruction in RiscvParser.mv_instructions:
            assert(len(operands) == 2)
            target = operands[0]
            dependencies.add(operands[1])
            op_type = OpType.MOVE

        elif instruction in RiscvParser.add_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            if 'i' not in instruction:
                dependencies.add(operands[2])
                op_type = OpType.ARITHMETIC
            else:
                op_type = OpType.ARITHMETIC_IMM
                
        elif instruction in RiscvParser.sub_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC

        elif instruction in RiscvParser.mul_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC

        elif instruction in RiscvParser.comp_and_set_instructions:
            # Compare and set instructions
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            if 'i' not in instruction:
                dependencies.add(operands[2])
                op_type = OpType.ARITHMETIC
            else:
                op_type = OpType.ARITHMETIC_IMM
                
        elif instruction in RiscvParser.bit_op_instructions:
            # Bit-wise operation instructions
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            if 'i' not in instruction:
                dependencies.add(operands[2])
                op_type = OpType.ARITHMETIC
            else:
                op_type = OpType.ARITHMETIC_IMM
            
        elif instruction in RiscvParser.uncond_jump_instructions:
            # Unconditional jump instructions
            if instruction == "jal":
                assert(len(operands) == 2)
                target = operands[0]
            elif instruction == "jalr":
                assert(len(operands) == 3)
                target = operands[0]
                dependencies.add(operands[1])
            else:
                assert(len(operands) == 1)
            op_type = OpType.BRANCH

        elif instruction in RiscvParser.cond_jump_2ops_instructions:
            # Conditional jump instructions with 2 operands
            assert(len(operands) == 2)
            dependencies.add(operands[0])
            op_type = OpType.BRANCH

        elif instruction in RiscvParser.cond_jump_3ops_instructions:
            # Conditional jump instructions with 3 operands
            assert(len(operands) == 3)
            dependencies.add(operands[0])
            dependencies.add(operands[1])
            op_type = OpType.BRANCH
        else:
            # An unknown instruction has been encountered
            raise ValueError(f"[ERROR] Unknown instruction {instruction}")

        # if target is not None:
        #     # Adds the target as a dependency
        #     dependencies.add(target)

        new_vertex = Vertex(id, instruction, operands,
                            target=target, dependencies=dependencies,
                            op_type=op_type, cpu=cpu, insn_addr=insn_addr)

        return new_vertex
        