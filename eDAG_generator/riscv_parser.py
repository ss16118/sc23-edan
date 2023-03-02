from typing import List, Optional, Dict, Tuple
import re
from instruction_parser import InstructionParser
from eDAG import Vertex, OpType


class RiscvParser(InstructionParser):
    """
    An eDAG parser which only targets the RISC-V assembly trace
    """
    # A dictionary that maps specific letters in instructions
    # to the size of data that is manipulated
    data_size_symbol = {
        "b": 1,
        "h": 2,
        "w": 4,
        "d": 8
    }
    load_instructions = {
        "lb", "lh", "lw", "ld", "li", "lhu", "lbu", "lui",
        "fld", "flw", "lwu"
    }
    store_instructions = { "sb", "sh", "sw", "sd", "fsd" }
    mv_instructions = { "mv", "sext.w", "fmv.d.x", "fmv.d" }
    # FIXME: Probably it is better to combine all the arithmetic
    # operations into one category
    add_instructions = { "addi", "addw", "addiw", "add" }
    sub_instructions = { "sub", "subw", "neg", "negw" }
    mul_instructions = { "mul", "mulw" }
    div_instructions = { "divw", "divuw", "div" }
    rem_instructions = { "remw", "remuw" }
    atomic_op_instructions = {
        "amoswap.w", "amoxor.w", "amoadd.w", "amoxor.w",
        "amoand.w", "amoor.w", "amomin.w", "amomax.u", "amominu.w", "amomaxu.w" 
    }
    comp_and_set_instructions = { "slti", "sltiu", "slt", "sltu" }
    bit_op_instructions = \
        { "xori", "ori", "andi", "slli", "srli", "srai", "sll", "xor", "srl", 
        "sra", "or", "and", "srliw", "sraiw", "sraw" }
    uncond_jump_instructions = { "j", "jal", "jalr" }
    cond_br_2ops_instructions = { "beqz", "bgez", "blez", "bgtz", "bnez" }
    cond_br_3ops_instructions = {
        "beq", "bne", "bge", "bgt", "bgtu", "blt", "bltu", "bgeu", "ble", "bleu"
    }
    ret_instructions = { "ret", "uret", "sret", "mret", "tail", "ecall" }

    # Floating point operations
    fp_2ops_instructions = { "fcvt.d.w", "fcvt.w.d" }
    fp_3ops_instructions = {
        "fsub.d", "fsub.s", "fmul.d", "fmul.s", "fmadd.s", "fdiv.d", "fdiv.s",
        "flt.d", "flt.s", "fadd.d", "fadd.s"
     }
    fp_4ops_instructions = { "fmadd.d", "fmadd.s", "fnmadd.d" }
    # Uncategorized instructions
    uncategorized_instructions = { "auipc" }

    # Associative and commutative operations
    comm_assoc_ops = {
        "xor", "and", "or", "mul", "add", "addw", "mulw",
        "fmul.d", "fmul.s", "fmadd.d", "fmadd.s"
    }

    reg_offset_pattern = re.compile(r"-?\d*\((\w+\d?)\)")

    # =========== Constructor ===========
    def __init__(self) -> None:
        super().__init__()


    def __get_offset_reg(self, operand: str) -> str:
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
    
    def is_ret_instruction(self, instruction: str) -> bool:
        """
        Returns True if the given instruction is considered `return`.
        """
        return instruction in RiscvParser.ret_instructions

    def get_load_data_addr(self, line: str) -> Optional[str]:
        """
        Given a single line from the assembly trace, returns the virtual
        memory address of the data that is loaded. Returns None
        if the given line is not a memory load instruction.
        """
        parsed = self.parse_line(line)
        if parsed is not None:
            insn = parsed["instruction"]
            assert(insn is not None)
            # Checks whether the instruction is loading from memory
            if insn in RiscvParser.load_instructions and "i" not in insn:
                data_addr = parsed["data_addr"]
                assert(data_addr is not None)
                return data_addr
        return None

    def get_insn_data_size(self, opcode: str) -> int:
        """
        Give the opcode of an instruction, returns the size of data which
        is manipulated.
        TODO: Currently only memory access operations are supported
        """
        for symbol, size in RiscvParser.data_size_symbol.items():
            if symbol in opcode:
                return size
        raise ValueError(f"[ERROR] Unknown data size for opcode: {opcode}")

    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parses a single line of assembly trace, and returns
        all the relevant information in a dictionary that contains the 
        following keys. Note that some of the values might be None.
        
        ["cpu_id", "insn_addr", "instruction", "operands", "data_addr"]
        
        If a return instruction is encountered, returns None.
        """
        res = dict.fromkeys(
            ["cpu", "insn_addr", "instruction", "operands", "data_addr"]
        )
        # Splits the line by the given delimiter
        tokens = line.split(";")
        if len(tokens) < 3:
            return None
        # The first two tokens are CPU ID and instruction address
        cpu_id, insn_addr, *tokens = tokens
        insn_tokens = tokens[0].split()
        instruction = insn_tokens[0]
        res["cpu"] = int(cpu_id)
        res["insn_addr"] = insn_addr
        res["instruction"] = instruction
        res["operands"] = []

        # Skips instructions that do not have any operands, e.g. ret
        if instruction in RiscvParser.ret_instructions:
            return res
        
        operands = insn_tokens[1].split(",")
        res["operands"] = operands
        if len(tokens) > 1:
            # The last token should be the memory address of the data
            # that has been accessed
            res["data_addr"] = tokens[-1]
        return res

    def generate_vertex(self, id: int, instruction: str,
                        operands: List[str], cpu: Optional[int] = None,
                        insn_addr: Optional[str] = None,
                        data_addr: Optional[str] = None) -> Vertex:
        """
        A function that converts the given assembly instruction
        and its corresponding operands into an eDAG vertex.
        """
        # TODO Not the most efficient way to do this, probably 
        # needs to be improved
        target = None
        op_type = None
        dependencies = set()
        data_size = 0
        is_comm_assoc = instruction in RiscvParser.comm_assoc_ops
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
                assert(data_addr is not None)
                dependencies.add(self.__get_offset_reg(operands[1]))
                dependencies.add(data_addr)
                data_size = self.get_insn_data_size(instruction)
                op_type = OpType.LOAD_MEM
            else:
                op_type = OpType.LOAD_IMM

        elif instruction in RiscvParser.store_instructions:
            assert(len(operands) == 2)
            # Target of a `load` instruction is always the second operand
            # target = operands[1]
            assert(data_addr is not None)
            target = data_addr
            dependencies.add(operands[0])
            dependencies.add(self.__get_offset_reg(operands[1]))
            # data_size = self.get_insn_data_size(instruction)
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
            target = operands[0]
            dependencies.add(operands[1])
            if instruction in { "neg", "negw" }:
                assert(len(operands) == 2)
            else:
                assert(len(operands) == 3)
                dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC

        elif instruction in RiscvParser.mul_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC
        
        elif instruction in RiscvParser.div_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC

        elif instruction in RiscvParser.rem_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC

        elif instruction in RiscvParser.atomic_op_instructions:
            # Atomic operations that require memory access
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(self.__get_offset_reg(operands[2]))
            # FIXME: Should probably have a separate Op type for atomic operations
            # The amount of data movement is data_size * 2 since an atomic
            # operation both loads and stores to memory
            data_size = self.get_insn_data_size(instruction) * 2
            op_type = OpType.STORE_MEM

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
            op_type = OpType.JUMP

        elif instruction in RiscvParser.cond_br_2ops_instructions:
            # Conditional branch instructions with 2 operands
            assert(len(operands) == 2)
            dependencies.add(operands[0])
            op_type = OpType.BRANCH

        elif instruction in RiscvParser.cond_br_3ops_instructions:
            # Conditional branch instructions with 3 operands
            assert(len(operands) == 3)
            dependencies.add(operands[0])
            dependencies.add(operands[1])
            op_type = OpType.BRANCH

        elif instruction in RiscvParser.fp_2ops_instructions:
            # Floating point operations with two operands
            # The number of elements in the operand list can
            # be either 2 or 3 depending on whether the round mode
            # field exists or not
            assert(len(operands) == 2 or len(operands) == 3)
            if len(operands) == 2:
                # If rounding mode does not exist
                target = operands[0]
                dependencies.add(operands[1])
            else:
                # If rounding mode exists, ignore it
                target = operands[1]
                dependencies.add(operands[2])
            op_type = OpType.ARITHMETIC
        
        elif instruction in RiscvParser.fp_3ops_instructions:
            # Floating point operations with three operands
            assert(len(operands) == 3 or len(operands) == 4)
            if len(operands) == 3:
                # If rounding mode does not exist
                target = operands[0]
                dependencies.add(operands[1])
                dependencies.add(operands[2])
            else:
                # If rounding mode exists, ignore it
                target = operands[1]
                dependencies.add(operands[2])
                dependencies.add(operands[3])
            op_type = OpType.ARITHMETIC

        elif instruction in RiscvParser.fp_4ops_instructions:
            # Floating point operations with four operands
            assert(len(operands) == 4 or len(operands) == 5)
            if len(operands) == 4:
                # If rounding mode does not exist
                target = operands[0]
                dependencies.update(operands[1:])
            else:
                # If rounding mode does exist, ignore it
                target = operands[1]
                dependencies.update(operands[2:])
            op_type = OpType.ARITHMETIC
        
        elif instruction in RiscvParser.ret_instructions:
            if instruction == "ret":
                # In RISC-V 'ret' is a pseudo-instruction that
                # depends on the value in register `ra`
                assert(len(operands) == 0)
                dependencies.add("ra")
            op_type = OpType.JUMP

        elif instruction in RiscvParser.uncategorized_instructions:
            # Uncategorized instructions
            if instruction == "auipc":
                assert(len(operands) == 2)
                target = operands[0]
            op_type == OpType.UNCATEGORIZED

        else:
            # An unknown instruction has been encountered
            raise ValueError(f"[ERROR] Unknown instruction {instruction}")

        new_vertex = Vertex(id, instruction, operands,
                            target=target, dependencies=dependencies,
                            op_type=op_type, is_comm_assoc=is_comm_assoc,
                            cpu=cpu, insn_addr=insn_addr, data_addr=data_addr,
                            data_size=data_size)

        return new_vertex
        