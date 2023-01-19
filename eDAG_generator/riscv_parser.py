from typing import List, Optional, Dict
import re
from edag_parser import EDagParser
from eDAG import Vertex, EDag


class RiscvParser(EDagParser):
    """
    An eDAG parser which only targets the RISC-V assembly trace
    """

    load_instructions = { "lb", "lh", "lw", "ld", "li", "lhu", "lbu", "lui" }
    store_instructions = { "sb", "sh", "sw", "sd" }
    mv_instructions = { "mv", "sext.w" }
    add_instructions = { "addi", "addw", "addiw", "add" }
    sub_instructions = { "sub" }
    mul_instructions = { "mul", "mulw" }
    comp_and_set_instructions = { "slti", "sltiu", "slt", "sltu" }
    bit_op_instructions = \
        { "xori", "ori", "andi", "slli", "srli", "srai", "sll", "xor", "srl", 
        "sra", "or", "and" }
    uncond_jump_instructions = { "j", "jal", "jalr" }
    cond_jump_instructions = \
        { "beq", "bne", "bge", "blt", "bltu", "bgeu", "bgez" }
    ret_instructions = { "ret", "uret", "sret", "mret" }

    reg_offset_pattern = re.compile(r"-?\d+\((\w+\d?)\)")

    # =========== Constructor ===========
    def __init__(self, trace_file: str, only_mem_acc: bool = False,
                remove_single_vertices: bool = True) -> None:
        """
        Takes in the path to the trace file as input for the constructor
        """
        super().__init__(trace_file, only_mem_acc, remove_single_vertices)


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

    def _instruction_to_vertex(self, id: int, instruction: str,
                                operands: List[str]) -> Vertex:
        """
        An private helper function that converts the given assembly instruction
        and its corresponding operands into an eDAG vertex.
        """
        # TODO Not the most efficient way to do this, probably 
        # needs to be changed
        target = None
        is_mem_acc = False
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
                is_mem_acc = True

        elif instruction in RiscvParser.store_instructions:
            assert(len(operands) == 2)
            # Target of a `load` instruction is always the second operand
            target = operands[1]
            dependencies.add(operands[0])
            dependencies.add(self._get_offset_reg(operands[1]))
            is_mem_acc = True

        elif instruction in RiscvParser.mv_instructions:
            assert(len(operands) == 2)
            target = operands[0]
            dependencies.add(operands[1])

        elif instruction in RiscvParser.add_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            if 'i' not in instruction:
                dependencies.add(operands[2])
                
        elif instruction in RiscvParser.sub_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])

        elif instruction in RiscvParser.mul_instructions:
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            dependencies.add(operands[2])

        elif instruction in RiscvParser.comp_and_set_instructions:
            # Compare and set instructions
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            if 'i' not in instruction:
                dependencies.add(operands[2])
                
        elif instruction in RiscvParser.bit_op_instructions:
            # Bit-wise operation instructions
            assert(len(operands) == 3)
            target = operands[0]
            dependencies.add(operands[1])
            if 'i' not in instruction:
                dependencies.add(operands[2])
            
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

        elif instruction in RiscvParser.cond_jump_instructions:
            # Conditional jump instructions
            assert(len(operands) == 3)
            dependencies.add(operands[0])
            dependencies.add(operands[1])

        else:
            # An unknown instruction has been encountered
            raise ValueError(f"[ERROR] Unknown instruction {instruction}")

        if target is not None:
            # Adds the target as a dependency
            dependencies.add(target)

        new_vertex = Vertex(id, instruction, operands,
                            target=target, dependencies=dependencies,
                            is_mem_acc=is_mem_acc)

        return new_vertex
        

    def parse(self) -> EDag:
        riscv_trace = open(self.trace_file, "r")
        lines = riscv_trace.readlines()

        eDag = EDag()
        # A dictionary that maps a register or memory location represented
        # as a string to the most recent vertex which updated its value
        curr_vertex: Dict[str, Vertex] = {}
        # A number that is used to uniquely identify each vertex
        # increments after a vertex is added to the eDAG
        vertex_id = 0

        # Iterates through every line except for the last in the trace file
        for line in lines[:-1]:
            # Splits the line by white spaces and ignores the first two tokens
            tokens = line.split()[2:]
            instruction = tokens[0]
            # Skips the return instructions
            if instruction in RiscvParser.ret_instructions:
                continue
            operands = tokens[1].split(",")
            
            # Creates a new vertex as per the instruction
            new_vertex = \
                self._instruction_to_vertex(vertex_id, instruction, operands)

            # Creates dependency edges
            if (not self.only_mem_acc) or \
                (self.only_mem_acc and new_vertex.is_mem_acc):
                for dep in new_vertex.dependencies:
                    source = curr_vertex.get(dep)
                    if source is not None:
                        eDag.add_edge(source, new_vertex)
                
                if new_vertex.target is not None:
                    curr_vertex[new_vertex.target] = new_vertex

                eDag.add_vertex(new_vertex)

                vertex_id += 1
        riscv_trace.close()

        if self.remove_single_vertices:
            eDag.remove_single_vertices()
        return eDag