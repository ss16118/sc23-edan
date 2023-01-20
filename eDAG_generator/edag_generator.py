import os
from typing import List, Optional, Dict
from enum import Enum, auto
from eDAG import EDag, Vertex
from instruction_parser import InstructionParser
from riscv_parser import RiscvParser


class ISA(Enum):
    RISC_V = auto()
    ARM = auto()
    X86 = auto()


class EDagGenerator:
    """
    A class whose primary function is to parse the assembly instructions in the 
    given trace file to a execution DAG where the vertices represent operations 
    while the edges denote the dependencies between the operations.
    """
    # Maps ISA to its corresponding `InstructionParser` object
    isa_to_parser: Dict[ISA, InstructionParser] = {
        ISA.RISC_V : RiscvParser
    }

    # =========== Constructor ===========
    def __init__(self, trace_file: str, isa: ISA, only_mem_acc: bool,
                remove_single_vertices: bool = True) -> None:
        """
        @param trace_file: path to the trace file as input for the constructor.
        @isa: the ISA of the assembly instructions contained in the trace file.
        @param only_mem_acc: If set to True, only memory access related
        instructions will be included in the parsed execution DAG.
        @param remove_single_vertices: If set to True, vertices that have no
        dependencies will be removed from the eDAG.
        """
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.only_mem_acc = only_mem_acc
        self.remove_single_vertices = remove_single_vertices
        self.parser = EDagGenerator.isa_to_parser.get(isa)
        if self.parser is None:
            raise ValueError(f"[ERROR] ISA {isa.name} is not yet supported")
        else:
            self.parser = self.parser()

    def generate(self) -> EDag:
        """
        Converts the given instruction trace into a single execution DAG
        """
        trace = open(self.trace_file, "r")
        lines = trace.readlines()

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
                self.parser.generate_vertex(vertex_id, instruction, operands)

            # Creates dependency edges
            if (not self.only_mem_acc) or \
                (self.only_mem_acc and new_vertex.is_mem_acc):

                if self.only_mem_acc and new_vertex.is_mem_acc:
                    pass

                for dep in new_vertex.dependencies:
                    source = curr_vertex.get(dep)
                    if source is not None:
                        eDag.add_edge(source, new_vertex)
                
                if new_vertex.target is not None:
                    curr_vertex[new_vertex.target] = new_vertex

                eDag.add_vertex(new_vertex)

            vertex_id += 1
        trace.close()

        if self.remove_single_vertices:
            eDag.remove_single_vertices()
        return eDag
