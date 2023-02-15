import os
from tqdm import tqdm
from typing import List, Optional, Dict
from enum import Enum, auto
from eDAG import EDag, Vertex, OpType
from instruction_parser import InstructionParser
from riscv_parser import RiscvParser
from cache_model import CacheModel
from edag_sanitizer import *

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

    # Maps ISA to its corresponding `EDagSanitizer` object
    isa_to_sanitizer: Dict[ISA, EDagSanitizer] = {
        ISA.RISC_V : RiscvEDagSanitizer
    }
    

    # =========== Constructor ===========
    def __init__(self, trace_file: str, isa: ISA, only_mem_acc: bool,
                remove_single_vertices: bool = True, simplified: bool = False,
                cache_model: Optional[CacheModel] = None)-> None:
        """
        @param trace_file: path to the trace file as input for the constructor.
        @isa: the ISA of the assembly instructions contained in the trace file.
        @param only_mem_acc: If set to True, only memory access related
        instructions will be included in the parsed execution DAG.
        @param remove_single_vertices: If set to True, vertices that have no
        dependencies will be removed from the eDAG.
        @param simplified: If set to True, only the most essential vertices
        and dependencies will be kept, while control dependencies and
        non-relevant vertices unrelated to the core of computation will be
        removed. This is used to for theoretical analysis of work and depth
        of parallel algorithms.
        @param cache_model: Specifies the cache model to use while generating
        the eDAG.
        """
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.only_mem_acc = only_mem_acc
        self.remove_single_vertices = remove_single_vertices

        # eDAG sanitizer initialization
        self.simplified = simplified
        self.sanitizer = None
        if self.simplified:
            self.sanitizer = EDagGenerator.isa_to_sanitizer.get(isa)
            if self.sanitizer is None:
                raise ValueError(f"[ERROR] ISA {isa.name} sanitizer is not yet supported")
            else:
                self.sanitizer = self.sanitizer()
        
        # eDAG parser initialization
        self.parser = EDagGenerator.isa_to_parser.get(isa)
        if self.parser is None:
            raise ValueError(f"[ERROR] ISA {isa.name} is not yet supported")
        else:
            self.parser = self.parser()
        
        # Cache model initialization
        self.cache_model = cache_model

    def generate(self) -> EDag:
        """
        Converts the given instruction trace into a single execution DAG
        """
        trace = open(self.trace_file, "r")

        eDag = EDag()
        # A dictionary that maps a register or memory location represented
        # as a string to the most recent vertex which updated its value
        curr_vertex: Dict[str, Vertex] = {}
        # A number that is used to uniquely identify each vertex
        # increments after a vertex is added to the eDAG
        vertex_id = 0

        # A variable that keeps track of the last generated vertex
        prev_vertex = None

        # Iterates through every line in the trace file
        for line in tqdm(trace):
            parsed_line = self.parser.parse_line(line)

            if parsed_line is None:
                continue
            
            # Creates a new vertex as per the instruction
            new_vertex: Vertex = \
                self.parser.generate_vertex(id=vertex_id, **parsed_line)

            if new_vertex.is_mem_acc:
                # new_vertex.data_addr = addr
                addr = new_vertex.data_addr
                # Keeps track of the writes to memory addresses
                if new_vertex.op_type == OpType.STORE_MEM:
                    curr_vertex[addr] = new_vertex
                elif new_vertex.op_type == OpType.LOAD_MEM:
                    dep = curr_vertex.get(addr)
                    if dep is not None:
                        # Creates a dependency between the vertex that 
                        # previously wrote to the same address and the 
                        # new vertex
                        new_vertex.dependencies.add(dep)

            # If a cache model is used
            if self.cache_model is not None and \
                new_vertex.op_type == OpType.LOAD_MEM:
                new_vertex.cache_hit = \
                    self.cache_model.find(new_vertex.data_addr)

            is_critical = True
            # if self.simplified:
                # Only critical vertices are kept
                # is_critical = self.sanitizer.is_critical_vertex(new_vertex)

            if is_critical:
                eDag.add_vertex(new_vertex)

                if not self.simplified and \
                    new_vertex.target is not None:
                    # If `simplified` is True, only true dependencies
                    # will be kept
                    new_vertex.dependencies.add(new_vertex.target)

                # Creates dependency edges
                for dep in new_vertex.dependencies:
                    source = curr_vertex.get(dep)
                    if source is not None:
                        eDag.add_edge(source, new_vertex)
                
                if prev_vertex and prev_vertex.op_type == OpType.BRANCH:
                    # If the previous vertex contains branch/jump
                    # instruction, adds a dependency between it
                    # and the current vertex
                    eDag.add_edge(prev_vertex, new_vertex)

                if new_vertex.target is not None:
                    curr_vertex[new_vertex.target] = new_vertex
            
            vertex_id += 1
            prev_vertex = new_vertex
        trace.close()

        # if self.simplified:
        #     self.sanitizer.sanitize_edag(eDag)

        if self.only_mem_acc:
            eDag.filter_vertices(lambda v: v.is_mem_acc)

        if self.remove_single_vertices:
            eDag.remove_single_vertices()
        return eDag
