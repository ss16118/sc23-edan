import os
from tqdm import tqdm
from typing import List, Optional, Dict
from enum import Enum, auto
from time import time
from collections import defaultdict
from eDAG import EDag, Vertex, OpType
from instruction_parser import InstructionParser
from riscv_parser import RiscvParser
from cache_model import CacheModel
from cpu_model import CPUModel
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
    def __init__(self, trace_file: str, isa: ISA, 
                only_mem_acc: bool, sanitize: bool = False,
                cache_model: Optional[CacheModel] = None,
                cpu_model: Optional[CPUModel] = None)-> None:
        """
        @param trace_file: path to the trace file as input for the constructor.
        @isa: the ISA of the assembly instructions contained in the trace file.
        @param only_mem_acc: If set to True, only memory access related
        instructions will be included in the parsed execution DAG.
        @param sanitize: If set to True, only the most essential vertices
        and dependencies will be kept, while control dependencies and
        non-relevant vertices unrelated to the core of computation will be
        removed. This is used to for theoretical analysis of work and depth
        of parallel algorithms.
        @param cache_model: Specifies the cache model to use while generating
        the eDAG.
        @param cpu_model: A CPU model that will be leveraged to estimate
        the number of computation cycles needed by each vertex.
        """
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.only_mem_acc = only_mem_acc

        # eDAG sanitizer initialization
        self.sanitize = sanitize
        self.sanitizer = None
        if sanitize:
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
        # CPU model initialization
        self.cpu_model = cpu_model

    def generate(self) -> EDag:
        """
        Converts the given instruction trace into a single execution DAG
        """
        trace = open(self.trace_file, "r")

        eDag = EDag()
        # A dictionary that maps a CPU index and a register or
        # memory location, represented as a string to the most recent
        # vertex which updated its value
        curr_vertex: Dict[int, Dict[str, Vertex]] = defaultdict(dict)

        # A number that is used to uniquely identify each vertex
        # increments after a vertex is added to the eDAG
        vertex_id = 0

        # Iterates through every line in the trace file
        prev_time = time()
        for line in trace:
            if vertex_id % 1000000 == 0 and vertex_id != 0:
                curr_time = time()
                print(f"[INFO] Progress: {vertex_id} [{int(1000000 / (curr_time - prev_time))} iter/s]")
                prev_time = curr_time
            # Strips the newline character on the right of a line
            line = line.strip()
            parsed_line = self.parser.parse_line(line)
            
            if parsed_line is None:
                continue
            # Creates a new vertex as per the instruction
            new_vertex: Vertex = \
                self.parser.generate_vertex(id=vertex_id, **parsed_line)
            
            cpu_id = new_vertex.cpu

            # If a cache model is used
            if self.cache_model is not None and \
                new_vertex.op_type == OpType.LOAD_MEM:
                cache_hit = self.cache_model.find(new_vertex.data_addr)

                new_vertex.cache_hit = cache_hit
                if cache_hit:
                    # If the data access is a cache hit, reduces the amount
                    # of data movement to 0 since it does not need to
                    # access the main memory
                    new_vertex.data_size = 0
            
            # If a CPU model is used
            if self.cpu_model is not None:
                new_vertex.cycles = self.cpu_model.get_op_cycles(new_vertex)

            is_critical = True
            # if self.sanitize:
            #     # Only critical vertices are kept
            #     is_critical = self.sanitizer.is_critical_vertex(new_vertex)

            if is_critical:
                eDag.add_vertex(new_vertex)

                # if not self.sanitize and \
                #     new_vertex.target is not None:
                #     # If `simplified` is True, only true dependencies
                #     # will be kept
                #     new_vertex.dependencies.add(new_vertex.target)

                # Creates dependency edges
                for dep in new_vertex.dependencies:
                    # source = curr_vertex.get(dep)
                    source = curr_vertex[cpu_id].get(dep)
                    if source is not None:
                        eDag.add_edge(source, new_vertex)
                
                # if prev_vertex and prev_vertex.op_type == OpType.BRANCH:
                #     # If the previous vertex contains branch/jump
                #     # instruction, adds a dependency between it
                #     # and the current vertex
                #     eDag.add_edge(prev_vertex, new_vertex)

                if new_vertex.target is not None:
                    curr_vertex[cpu_id][new_vertex.target] = new_vertex
            
            vertex_id += 1
        trace.close()

        # if self.sanitize:
        #     self.sanitizer.sanitize_edag(eDag)

        if self.only_mem_acc:
            eDag.filter_vertices(lambda v: v.is_mem_acc)

        return eDag
