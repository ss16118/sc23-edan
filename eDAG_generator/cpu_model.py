from typing import Dict, Optional, List, Set, Tuple
from eDAG import Vertex


class CPUModel:
    """
    A class representation of simplified CPU model that assigns estimated
    number of CPU cycles to a single vertex in the eDAG.
    """
    def __init__(self, insn_cycles: List[Tuple[Set[str], float]],
                 use_cache_model: bool = False,
                 frequency: int = 3.6 * 10 ** 9,
                 cache_hit_cycles: int = 4) -> None:
        """
        @param insn_cycles: A list of tuples that maps a set of instructions to
        their corresponding estimated compute cycles.
        @param frequency: The frequency of the CPU, whose reciprocal, when
        multiplied with the total number of cycles executed, will be used
        to estimate the time.
        @param use_cache_model: If set to True, memory load instructions
        that are cache hits will not be considered.
        @param cache_hit_cycles: The number of clock cycles that will take
        for a memory load to complete if it is cache hit. Will only be
        considered when `use_cache_model` is set to True.
        TODO Needs to consider multi-level cache
        """
        self.insn_cycles = insn_cycles
        self.frequency = frequency
        self.use_cache_model = use_cache_model
        self.cache_hit_cycles = cache_hit_cycles
    
    def get_op_cycles(self, vertex: Vertex) -> float:
        """
        Retrieves the number of cycles that will take the given vertex
        to complete based on the information specified in `insn_cycles`.
        If the opcode is not contained in `insn_cycles`, an exception
        will be raised.
        """
        # TODO Find a more accurate CPU model
        # First checks if the vertex is a cache hit
        if vertex.cache_hit and self.use_cache_model:
            return self.cache_hit_cycles
        
        opcode = vertex.opcode
        for opcode_set, op_cycles in self.insn_cycles:
            if opcode in opcode_set:
                return op_cycles
        raise ValueError(f"[ERROR] Opcode '{opcode}' has unknown cycles")

