from typing import Optional
from abc import ABC, abstractmethod
from eDAG import OpType, EDag, Vertex


class EDagSanitizer(ABC):
    """
    An abstract class that represents eDAG sanitizers that
    is invoked in the case where only critical sections
    of the eDAG are to be kept.
    """
    @abstractmethod
    def is_critical_vertex(self, vertex: Vertex) -> bool:
        pass

    @abstractmethod
    def sanitize_edag(self, eDag: EDag) -> None:
        pass


class RiscvEDagSanitizer(EDagSanitizer):
    """
    An object that is used specifically for removing the non-critical
    sub-graphs in a eDAG generated parsing RISC-V trace.
    """
    non_critical_vertex_types = {
        OpType.LOAD_IMM,
        # OpType.STORE_MEM,
        OpType.MOVE,
        OpType.BRANCH
    }

    def __init__(self) -> None:
        super().__init__()


    def is_critical_vertex(self, vertex: Vertex) -> bool:
        """
        Determines whether the given vertex is critical by checking
        its `op_type`. Returns True, if it belongs to a predefined set of critical op types, False otherwise.
        """
        return vertex.op_type not in RiscvEDagSanitizer.non_critical_vertex_types

    def sanitize_edag(self, eDag: EDag) -> None:
        # Utilizes a set of heuristic rules to remove the
        # remaining non-critical vertices in the given eDAG
        # Heuristics:
        # 1. Removes all predecessors of LOAD_MEM operators
        # 2. Removes subgraphs of only arithmetic operations as they
        # are likely to be control vertices

        # Heuristic 1
        # All predecessors of LOAD_MEM vertices will be removed
        def cond(vertex: Vertex):
            for successor in eDag.adj_list[vertex][EDag._out]:
                if successor.op_type == OpType.LOAD_MEM:
                    return False
            return True
        eDag.filter_vertices(cond)
        # Heuristic 2
        # Remove a chain of identical arithmetic operations
        arithmetic_ops = { OpType.ARITHMETIC_IMM, OpType.ARITHMETIC }
        
        def remove(subgraph: EDag) -> bool:
            """
            A helper function that identifies whether the given subgraph
            contains only arithmetic operations. If that is the case,
            the subgraph should likely be removed, and True is returned.
            """
            for vertex in subgraph.vertices:
                if vertex.op_type not in arithmetic_ops:
                    return False
            return True
        # Constructs a list of disjoint subgraphs
        eDag.split_disjoint_subgraphs()
        for subgraph in eDag.disjoint_subgraphs:
            # Removes the entire subgraph that satisfies the pattern
            if remove(subgraph):
                eDag.remove_subgraph(subgraph)
