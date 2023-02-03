from optimizer_fsm import *
from optimizer import SubgraphOptimizer
from eDAG import EDag


class RiscvSubgraphOptimizer(SubgraphOptimizer):
    """
    A subgraph optimizer that targets eDAGs generated from RISC-V
    assembly traces.
    """
    def __init__(self) -> None:
        super().__init__()
        self.reduction_fsm = ReductionFSM()

    def optimize(self, subgraph: EDag, verbose: bool = True) -> None:
        """
        Optimizes the given subgraph based on a set of pre-defined patterns
        """
        # Converts the given subgraph into vertices
        vertices = subgraph.sorted_vertices
        self.reduction_fsm.reset()
        prev_depth = subgraph.get_depth()
        optimization_count = 0
        for vertex in vertices:
            pattern_found = self.reduction_fsm.step(vertex)
            if pattern_found:
                optimization_count += 1
                self.reduction_fsm.optimize(subgraph)
                self.reduction_fsm.reset()
                if verbose:
                    print("[INFO] Found reduction pattern in subgraph")
                
        if self.reduction_fsm.pattern_found():
            optimization_count += 1
            self.reduction_fsm.optimize(subgraph)
            if verbose:
                print("[INFO] Found reduction pattern in subgraph")
        if verbose:
            curr_depth = subgraph.get_depth()
            print(f"[INFO] {optimization_count} optimizations found: "
                f"depth reduced from {prev_depth} to {curr_depth}")
