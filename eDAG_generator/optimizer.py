from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from eDAG import EDag
from utils import *


class Optimizer(ABC):
    """
    An abstract class that needs to be inherited by all
    the ISA-specific optimizer objects.
    """
    def __init__(self, eDag: EDag) -> None:
        self.eDag = eDag

    @abstractmethod
    def optimize(self, eDag: EDag, verbose: bool) -> None:
        pass


class RiscvOptimizer(Optimizer):
    """
    An optimizer implemented specifically for eDAGs generated
    from RISC-V instruction traces.
    """
    def __init__(self, eDag: EDag) -> None:
        super().__init__(eDag)

    def __identify_loop_idx_increment(self, topo_sorted: array) \
        -> List[List[int]]:
        """
        A helper function that, given a list containing the IDs of
        the vertices in the eDAG in topological order, returns
        a list of lists containing the vertices that are identified as
        loop index increment chains.
        """

        # A set of opcodes that might indicate that a loop index
        # is being incremented
        loop_idx_inc_ops = { "addi", "addiw" }

        chains = []
        curr_chain = []
        visited = set()

        for v_id in topo_sorted:
            # Skips over vertices that have been visited
            if v_id in visited:
                continue
            
            vertex = self.eDag.id_to_vertex[v_id]
            if vertex.opcode in loop_idx_inc_ops:
                # Identifies the first vertex that starts
                # the chain (i.e. `origin`)
                curr_chain.append(v_id)
                origin = v_id
                curr = v_id
                # Checks all the targets of the vertex.
                # At least one of them needs to satisfy the
                # conditions of being in a chain
                while True:
                    break_chain = True
                    in_ids, out_ids = self.eDag.adj_list[curr]
                    # To satisfy the condition of being in a
                    # chain, the vertex must be have the exact
                    # same instruction as the origin and must
                    # have in-degree 1
                    if len(in_ids) == 1 or curr == origin:
                        for out_id in out_ids:
                            out_v = self.eDag.id_to_vertex[out_id]
                            if vertex.has_same_insn(out_v):
                                curr = out_id
                                break_chain = False
                    
                    if break_chain:
                        if len(curr_chain) > 1:
                            chains.append(curr_chain)
                        curr_chain = []
                        break
                        
                    curr_chain.append(curr)
                    visited.add(curr)

        return chains

    def optimize(self, verbose: bool) -> None:
        """
        Optimizes the given eDAG with the following heuristics:
        1. Tries to loop counter increment pattern, where a chain
        of vertices all add the same constant to a register. Assumes
        that optimization was turned on while it was compiled.
        
        After identifying this pattern, the first vertex in the chain
        will be chosen as `origin`, and the value of the immediate
        operand will be propagated to the other vertices down the 
        chain and they will all depend on `origin`. For example:

        (v1: addi a1,a1,4) -> (v2: addi a1,a1,4) -> (v3: addi a1,a1,4) =>
        {
            (v1: addi a1,a1,4) -> (v2: addi a1,a1,8),
            (v1: addi a1,a1,4) -> (v3: addi a1,a1,12)
        }

        ======= Caution =======
        Note that this is a irreversible procedure, once
        optimized, the eDAG cannot be reverted back to its
        original state without regenerating it entirely.
        """

        topo_sorted = self.eDag.topological_sort()

        # =============================
        # Identifies the loop index increment pattern
        # =============================
        idx_inc_chains = self.__identify_loop_idx_increment(topo_sorted)
        if verbose:
            print(f"[INFO] Identified {len(idx_inc_chains)} loop index "
                  "chains that can be simplified")

        for i, chain in enumerate(idx_inc_chains):
            if verbose:
                print(f"[INFO] Optimizing chain {i + 1} with {len(chain)} vertices")
            # The first vertex of the chain is the origin
            origin_id = chain[0]
            prev_id = origin_id
            imm_val = self.eDag.id_to_vertex[origin_id].imm_val
            curr_val = imm_val

            for curr_id in chain[1:]:
                # Removes the dependency between the current vertex
                # and its direct predecessor
                self.eDag.remove_edge(prev_id, curr_id)
                # Changes the immediate value
                curr = self.eDag.id_to_vertex[curr_id]
                curr_val += imm_val
                curr.imm_val = curr_val
                # Changes the operand itself
                # FIXME Can probably be optional to save some compute
                # operands = list(curr.operands)
                # operands[-1] = str(curr_val)
                curr.operands[-1] = str(curr_val)
                # Adds a dependency between the current vertex
                # and the origin
                self.eDag.add_edge(origin_id, curr_id)
                prev_id = curr_id

class EDagOptimizer(object):
    """
    A class representation for eDAG optimizers.
    """
    isa_to_optimizer: Dict[ISA, Optimizer] = {
        ISA.RISC_V: RiscvOptimizer
    }
    def __init__(self, eDag: EDag, isa: ISA) -> None:
        """
        The target of optimization will be the given EDag object.
        @param eDag: An EDag object that is to be optimized.
        @param isa: The ISA based on which the given eDAG was generated.
        """
        self.optimizer = EDagOptimizer.isa_to_optimizer.get(isa)
        if self.optimizer is None:
            raise ValueError(f"[ERROR] ISA {isa.name} is not yet supported")
        self.optimizer = self.optimizer(eDag)

    def optimize(self, verbose: bool = True) -> None:
        """
        Tries to optimize the work and depth of the eDAG based
        on predefined heuristics. Note that modifications to the
        eDAG will be done in place to save memory. A wrapper function
        around `Optimizer.optimize()`.

        @param verbose: If True, will print detailed information
        to stdout.
        """
        self.optimizer.optimize(verbose)