from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Callable
from eDAG import EDag, Vertex, OpType


class OptimizerState:
    """
    A class representation of a single state in a finite state machine (FSM)
    used in the optimizer.
    """
    def __init__(self, name: str) -> None:
        """
        @param name: The name of the state represented as a single string.
        Note this is also used as the ID of the state. Hence, two unique
        states need to have two different names.
        """
        self.name = name
        # Contains a list of vertices in an eDAG that have caused
        # transition to this state
        self.vertices = []
    
    def reset(self) -> None:
        """
        Resets the list of vertices collected
        """
        self.vertices.clear()
    
    def __str__(self) -> str:
        return f"{self.name}"

    def __hash__(self) -> int:
        return hash((self.name))

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name

class OptimizerFSM(ABC):
    """
    An abstract class representation of a finite state machine (FSM)
    used in the subgraph optimizer to recognize patterns.
    """
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the current state to the starting state. Clears
        all the collected vertices.
        """
        pass
    
    @abstractmethod
    def step(self, vertex: Vertex) -> bool:
        """
        Given a vertex in an eDAG, transitions to the next state.
        Also collects the given vertex and stores them in the states.
        Returns a boolean indicating whether a pattern has been identified.
        """
        pass

    @abstractmethod
    def pattern_found(self) -> bool:
        """
        Based on the current state of the FSM, determine whether a valid
        pattern has been found.
        """
        pass
    
    @abstractmethod
    def optimize(self, eDag: EDag) -> None:
        """
        Optimizes the given eDAG in place as per the pattern discovered
        by the state machine.
        """
        pass


class ReductionFSM(OptimizerFSM):
    """
    An OptimizerFSM for recognizing the reduction pattern.
    The FSM should be ISA agnostic.
    An example of the reduction pattern and its
    optimization are as follows:
    LD      LD      LD      LD          LD      LD      LD      LD
    |       |       |       |   ->       \      /        \      /
    ADD --- ADD --- ADD --- ADD            ADD             ADD
                                             \             /
                                              \           /
                                               --- ADD ---
    """
    def __init__(self) -> None:
        super().__init__()
        # All states in the FSM:
        self.start = OptimizerState("START")
        self.load_1 = OptimizerState("LD1")
        self.load_2 = OptimizerState("LD2")
        self.arithmetic = OptimizerState("ARITHMETIC")
        self.end = OptimizerState("END")
        self.final_states = { self.end }
        self.states = { 
            self.start, self.load_1, self.load_2, self.arithmetic, self.end
        }
        # Sets the default starting state
        self.curr_state = self.start
        self.prev_state = None
    
    def reset(self) -> None:
        self.curr_state = self.start
        self.prev_state = None
        for state in self.states:
            state.reset()

    def step(self, vertex: Vertex) -> bool:
        """
        Transitions to the next state based on the current state
        and the given eDAG vertex.
        Returns True if a valid pattern is identified.
        """
        load_ops = { OpType.LOAD_MEM }
        next = None
        if self.curr_state == self.start:
            # Transition handling for START
            if vertex.op_type in load_ops:
                next = self.load_1
            else:
                next = self.start
        
        elif self.curr_state == self.load_1:
            # Transition handling for LD1
            if vertex.op_type in load_ops:
                next = self.load_2
            elif vertex.is_comm_assoc:
                next = self.arithmetic
            else:
                next = self.start

        elif self.curr_state == self.load_2:
            # Transition handling for LD2
            if vertex.is_comm_assoc:
                next = self.arithmetic
            else:
                next = self.start
        
        elif self.curr_state == self.arithmetic:
            # Transition handling for ARITHMETIC
            if vertex.op_type in load_ops:
                next = self.load_2
            else:
                # Can only proceed to the end state when LD2 has collected
                # more than two vertices, meaning that more than two
                # arithmetic operations were executed
                if len(self.load_2.vertices) > 1:
                    next = self.end
                else:
                    next = self.start
        else:
            raise ValueError("[ERROR] Should not be here")
        
        self.prev_state = self.curr_state
        self.curr_state = next
        self.curr_state.vertices.append(vertex)
        # Resets the FSM if it has returned to START
    
        if next == self.start:
            self.reset()
            
        return self.curr_state in self.final_states
    
    def pattern_found(self) -> bool:
        return self.curr_state == self.arithmetic or \
            self.curr_state == self.end

    def optimize(self, eDag: EDag) -> None:
        """
        Restructures the sequential eDAG so that it has a depth of O(log(n))
        as opposed to O(n).
        """
        def restructure(in_vertices: List[Vertex],
                        out_vertices: List[Vertex],
                        leftover: Optional[Vertex] = None) -> None:
            """
            A helper function that recursively assigns every two in-vertices
            to an out-vertex.
            """
            # Base case:
            # If both `in_vertices` and `out_vertices` are empty
            if len(in_vertices) == 0 and len(in_vertices) == 0:
                return

            if len(in_vertices) == 1 and \
                leftover is not None:
                in_vertices.append(leftover)
                leftover = None
            
            # The out_vertices for next iteration of recursion
            next_in = []
            out_set = set(out_vertices)
            while len(in_vertices) > 1:
                v1 = in_vertices.pop(0)
                v2 = in_vertices.pop(0)
                # Removes all edges to any of the out vertices
                eDag.adj_list[v1][EDag._out] = \
                    eDag.adj_list[v1][EDag._out].difference(out_set)
                eDag.adj_list[v2][EDag._out] = \
                    eDag.adj_list[v2][EDag._out].difference(out_set)
                
                assert(len(out_vertices) > 0)
                out = out_vertices.pop(0)
                eDag.adj_list[out][EDag._out] = \
                    eDag.adj_list[out][EDag._out].difference(out_set)
                next_in.append(out)
                # Adds an dependency from v1 and v2 to one of the out-vertices
                eDag.add_edge(v1, out)
                eDag.add_edge(v2, out)
            if len(in_vertices) == 1 and leftover is None:
                leftover = in_vertices.pop(0)
            assert(len(in_vertices) == 0)

            restructure(next_in, out_vertices, leftover)

        load_vertices = self.load_1.vertices + self.load_2.vertices
        arithmetic_vertices = self.arithmetic.vertices
        restructure(load_vertices, arithmetic_vertices)
        