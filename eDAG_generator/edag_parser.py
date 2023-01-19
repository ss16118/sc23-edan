import os
from typing import List, Optional
from abc import ABC, abstractmethod
from eDAG import EDag


class EDagParser(ABC):
    """
    A generic abstract class that represents objects whose primary function 
    is to parse the assembly instructions in the given trace file to a 
    execution DAG where the vertices represent operations while the edges 
    denote the dependencies between the operations.
    """
    def __init__(self, trace_file: str, only_mem_acc: bool,
                remove_single_vertices: bool = True) -> None:
        """
        Takes in the path to the trace file as input for the constructor.
        If `only_mem_acc` is set to True, only memory access related
        instructions will be included in the parsed execution DAG.
        If `remove_single_vertices` is set to True, vertices that have no
        dependencies will be removed from the eDAG.
        """
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.only_mem_acc = only_mem_acc
        self.remove_single_vertices = remove_single_vertices

    @abstractmethod
    def parse(self) -> EDag:
        pass
