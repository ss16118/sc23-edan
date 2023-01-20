from abc import ABC, abstractmethod
from typing import List
from eDAG import Vertex

class InstructionParser(ABC):
    """
    A generic abstract class that represents 
    """
    @abstractmethod
    def generate_vertex(self, id: int, instruction: str,
                        operands: List[str]) -> Vertex:
        pass