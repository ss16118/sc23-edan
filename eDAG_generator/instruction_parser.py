from abc import ABC, abstractmethod
from typing import List, Optional
from eDAG import Vertex

class InstructionParser(ABC):
    """
    A generic abstract class that represents 
    """
    @abstractmethod
    def is_ret_instruction(self, instruction: str) -> bool:
        pass

    @abstractmethod
    def generate_vertex(self, id: int, instruction: str,
                        operands: List[str],
                        cpu: Optional[int] = None,
                        insn_addr: Optional[str] = None) -> Vertex:
        pass