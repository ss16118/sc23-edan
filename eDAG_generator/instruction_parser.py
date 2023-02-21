from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from eDAG import Vertex

class InstructionParser(ABC):
    """
    A generic abstract class that represents 
    """
    @abstractmethod
    def is_ret_instruction(self, instruction: str) -> bool:
        pass
    
    @abstractmethod
    def get_load_data_addr(self, line: str) -> Optional[str]:
        pass

    @abstractmethod
    def parse_line(self, line: str) -> Dict:
        pass

    @abstractmethod
    def generate_vertex(self, id: int, instruction: str,
                        operands: List[str],
                        cpu: Optional[int] = None,
                        insn_addr: Optional[str] = None,
                        data_addr: Optional[str] = None) -> Vertex:
        pass