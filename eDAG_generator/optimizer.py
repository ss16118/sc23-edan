from abc import ABC, abstractmethod
from eDAG import EDag


class SubgraphOptimizer(ABC):
    """
    An abstract class for subgraph optimizers. All objects
    that inherit from this class need to implement the `optimize()`
    function.
    """
    @abstractmethod
    def optimize(self, subgraph: EDag, verbose: bool = True) -> None:
        pass