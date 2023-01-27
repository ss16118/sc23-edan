from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict

class Cache(ABC):
    """
    An abstract class that represents specific caching strategy implementations
    """
    def __init__(self, size: int) -> None:
        self.size = size

    @abstractmethod
    def get(self, tag: int) -> bool:
        pass
    
    @abstractmethod
    def put(self, tag: int, val: int) -> None:
        pass
    
    @abstractmethod
    def to_list(self) -> List[Optional[int]]:
        pass


class LRUCache(Cache):
    """
    A class representation of LRU caching strategy based on OrderedDict.
    Implementation from:
    https://www.javatpoint.com/lru-cache-in-python
    """
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.lru_cache = OrderedDict()

    def get(self, tag: int) -> bool:
        """
        Searches for a tag in the LRU cache, if it is found, returns True,
        False otherwise.
        """
        if tag not in self.lru_cache:
            return False
        
        val = self.lru_cache.pop(tag)
        self.lru_cache[tag] = val
        return True
        

    def put(self, tag: int, val: int) -> None:
        """
        Inserts a new cache line into the cache. Invokes the LRU eviction
        strategy if the cache is full.
        """
        if tag in self.lru_cache:
            # If the tag already exists
            self.lru_cache.pop(tag)
        
        if len(self.lru_cache) == self.size:
            # If the cache is full evicts the last item
            self.lru_cache.popitem(tag)
        self.lru_cache[tag] = val
    
    def to_list(self) -> List[Optional[int]]:
        """
        Converts the content in the cache as a list of integers representing
        the tags of the cache line. The unfilled entries in the cache are
        denoted by None.
        """
        repr = []
        for tag in self.lru_cache.keys():
            repr.append(tag)
        for _ in range(self.size - len(self.lru_cache)):
            repr.append(None)
        return repr

