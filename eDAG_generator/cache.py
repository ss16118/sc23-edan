from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from collections import OrderedDict

class Cache(ABC):
    """
    An abstract class that represents specific caching strategy implementations
    """
    def __init__(self, size: int) -> None:
        self.size = size

    @abstractmethod
    def get(self, key: int) -> bool:
        pass
    
    @abstractmethod
    def put(self, key: int, val: int) -> None:
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