from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from collections import OrderedDict
from enum import Enum, auto
from texttable import Texttable
from math import log
from cache import *


class EvictionStrategy(Enum):
    RANDOM = auto()
    FIFO = auto()
    LRU = auto()


class CacheModel(ABC):
    """
    A generic abstract class that represents cache
    """
    def __init__(self, cache_line_size: int,
                addr_len: int,
                strategy: EvictionStrategy) -> None:
        """
        @param cache_line_size: the size of a single line of cache in bytes.
        @param addr_len: the number of bits in an address, indicating how
        large the address space is.
        @param eviction_strategy: the eviction strategy used. Possible options
        are { `RANDOM`, `FIFO`, `LRU` }.
        """
        self.cache_line_size = cache_line_size
        self.addr_len = addr_len
        self.strategy = strategy

    @abstractmethod
    def find(self, addr: int) -> bool:
        """
        Checks if the given address is in the cache. Returns a boolean that
        indicates whether it is a cache hit or cache miss (True for
        cache hit and False for cache miss). If the address is not found in
        cache, performs operations which insert the corresponding cache line
        into the cache.
        """
        pass

    @abstractmethod
    def print_state(self) -> None:
        """
        Displays the cache state as a string.
        """
        pass


class SingleLevelSetAssociativeCache(CacheModel):
    """
    A class representation of single level set associative cache.
    FIXME: Now this cache model simulates the behavior of a write-through
    cache, where the store operations are write back to memory. Since
    this triggers a main memory access, it counts as a cache miss.
    Later on, maybe an implementation of write-back cache.
    """
    # Maps eviction strategy to specific implementations of cache
    strategy_to_cache: Dict[EvictionStrategy, Cache] = {
        EvictionStrategy.LRU: LRUCache
    }

    def __init__(self, cache_line_size: int = 8,
                cache_size: int = 64,
                addr_len: int = 64,
                k: int = 2,
                strategy: EvictionStrategy = EvictionStrategy.LRU) -> None:
        """
        @param cache_line_size: Number of bytes in a cache line.
        @param cache_size: Total size of the cache in bytes.
        @param addr_len: Number of bits of an address.
        @param k: Number of cache lines in a set. k-way associative cache.
        @param strategy: Cache eviction strategy.
        """
        super().__init__(cache_line_size, addr_len, strategy)
        # Ensures that the cache size is divisible by the cache line size
        assert(cache_size % cache_line_size == 0)
        self.num_cache_lines = cache_size // cache_line_size
        # Ensures that the number of cache lines in a set is a whole number
        assert(self.num_cache_lines % k == 0)
        self.num_sets = self.num_cache_lines // k
        self.k = k
        self.cache_size = cache_size
        """
        Representation of a single cache line
        |------ Tag -----|--- Set ---|- Offset -|
        +---------------------------------------+
        |  w - log(M/B)  | log(M/kB) |  log(B)  |
        +---------------------------------------+
        w: number of bits in an address
        M: cache size
        k: k-way associative
        B: cache line size / block size
        """
        # Number of bits in offset
        self.offset_bit_size = int(log(cache_line_size, 2))
        # Number of bits in that is used to represent the set
        self.set_bit_size = int(log(self.num_sets, 2))
        self.tag_bit_size = \
            addr_len - self.offset_bit_size - self.set_bit_size

        cache_impl = \
            SingleLevelSetAssociativeCache.strategy_to_cache.get(strategy)
        if cache_impl is None:
            raise ValueError(f"[ERROR] Eviction strategy {strategy.name} is not yet supported")
        # The actual cache is represented by an array of Cache objects
        self.cache: List[Cache] = [cache_impl(k) for _ in range(self.num_sets)]
        self.set_bit_mask = self._get_set_bit_mask()
        self.tag_bit_mask = self._get_tag_bit_mask()

    def _get_set_bit_mask(self) -> int:
        """
        A helper function that computes the bit-mask that is
        used to retrieve the set number in an address.
        """
        set_bit_mask = 0
        # Pushes a number of ones to bit-mask, and this number should
        # equal to the number of set bits
        for _ in range(self.set_bit_size):
            set_bit_mask <<= 1
            set_bit_mask |= 1
        # Pushes a number of zeroes to bit-mask, and this number should
        # equal to the number of offset bits
        for _ in range(self.offset_bit_size):
            set_bit_mask <<= 1
        return set_bit_mask

    def _get_tag_bit_mask(self) -> int:
        """
        A helper function that computes the bit-mask that is used to
        retrieve the tag number in an address.
        """
        tag_bit_mask = 0
        # Pushes a number of ones to bit-mask, which should be equal
        # to the number of tag bits
        for _ in range(self.tag_bit_size):
            tag_bit_mask <<= 1
            tag_bit_mask |= 1
        # Pushes a number of zeroes to bit-mask, which should equal
        # to the number of remaining bits
        for _ in range(self.set_bit_size + self.offset_bit_size):
            tag_bit_mask <<= 1
        return tag_bit_mask
        
    def find(self, addr: str, addr_format: int = 16) -> bool:
        """
        Determines if the given address is in the cache. Returns True if it is
        a cache hit, False otherwise. Inserts the cache line into the cache.

        @param addr: the address of the data to be fetched.
        @param addr_format: an integer indicating number format of the address
        string. The default is 16, which means that by default that the given
        address should be in hex form.
        """
        addr_int = int(addr, addr_format)
        # Finds the set number of the address by applying the set bit-mask
        set = (addr_int & self.set_bit_mask) >> self.offset_bit_size
        # Finds the tag number of the address
        tag = (addr_int & self.tag_bit_mask) \
            >> (self.set_bit_size + self.offset_bit_size)
        cache_hit = self.cache[set].get(tag)
        if not cache_hit:
            self.cache[set].put(tag, addr_int)
        return cache_hit

    def print_state(self) -> None:
        """
        Displays the current state of the set-associative cache as a string.
        Shows which cache lines are filled and which are free, as well
        as the data currently stored in the cache.
        """
        cache_state_table = Texttable()
        cache_state_table.header(["Set", "Tag"])
        for set in range(self.num_sets):
            content = self.cache[set].to_list()
            for tag in content:
                if tag is not None:
                    cache_state_table.add_row([str(set), hex(tag)])
                else:
                    cache_state_table.add_row([str(set), ""])
        cache_state_table.set_cols_align(['c','c'])
        print(cache_state_table.draw())

