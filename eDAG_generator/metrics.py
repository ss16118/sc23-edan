# All classes related to calculating different types
# of metrics based on eDAGs
import os
import math
import bisect
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Set, Dict, Tuple, Union
from array import array
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from instruction_parser import InstructionParser
from eDAG import EDag, OpType



class ReuseDistance:
    """
    An object that encapsulates all the functionalities related to
    reuse distance computation.
    """
    def __init__(self, eDag: EDag) -> None:
        """
        @param eDag: An execution DAG object which will be the
        target of reuse distance analysis.
        """
        self.eDag = eDag
        # Keeps only the memory load vertices
        # Note that this modifies the eDAG in place
        self.eDag.filter_vertices(lambda v: v.op_type == OpType.LOAD_MEM)

    def __compute_reuse_distance(self, addrs: List[str]) -> Dict[int, int]:
        """
        A private helper function that computes reuse distance 
        of each unique data address by iterating through the list of
        given virtual addresses that were accessed.
        
        FIXME: Since it is definitely not the most efficient way to do
        this, it might take a considerable amount of time to obtain
        the reuse histograms for large traces.

        References:
        https://roclocality.org/2015/03/18/efficient-computation-of-reuse-distance/
        https://www.cc.gatech.edu/classes/AY2019/cs7260_spring/lecture30.pdf

        Returns the reuse histogram as a dictionary where the keys are
        distances and the values are numbers of occurrences.
        """
        res = defaultdict(int)
        
        dist = OrderedDict()

        def get_distance(addr: str) -> int:
            """
            A helper function that returns the reuse distance
            of the given address based on its index in `dist`
            """
            # print(dist)
            if addr not in dist:
                # If address is seen for the first time, 
                # pushes the address to the front of `dist` and
                # returns infinity
                dist[addr] = None
                return math.inf
            # Otherwise, traverses through `dist` and finds the index
            # of the key that matches `addr`
            distance = -1
            for idx, key in enumerate(dist.keys()):
                if key == addr:
                    distance = len(dist) - idx - 1
            # Re-inserts the given address into `dist` so that it is
            # at the very front
            dist.pop(addr)
            dist[addr] = None
            assert(distance >= 0)
            return distance
        
        for addr in addrs:
            distance = get_distance(addr)
            res[distance] += 1

        return res

    def get_sequential_reuse_histogram(self) -> Dict[int, int]:
        """
        Returns the reuse distance histogram of the eDAG as if
        it was executed in a sequential program.
        """
        sorted_vs = self.eDag.sorted_vertices
        addrs = [v.data_addr for v in sorted_vs]
        return self.__compute_reuse_distance(addrs)

    def get_all_reuse_histograms(self) -> List[Dict[int, int]]:
        """
        Returns a list containing the reuse distance histograms
        each corresponding with a specific topological ordering of the
        memory load vertices in the eDAG.
        """
        # Uses networkx to obtain all the topological sorts
        graph = nx.DiGraph(self.eDag.edges(False))
        # Finds all the topological sorts with the help of
        # the networkx library
        topo_sorts = nx.all_topological_sorts(graph)
        res = []
        # Currently the limit is 100 since otherwise it will produce way
        # too many plots
        lim = 100
        count = 0
        for sort in tqdm(topo_sorts):
            addrs = [self.eDag.id_to_vertex[v_id].data_addr for v_id in sort]
            res.append(self.__compute_reuse_distance(addrs))
            count += 1
            if count == lim:
                break
        
        return res
        

class BandwidthUtilization:
    """
    A object that computes the bandwidth utilization
    """
    def __init__(self, eDag: EDag, cpu_frequency: int = 10 ** 9) -> None:
        """
        @param eDag: An execution DAG object which will be the
        target of bandwidth utilization analysis.
        @param cpu_frequency: Frequency of CPU which will be used
        for various bandwidth estimation based on the number
        of CPU cycles.
        """
        self.eDag = eDag
        self.cpu_frequency = cpu_frequency

    def get_critical_path_cycles(self, return_dp: bool = False) \
        -> Union[float, Tuple[float, array]]:
        """
        Computes the critical path of the eDAG based on the number of CPU
        cycles associated with each vertex. In turn, the total number of
        compute cycles in the critical path will be returned.
        Note that this function should only be called if a CPU model
        has been used to construct the eDAG and all the vertices have
        their corresponding CPU cycles.
        @param return_dp: If True, the array that is used to the store
        the intermediate values during the computation will also
        be returned in a tuple along with the number of cycles.
        """
        # The critical path is computed with a similar approach as
        # the depth function
        # Computes the topologically sorted list of vertices
        topo_sorted = self.eDag.topological_sort(reverse=True)
        dp = array('f', [0] * len(topo_sorted))
        length = 0
        for v_id in tqdm(topo_sorted):
            _, out_vertices = self.eDag.adj_list[v_id]
            vertex = self.eDag.id_to_vertex[v_id]
            assert vertex.cycles is not None
            if not out_vertices:
                # If vertex is an end node whose out-degree is 0
                dp[v_id] = vertex.cycles
            else:
                for out_vertex_id in out_vertices:
                    new_val = dp[out_vertex_id] + vertex.cycles
                    dp[v_id] = dp[v_id] if dp[v_id] > new_val else new_val
            length = length if length > dp[v_id] else dp[v_id]
        
        if return_dp:
            return (length, dp)

        return length
    
    def get_critical_path(self, dp: Optional[array] = None) -> List[int]:
        """
        Computes the critical path in the eDAG as a sequence of vertices
        in based on the number of CPU cycles assigned to each vertex.
        
        See the similar function `EDag.get_longest_path()` for more details.
        FIXME Duplicate code as `EDag.get_longest_path()`
        @param dp: If provided, will forgo invoking `get_critical_path_cycles()`
        """
        if dp is None:
            _, dp = self.get_critical_path_cycles(True)
        
        path = []
        curr = None
        curr_cycles = 0
        # Finds the vertex that starts the critical path
        # FIXME Can probably use reduce
        for vertex in self.eDag.get_starting_vertices(True):
            if dp[vertex] > curr_cycles:
                curr = vertex
                curr_cycles = dp[vertex]
        assert curr is not None
        path.append(curr)
        # Follows the staring node until the end is reached
        while True:
            _, out_vertices = self.eDag.adj_list[curr]
            curr_cycles = -1
            for out_vertex in out_vertices:
                if dp[out_vertex] > curr_cycles:
                    curr = out_vertex
                    curr_cycles = dp[out_vertex]
            
            if curr_cycles == -1:
                break
            else:
                path.append(curr)

        return path

    def get_avg_bandwidth(self,
                          critical_path_cycles: Optional[float] = None) \
                            -> float:
        """
        Computes the estimation of average memory bandwidth utilization in
        bytes per second by first computing the total amount of data movement
        in the eDAG and then divide it by the time it takes to traverse
        the critical path in the eDAG.
        @param critical_path_cycles: If provided, will forgo invoking
        `get_critical_path_cycles()`.
        """
        if critical_path_cycles is None:
            # Calculates the critical path length in terms of the number
            # of CPU cycles it takes
            critical_path_cycles = self.get_critical_path_cycles()
        
        print(f"[DEBUG] Critical path length: {critical_path_cycles} cycles")
        # Calculates the total amount of data movement in bytes between
        # the CPU and the main memory
        data_movement = 0
        for vertex_id in self.eDag.vertices:
            vertex = self.eDag.id_to_vertex[vertex_id]
            data_movement += vertex.data_size
        
        return data_movement * self.cpu_frequency / critical_path_cycles

    def get_data_movement_over_time(self, cycles: float = 10.0,
                                        use_bandwidth: bool = True) \
                                            -> Tuple[List, np.array]:
        """
        Estimates how the amount of data movement, i.e. number of
        bytes transferred between CPU and the main memory, in an application
        changes based on the given time interval expressed in the number of
        cycles as well as the critical path through the eDAG.
        This is a metric, like parallelism histogram, mainly used
        to discover potential bursts in a program.

        @param cycles: Determines the granularity of time at which the
        data movement will be computed. Can also be viewed as the span of
        a single bin in a histogram. For instance, if the critical path
        is 200 cycles, `cycles` is 20, instructions / vertices will be
        split into 200 / 20 = 10 bins, and data movement will be
        computed separately for each bin.

        @return A tuple containing a two items. The first item will be
        the bins or time intervals while the second item is a numpy array 
        containing the amount of data movement for each time interval, either
        in bytes or in bytes/s, as per the value of `use_bandwidth`.
        """
        topo_sorted = self.eDag.topological_sort(reverse=False)
        dp = array('f', [0] * len(topo_sorted))
        # Computes the maximum number of cycles it takes to
        # reach each vertex from any starting node
        # The algorithm is similar to the one used in `EDag.get_vertex_ran()`
        max_cycles = 0
        for v_id in topo_sorted:
            _, out_vertices = self.eDag.adj_list[v_id]
            vertex = self.eDag.id_to_vertex[v_id]
            assert vertex.cycles is not None
            new_val = dp[v_id] + vertex.cycles
            for out in out_vertices:
                dp[out] = dp[out] if dp[out] > new_val else new_val
            max_cycles = max_cycles if max_cycles > new_val else new_val
        # `max_cycles` should equal to `critical_path_cycles`
        # Splits the time between 0 and `max_cycles` into bins
        # each of size `cycles`
        num_bins = math.ceil(max_cycles / cycles) + 1
        bins = list(range(0, int(max_cycles + cycles), int(cycles)))
        assert len(bins) == num_bins
        res = np.zeros(shape=num_bins,
                        dtype=np.float32 if use_bandwidth else np.int32)
        
        for v_id, v_cycles in enumerate(dp):
            # `v_cycles` is the number of CPU cycles it takes
            # to reach `vertex`, i.e. at which time the
            # vertex will be executed
            vertex = self.eDag.id_to_vertex[v_id]
            # Ignores vertices that do not access memory
            if vertex.data_size > 0:
                # Computes how many buckets the 
                start_bin = bisect.bisect_left(bins, v_cycles)
                end_bin = bisect.bisect_left(bins, v_cycles + vertex.cycles)
                res[start_bin:end_bin] += vertex.data_size
        if use_bandwidth:
            # Converts bytes to bytes/s
            res = res * self.cpu_frequency / cycles
        return bins, res