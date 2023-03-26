# All classes related to calculating different types
# of metrics based on eDAGs
import os
import math
import bisect
import numpy as np
import networkx as nx
from scipy.stats import linregress
from typing import List, Optional, Set, Dict, Tuple, Union, Callable
from array import array
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from instruction_parser import InstructionParser
from eDAG import EDag, OpType, Vertex



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


    def get_avg_bandwidth(self,
                          depth: Optional[float] = None) \
                            -> float:
        """
        Computes the estimation of average memory bandwidth utilization in
        bytes per second by first computing the total amount of data movement
        in the eDAG and then divide it by the time it takes to traverse
        the critical path in the eDAG.
        @param depth: If provided, will forgo invoking
        `EDag.get_depth()`.
        """
        if depth is None:
            # Calculates the critical path length in terms of the number
            # of CPU cycles it takes
            depth = self.eDag.get_depth()
        
        print(f"[DEBUG] Critical path length: {depth} cycles")
        # Calculates the total amount of data movement in bytes between
        # the CPU and the main memory
        data_movement = 0
        for vertex_id in self.eDag.vertices:
            vertex = self.eDag.id_to_vertex[vertex_id]
            data_movement += vertex.data_size
        
        return data_movement * self.cpu_frequency / depth

    def get_data_movement_over_time(self, cycles: float = 10.0,
                                    mode: Optional[str] = None) \
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

        @param mode: Specifies the representation of the result. If None,
        will return the data movement as number of bytes transferred. Other
        possible options include "bandwidth" and "requests". If "bandwidth"
        is chosen, will convert the result from number of bytes to
        bandwidth utilization. If "requests" is used, will return the result
        as the number of outstanding memory requests issued regardless of
        the size of data that is transferred.

        @return A tuple containing a two items. The first item will be
        the bins or time intervals while the second item is a numpy array 
        containing the amount of data movement for each time interval, either
        in bytes or in bytes/s, as per the value of `use_bandwidth`.
        """
        use_bandwidth = False
        use_requests = False
        if mode is not None:
            if mode == "bandwidth":
                use_bandwidth = True
            elif mode == "requests":
                use_requests = True
            else:
                raise ValueError(f"[ERROR] Unknown mode: {mode}")
        
        # Computes the maximum number of cycles it takes to
        # reach each vertex from any starting node
        max_cycles, dp = self.eDag.get_vertex_depth()

        # `max_cycles` should equal to `critical_path_cycles`
        # Splits the time between 0 and `max_cycles` into bins
        # each of size `cycles`
        num_bins = math.ceil(max_cycles / cycles)
        bins = list(range(cycles, int(max_cycles + cycles), int(cycles)))
        # print("Number of bins: ", bins)
        assert len(bins) == num_bins
        res = np.zeros(shape=num_bins,
                        dtype=np.float32 if use_bandwidth else np.int32)

        for v_id, v_cycles in enumerate(dp):
            # `v_cycles` is the number of CPU cycles it takes
            # to reach `vertex`, i.e. at which time the
            # vertex will be executed
            vertex = self.eDag.id_to_vertex[v_id]
            data_size = vertex.data_size
            # Ignores vertices that do not access memory
            if data_size > 0:
                # Computes how many buckets the memory access crosses
                if cycles > 1:
                    start_bin = bisect.bisect_left(bins, v_cycles)
                    end_bin = bisect.bisect_left(bins, v_cycles + vertex.cycles)
                else:
                    start_bin = int(v_cycles)
                    end_bin = int(v_cycles + vertex.cycles)
                
                if use_requests:
                    res[start_bin:end_bin] += 1
                elif use_bandwidth:
                    res[start_bin:end_bin] += \
                        data_size * self.cpu_frequency / vertex.cycles
                else:
                    res[start_bin:end_bin] += data_size
    
        return bins, res
    

class MemoryLatencySensitivity:
    """
    An object that computes the memory latency sensitivity
    of a given eDAG based on different requirements and approaches.
    """
    def __init__(self, eDag: EDag) -> None:
        """
        @param eDag: An execution DAG object which will be the target
        of memory latency sensitivity analysis.
        """
        self.eDag = eDag

    def __recompute_crit_path_with_delta(self, dp: array, delta: float, 
                                         topo_sorted: List[int],
                                         cond: Callable[[Vertex], bool]) \
                                                -> float:
        """
        Recomputes the critical path of an eDAG with the given additional
        latency `delta`.
        To make it more general, the additional latency is only added to 
        vertices that satisfy the given condition.
        Returns a single float representing the new critical path length.
        """
        length = 0
        for v_id in topo_sorted:
            vertex = self.eDag.id_to_vertex[v_id]
            _, out_vertices = self.eDag.adj_list[v_id]
            for out_id in out_vertices:
                new_val = dp[out_id] + vertex.cycles
                dp[v_id] = dp[v_id] if dp[v_id] > new_val else new_val
            if cond(vertex):
                dp[v_id] += delta
            length = length if length > dp[v_id] else dp[v_id]
        return length
    
    def get_random_delay_dist(self, trial_num: int = 1000,
                              remote_mem_per: float = 0.4,
                              delta: float = 50,
                              dp: Optional[array] = None,
                              seed: Optional[int] = None) -> List[float]:
        """
        Assume a model where a certain percentage of memory is allocated
        remotely and the additional latency for accessing that piece of
        memory address is specified by `delta`. This function iterates
        through all memory access vertices, and at randomly decides whether
        the vertex is a remote memory access. After applying the additional
        latency to some vertices, the critical path is recalculated.
        This procedure is repeated for `trial_num` times, and all the
        re-computed critical path lengths are returned in a list.
        """
        # Creates a random number generator object with the given seed
        rng = np.random.default_rng(seed=seed)
        def cond(vertex: Vertex) -> bool:
            if vertex.is_mem_acc and not vertex.cache_hit:
                r = rng.random()
                return r <= remote_mem_per
            return False

        if dp is None:
            _, dp = self.eDag.get_depth(True)
        
        res = []

        topo_sorted = self.eDag.topological_sort(reverse=True)

        for _ in tqdm(range(trial_num)):
            res.append(self.__recompute_crit_path_with_delta(
                array('f', dp), delta, topo_sorted, cond))
        return res


    def get_crit_path_mem_acc_p(self) -> float:
        """
        Calculates the percentage of memory access vertices that lie
        on the critical path compared to the total number of memory
        access vertices in the eDAG.
        """
        # Calculates the number of memory access vertices along the
        # critical path
        m = self.eDag.get_depth(mem_acc_only=True)
        mem_acc_vertices = \
            self.eDag.get_vertices(lambda v: v.is_mem_acc and not v.cache_hit)
        print(f"[DEBUG] m: {m}, tot: {len(mem_acc_vertices)}")
        if not mem_acc_vertices:
            return 0
        return m / len(mem_acc_vertices)
        
    def get_simple_mls(self, return_k: bool = False, 
                       delta_range: Optional[range] = None,
                       depth: Optional[float] = None,
                       dp: Optional[array] = None,
                       critical_path: Optional[List[int]] = None) \
                        -> Union[Dict[float, float], float]:
        """
        Computes the memory latency sensitivity of the given eDAG as per
        Brent's theorem, which states that the upper bound for the
        execution of an eDAG is W/p + D where W is the total amount of
        work, D is the critical path length, or in this case, the depth, 
        and p is the number of processors in use. 
        Since we are only considering an ideal scenario in which an
        infinite number of processors are used, we know that the
        both the lower bound and upper bound of execution time will be
        determined simply by the critical path length. What this implies,
        in turn, is that the memory latency sensitivity of any application,
        under our idealized scenario, is entirely dependent on the number
        of memory access vertices on the critical path. To this end,
        we can vary the amount of additional latency within a given range
        and measure the corresponding critical path lengths. When we plot
        the slowdown against additional latency, the trendline we obtain
        should be asymptotically linear, and we define the slope k 
        to be the memory latency sensitivity.
        @param return_k: If True, will return the memory latency
        sensitivity as the slope of the trendline. If False, will
        return a dictionary that represents the graph itself where the
        keys are x and values are y.
        @param delta_range: The range of additional latency to explore,
        i.e. x values for the plot. If None, will use the default
        `range(0, 200, 10)`.
        @param depth: If given will forgo the invocation
        of `EDag.get_depth()`. Used as the baseline of comparison
        @param critical_path: A list containing the IDs of the vertices on
        the critical path. If given will forgo the invocation of
        `EDag.get_longest_path()`.
        """
        
        if depth is None or dp is None:
            depth, dp = self.eDag.get_depth(True)

        if return_k:
            # If we only need the slope of the trend line, we have to simply
            # to find the number of vertices along the critical
            # path with the most number of memory access vertices
            # and divide that by the original critical path length

            # Obtains the number of memory access vertices along the path
            m = self.eDag.get_depth(mem_acc_only=True)
            print(f"[DEBUG] Number of mem acc vs: {m}")
            return m / depth

        if delta_range is None:
            delta_range = range(0, 200, 10)
        
        if critical_path is None:
            critical_path = self.eDag.get_longest_path(dp=dp)

        topo_sorted = self.eDag.topological_sort(reverse=True)

        res = {}
        for delta in tqdm(list(delta_range)[1:]):
            # New critical path length
            # Makes sure to copy the original array
            new_cp = self.__recompute_crit_path_with_delta(
                        array('f', dp), delta, topo_sorted,
                        lambda v: v.is_mem_acc and not v.cache_hit)
            slowdown = new_cp / depth
            res[delta] = slowdown
            
        return res