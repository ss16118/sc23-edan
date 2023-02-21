# All classes related to calculating different types
# of metrics based on either traces or eDAGs
import os
import math
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Set, Dict, Tuple
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
        lim = 100
        count = 0
        for sort in tqdm(topo_sorts):
            addrs = [self.eDag.id_to_vertex[v_id].data_addr for v_id in sort]
            res.append(self.__compute_reuse_distance(addrs))
            count += 1
            if count == lim:
                break
        
        return res

    @staticmethod
    def plot_histogram(reuse_distance: Dict[int, int],
                       save_fig_path: Optional[str] = None) -> None:
        """
        Plots the given reuse histogram with matplotlib.
        If `save_fig_path` is not None, the generated histogram
        will be saved to the specified path.
        """
        # Unzips the key-value pairs in the dictionary into two lists
        reuse_distance = sorted(list(reuse_distance.items()),
                                key=lambda p: p[0])
        x, y = list(zip(*reuse_distance))
        x = list(map(lambda elem: str(elem), x))
        plt.figure(figsize=(10, 5))
        plt.bar(x, y)

        # Sets the axis labels
        plt.xlabel("Reuse distance")
        # plt.yscale("log")
        plt.ylabel("Number of references")

        _, labels = plt.xticks()
        plt.setp(labels, rotation=90)

        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        

class BandwidthUtilization:
    """
    A object that computes the bandwidth utilization
    """
    def __init__(self, eDag: EDag) -> None:
        """
        @param eDag: An execution DAG object which will be the
        target of bandwidth utilization analysis.
        """
        self.eDag = eDag

    def get_critical_path_cycles(self) -> float:
        """
        Computes the critical path of the eDAG based on the number of CPU
        cycles associated with each vertex. In turn, the total number of
        compute cycles in the critical path will be returned.
        Note that this function should only be called if a CPU model
        has been used to construct the eDAG and all the vertices have
        their corresponding CPU cycles.
        """
        # The critical path is computed with a similar approach as
        # the depth function
        # Computes the topologically sorted list of vertices
        topo_sorted = self.eDag.topological_sort(reverse=True)
        dp = array('f', [0] * len(topo_sorted))
        length = 0
        print(topo_sorted)
        for v_id in tqdm(topo_sorted):
            _, out_vertices = self.eDag.adj_list[v_id]
            vertex = self.eDag.id_to_vertex[v_id]
            assert(vertex.cycles is not None)
            if not out_vertices:
                # If vertex is an end node whose out-degree is 0
                dp[v_id] = vertex.cycles
            else:
                for out_vertex_id in out_vertices:
                    new_val = dp[out_vertex_id] + vertex.cycles
                    dp[v_id] = dp[v_id] if dp[v_id] > new_val else new_val
            length = length if length > dp[v_id] else dp[v_id]
                
        return length
        
    def compute_bandwidth(self, cpu_frequency: int = 10 ** 9) -> float:
        """
        Computes the estimation of average memory bandwidth utilization in
        bytes per second by first computing the total amount of data movement
        in the eDAG and then divide it by the time it takes to traverse
        the critical path in the eDAG.
        """
        # Calculates the critical path length in terms of the number
        # of CPU cycles it takes
        critical_path_length = self.get_critical_path_cycles()
        print(f"[DEBUG] Critical path length: {critical_path_length} cycles")
        # Calculates the total amount of data movement in bytes between
        # the CPU and the main memory
        data_movement = 0
        for vertex_id in self.eDag.vertices:
            vertex = self.eDag.id_to_vertex[vertex_id]
            data_movement += vertex.data_size
        
        return data_movement * cpu_frequency / critical_path_length
