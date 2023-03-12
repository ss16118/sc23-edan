import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from multiprocessing import Value, Lock
from array import array
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Union, Tuple
from eDAG import EDag, Vertex


G = 10 ** 9
M = 10 ** 6
K = 10 ** 3


class AtomicCounter(object):
    """
    An atomic counter that can be shared among multiple processes.
    Implementation from:
    https://stackoverflow.com/questions/2080660/how-to-increment-a-shared-counter-from-multiple-processes
    """
    def __init__(self, n: int = 0) -> None:
        """
        Initializes the value of the counter to be `n`.
        """
        self.val = Value('i', n)
    
    def increment(self) -> None:
        """
        Increments the value of the counter by 1 in a process-safe manner.
        """
        with self.val.get_lock():
            self.val.value += 1
    
    @property
    def value(self) -> int:
        """
        Returns the current value of the counter.
        """
        with self.val.get_lock():
            return self.val.value



def get_vertex_attrs(vertex: Vertex, highlight_mem_acc: bool) -> Dict:
    """
    Returns a dictionary describing the attributes of the given vertex based
    on its opcode and whether it is a cache-hit. Used specifically for
    eDAG visualization.
    """
    style = {
        "name": f"{vertex.id}",
        "label": str(vertex)
    }
    if highlight_mem_acc and vertex.is_mem_acc:
        style["style"] = "filled"
        if vertex.cache_hit:
            style["fillcolor"] = "green"
        else:
            style["fillcolor"] = "red"
    return style


def visualize_eDAG(eDag: EDag, highlight_mem_acc: bool = True,
            large_graph_thresh: int = 5000,
            vertex_rank: Optional[Dict[int, Set[int]]] = None) -> Digraph:
    """
    Converts the given eDAG to an graphviz.Digraph, which can then be
    rendered and saved as a PDF file.
    @param highlight_mem_acc: If set to True, vertices that perform memory
    accesses will be highlighted.
    @param large_graph_thresh: An integer which marks the threshold above
    which a graph will be considered large and the layout engine used for
    graphviz will be changed to sfdp instead of dot.
    @param vertex_ranks: If not None, vertices that are the same distance
    way from a starting vertex will be plotted on the same horizontal line.
    """
    # Uses sfdp as the layout engine for large graph
    engine = "sfdp" if len(eDag.vertices) > large_graph_thresh else None
    graph = Digraph(engine=engine,
                    graph_attr={"overlap": "scale"}, strict=True)
    if vertex_rank:
        # Organizes the diagram by placing vertices with the same
        # distance from the starting vertices on a single line
        for _, vertices in vertex_rank.items():
            subgraph = graph.subgraph()
            with graph.subgraph() as subgraph:
                subgraph.attr(rank="same")
                for vertex_id in vertices:
                    if vertex_id in eDag.removed_vertices:
                        continue
                    vertex = eDag.id_to_vertex[vertex_id]
                    subgraph.node(**get_vertex_attrs(vertex, highlight_mem_acc))
    else:
        for vertex_id in eDag.vertices:
            vertex = eDag.id_to_vertex[vertex_id]
            vertex_attrs = get_vertex_attrs(vertex, highlight_mem_acc)
            graph.node(**vertex_attrs)
    graph.edges(eDag.edges())
    return graph


def visualize_reuse_histogram(reuse_distance: Dict[int, int],
                        fig_path: Optional[str] = None) -> None:
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

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def visualize_path(eDag: EDag, graph: Digraph, highlight_mem_acc: bool,
                   longest_path: List[int], color: str = "orange") -> None:
    """
    Highlights the given path, i.e. the vertices and edges,
    in the visualization of the eDAG.
    """
    # Highlights the vertices
    for vertex_id in longest_path:
        vertex = eDag.id_to_vertex[vertex_id]
        vertex_attrs = \
            get_vertex_attrs(vertex, highlight_mem_acc)
        vertex_attrs["color"] = color
        vertex_attrs["penwidth"] = "3"
        graph.node(**vertex_attrs)
    
    # Highlights the edges along the longest path
    for i, source in enumerate(longest_path[:-1]):
        dst = longest_path[i + 1]
        graph.edge(str(source), str(dst), color=color, penwidth="3")


def visualize_data_movement_over_time(bins: List[int], data_movement: np.array,
                                      mode: Optional[str] = None,
                                      fig_path: Optional[str] = None) -> None:
    """
    Plots the data movement over time of a program based on the given
    bins and the numpy array containing the amounts of data moved either
    in bytes or bytes/s. See `Bandwidth.data_movement_over_time()` for
    more information.

    @param mode: See description of argument `mode` in 
    `BandwidthUtilization.get_data_movement_over_time()`.

    @param fig_path: If not None, the plot will be saved to the
    given file.
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
        
    plt.figure()
    y_label = "Bytes moved"
    x_label = "CPU cycles"
    if use_bandwidth:
        # Convert bytes/s to GB/s
        # print(data_movement)
        data_movement /= G
        y_label = "Bandwidth utilization [GB/s]"
    
    if use_requests:
        y_label = "Number of outstanding memory requests"

    assert len(bins) >= 2
    bar_width = bins[1] - bins[0]
    plt.bar(bins, data_movement, width=bar_width, align="edge")
    # plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def visualize_memory_latency_sensitivity(data: Dict[float, float],
                                         fig_path: Optional[str] = None) \
                                            -> None:
    """
    Plots the memory latency sensitivity graph where the x-axis is the
    additional latency and the y-axis is the slowdown.
    """

    x, y = list(zip(*data.items()))
    
    plt.plot(x, y)
    
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


"""
TODO: Move the following functions to be class functions of EDag.
"""

def get_critical_path_cycles(eDag: EDag, return_dp: bool = False) \
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
    topo_sorted = eDag.topological_sort(reverse=True)
    dp = array('f', [0] * len(topo_sorted))
    length = 0
    for v_id in tqdm(topo_sorted):
        _, out_vertices = eDag.adj_list[v_id]
        vertex = eDag.id_to_vertex[v_id]
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


def get_critical_path(eDag: EDag, dp: Optional[array] = None) -> List[int]:
    """
    Computes the critical path in the eDAG as a sequence of vertices
    in based on the number of CPU cycles assigned to each vertex.
    
    See the similar function `EDag.get_longest_path()` for more details.
    FIXME Duplicate code as `EDag.get_longest_path()`
    @param dp: If provided, will forgo invoking `get_critical_path_cycles()`
    """
    if dp is None:
        _, dp = get_critical_path_cycles(eDag, True)
    
    path = []
    curr = None
    curr_cycles = 0
    # Finds the vertex that starts the critical path
    # FIXME Can probably use reduce()
    for vertex in eDag.get_starting_vertices(True):
        if dp[vertex] > curr_cycles:
            curr = vertex
            curr_cycles = dp[vertex]
    assert curr is not None
    path.append(curr)
    # Follows the staring node until the end is reached
    while True:
        _, out_vertices = eDag.adj_list[curr]
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