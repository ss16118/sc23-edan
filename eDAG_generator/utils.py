import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from multiprocessing import Value, Lock
from typing import Dict, List, Optional, Set
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
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

