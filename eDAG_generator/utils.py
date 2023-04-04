import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from multiprocessing import Value, Lock
from array import array
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Union, Tuple, Iterable, Any
from eDAG import EDag, Vertex
from enum import Enum, auto
from matplotlib.animation import FuncAnimation, PillowWriter
from metrics import MemoryLatencySensitivity


G = 10 ** 9
M = 10 ** 6
K = 10 ** 3


class ISA(Enum):
    RISC_V = auto()
    ARM = auto()
    X86 = auto()


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
                    graph_attr={"overlap": "scale"},
                    node_attr={"fontsize": "14"},
                    strict=True)
    if vertex_rank:
        # Organizes the diagram by placing vertices with the same
        # distance from the starting vertices on a single line
        for _, vertices in sorted(vertex_rank.items(), key=lambda p: p[0]):
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
    # Adds memory dependencies caused by limited number of issue slots
    # as dashed lines
    for source, target in eDag.mem_slot_deps.items():
        graph.edge(str(source), str(target), style="dashed")
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
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 5))
    y_label = "Data moved [Bytes]"
    x_label = "CPU cycles"
    if np.max(data_movement) > 1000:
        data_movement = np.divide(data_movement, 1000)
        y_label = "Data moved [kB]"

    if use_bandwidth:
        # Convert bytes/s to GB/s
        # print(data_movement)
        data_movement /= G
        y_label = "Bandwidth utilization [GB/s]"
    
    if use_requests:
        y_label = "Number of outstanding memory requests"

    assert len(bins) >= 2
    bar_width = bins[1] - bins[0]
    plt.bar(bins, data_movement, width=bar_width, align="edge", alpha=1)
    # plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"[INFO] Data movement plot saved to {fig_path}")
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


def visualize_distribution(data: Iterable,
                           baseline: Optional[float] = None,
                           fig_path: Optional[str] = None) -> None:
    """
    Visualizes the given data in a histogram.
    """
    print(plt.hist(data, bins=20, color='orange'))
    if baseline is not None:
        plt.axvline(x=baseline, color="blue", label="Baseline")
        plt.legend()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def animate_crit_path_dist(mls_metric: MemoryLatencySensitivity,
                           trial_num: int = 1000,
                           baseline: Optional[float] = None,
                           dp: Optional[array] = None,
                           num_frames: int = 20,
                           seed: Optional[int] = 42,
                           fig_path: Optional[str] = None) -> None:
    """
    Animates how the critical path length distribution changes
    as different parameters vary.
    """
    num_bins = 20

    def animate(frame_number):
        if frame_number > 5:
            return
        plt.cla()
        p = frame_number / num_frames
        data = mls_metric.get_random_delay_dist(trial_num=trial_num,
                                                dp=dp,
                                                remote_mem_per=p,
                                                seed=seed)
        plt.hist(data, bins=num_bins, color="orange")
        # hist, bins = np.histogram(data, bins=num_bins)
        # plt.bar(bins[:-1], hist, color="orange")
        plt.title(f"p = {p:.3}")
        plt.xlabel("Critical Path Length (Cycles)")
        plt.ylabel("Counts")
        plt.ylim([0, trial_num + 5])
        if baseline is not None:
            plt.axvline(x=baseline, color="blue", label="Baseline")
            plt.legend()

    fig = plt.figure()
    # data = mls_metric.get_random_delay_dist(dp=dp, remote_mem_per=0)
    # plt.hist(data, num_bins, color='orange')
    anim = FuncAnimation(fig, animate, num_frames, repeat=True, blit=False,
                         cache_frame_data=True)

    if fig_path is not None:
        # Saves the animation as a gif file
        writer = PillowWriter(fps=2)
        anim.save(fig_path, writer=writer)
    else:
        plt.show()
    plt.close()


def save_data_to_file(file_path: str, data: str) -> None:
    """
    Saves the given data in the form of a string to the specified file.
    """
    with open(file_path, "w+") as file:
        file.write(data)

def save_data_movement_to_file(file_path: str, bins: List[int], data: array) \
    -> None:
    str_data = ""
    for bin, d in zip(bins, data):
        str_data += f"{bin},{d}\n"
    save_data_to_file(file_path, str_data)