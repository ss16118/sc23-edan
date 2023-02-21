from graphviz import Digraph
from typing import Dict, List, Optional, Set
from eDAG import EDag, Vertex


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


def visualize_longest_path(eDag: EDag, graph: Digraph, highlight_mem_acc: bool,
                        longest_path: List[int], color: str = "orange") -> None:
    """
    Highlights the given longest path, i.e. the vertices and edges,
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
        graph.edge(str(source), str(dst), color=color)
