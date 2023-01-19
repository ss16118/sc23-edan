from typing import List, Dict, Optional, Set, Tuple
from graphviz import Digraph


class Vertex:
    """
    A class representation of a single vertex in an execution DAG.
    It is made of a single assembly instruction that belongs to
    an arbitrary ISA.
    """
    def __init__(self, id: int, instruction: str, operands: List[str],
                target: Optional[str], dependencies: Set[str],
                is_mem_acc: bool) -> None:
        """
        @param id: a unique number given to each vertex by the parser.
        @param instruction: assembly instruction, e.g. add, sub, mv.
        @param operands: a list of operands represented as strings.
        @param target: target register or memory location, i.e. the register 
        or memory location whose value is updated. It can be None for 
        certain instructions. As an example, the target for `sd a1,-64(s0)` is
        the memory location `-64(s0)`.
        @param dependencies: a set of registers or memory locations on which
        this vertex depends. As an example, the command `sd a1,-64(s0)` depends
        on the registers a1 and s0, whereas the command `ld a5,-40(s0)` depends
        on the memory location denoted by `-40(s0)`.
        @param is_mem_acc: a boolean value that indicates whether
        this specific instruction contains memory access.
        """
        self.id = id
        self.instruction = instruction
        self.operands = tuple(operands)
        self.target = target
        self.dependencies = dependencies
        self.is_mem_acc = is_mem_acc

    def __str__(self) -> str:
        return f"{self.id}: {self.instruction} {','.join(self.operands)}"

    def __hash__(self) -> int:
        return hash((self.id, self.instruction, self.operands))


class EDag:
    """
    An object that represents an execution Directed Acyclic Graph (eDAG)
    as an adjacency list of different vertices
    """
    def __init__(self) -> None:
        self.vertices: Set[Vertex] = set()
        # An adjacency list that maps each vertex of a set
        # of vertices it points to
        self.adj_list: Dict[Vertex, Set[Vertex]] = {}

    def add_vertex(self, vertex: Vertex) -> None:
        """
        Adds a single vertex to this execution DAG.
        """
        # Makes sure that we do not add the same vertex more than once
        assert(vertex not in self.vertices)
        self.vertices.add(vertex)

        if self.adj_list.get(vertex) is None:
            self.adj_list[vertex] = set()

    def add_edge(self, source: Vertex, target: Vertex) -> None:
        """
        Adds a connection between the given source and target vertices,
        and the relationship between the vertices is that the target vertex
        depends on the source, as in the source precedes the target.
        """
        assert(source in self.vertices)
        self.adj_list[source].add(target)

    def edges(self) -> List[Tuple[str, str]]:
        """
        Converts the adjacency list to a list of directed edges between 
        two vertices. Note that vertices are represented with their IDs.
        For instance, an edge of ('1', '2') denotes a directed edge from 
        vertex 1 to vertex 2.
        """
        edges = []
        for source, targets in self.adj_list.items():
            for target in targets:
                edges.append((f"{source.id}", f"{target.id}"))
        return edges


    def remove_single_vertices(self) -> None:
        """
        Removes all the vertices that have no dependencies from the eDAG.
        "No dependencies" means that a vertex does not have any edges
        connected to it.
        """
        non_single_vertices = set()
        # Iterates through the adjacency list to collect all
        # vertices that have at least one edge connected to them
        for source, targets in self.adj_list.items():
            if len(targets) > 0:
                non_single_vertices.add(source)
                non_single_vertices = non_single_vertices.union(targets)
        single_vertices = self.vertices.difference(non_single_vertices)
        
        new_adj_list: Dict[Vertex, Set[Vertex]] = {}
        for source, targets in self.adj_list.items():
            if source not in single_vertices:
                new_adj_list[source] = targets.difference(single_vertices)
        self.vertices = non_single_vertices
        self.adj_list = new_adj_list

    def visualize(self, highlight_mem_acc: bool = True) -> Digraph:
        """
        Converts the eDAG to an graphviz.Digraph, which can then be
        rendered and saved as a PDF file.
        @param highlight_mem_acc: If set to True, vertices that perform memory
        accesses will be highlighted.
        """
        graph = Digraph()
        for vertex in self.vertices:
            if highlight_mem_acc and vertex.is_mem_acc:
                graph.node(f"{vertex.id}", str(vertex),
                            style="filled", color="red")
            else:
                graph.node(f"{vertex.id}", str(vertex))
        graph.edges(self.edges())
        return graph