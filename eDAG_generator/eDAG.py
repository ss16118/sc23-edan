from typing import List, Dict, Optional, Set, Tuple, Callable   
from graphviz import Digraph


class Vertex:
    """
    A class representation of a single vertex in an execution DAG.
    It is made of a single assembly instruction that belongs to
    an arbitrary ISA.
    """
    def __init__(self, id: int, instruction: str, operands: List[str],
                target: Optional[str], dependencies: Set[str],
                is_mem_acc: bool = False, is_mem_load: bool = False) -> None:
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
        @param is_mem_load: a boolean value indicating whether the instruction
        represented by this vertex is loading from memory.
        @param is_mem_acc: a boolean value that indicates whether
        this specific instruction contains memory access.
        """
        self.id = id
        self.instruction = instruction
        self.operands = tuple(operands)
        self.target = target
        self.dependencies = dependencies
        self.is_mem_acc = is_mem_acc
        self.is_mem_load = is_mem_load
        # given that this vertex represents a memory access
        # instruction, this argument indicates whether the data stored in the
        # memory address is already in cache according to a predefined
        # cache model. This variable should be set at a later stage
        # if a cache model is used.
        self.cache_hit = False

    def __str__(self) -> str:
        return f"{self.id}: {self.instruction} {','.join(self.operands)}"
    
    def __repr__(self) -> str:
        return f"{self.id}: {self.instruction} {','.join(self.operands)}"

    def __hash__(self) -> int:
        return hash((self.id, self.instruction, self.operands))


class EDag:
    """
    An object that represents an execution Directed Acyclic Graph (eDAG)
    as an modified adjacency list of different vertices
    """
    _in = 0
    _out = 1

    def __init__(self) -> None:
        self.vertices: Set[Vertex] = set()
        # A dictionary that maps the ID of the vertex to the
        # corresponding object. This is only present for
        # easier access to specific vertex
        self.id_to_vertex: Dict[int, Vertex] = {}
        # An adjacency list that maps each vertex to two sets
        # of vertices. The first set will be the vertices
        # on which it depends, while the second set will be
        # the vertices it points to, i.e. vertices that depend on it
        self.adj_list: Dict[Vertex, List[Set[Vertex]]] = {}

    def add_vertex(self, vertex: Vertex) -> None:
        """
        Adds a single vertex to this execution DAG.
        """
        if vertex not in self.vertices:
            self.vertices.add(vertex)
            self.id_to_vertex[vertex.id] = vertex

        if vertex not in self.adj_list:
            self.adj_list[vertex] = [set(), set()]

    def add_edge(self, source: Vertex, target: Vertex) -> None:
        """
        Adds a connection between the given source and target vertices,
        and the relationship between the vertices is that the target vertex
        depends on the source, as in the source precedes the target.
        """
        assert(source in self.vertices)
        assert(target in self.vertices)
        self.adj_list[source][EDag._out].add(target)
        self.adj_list[target][EDag._in].add(source)

    def get_starting_vertices(self) -> Set[Vertex]:
        """
        Retrieves a set of starting vertices, i.e. vertices whose
        in degree is 0.
        """
        res = set()

        in_out_degrees = self.get_in_out_degrees()
        for vertex_id, (in_degree, _) in in_out_degrees.items():
            if in_degree == 0:
                res.add(self.id_to_vertex[vertex_id])
        return res

    def edges(self) -> List[Tuple[str, str]]:
        """
        Converts the adjacency list to a list of directed edges between 
        two vertices. Note that vertices are represented with their IDs.
        For instance, an edge of ('1', '2') denotes a directed edge from 
        vertex 1 to vertex 2.
        """
        edges = []
        for source, (_, out_vertices) in self.adj_list.items():
            for target in out_vertices:
                edges.append((f"{source.id}", f"{target.id}"))
        return edges


    def remove_single_vertices(self) -> None:
        """
        Removes all the vertices that have no dependencies from the eDAG.
        "No dependencies" means that a vertex does not have any edges
        connected to it, i.e. both in and out degrees of the vertex are 0.
        """
        in_out_degrees = self.get_in_out_degrees()
        for vertex_id, (in_degree, out_degree) in in_out_degrees.items():
            if in_degree == 0 and out_degree == 0:
                # Retrieves the vertex to be removed from the dictionary
                vertex = self.id_to_vertex.get(vertex_id)
                assert(vertex is not None)
                del self.id_to_vertex[vertex_id]
                # Removes the vertex from the set of vertices in eDAG
                self.vertices.remove(vertex)

                # Removes the vertex from the adjacency list
                del self.adj_list[vertex]
        

    def get_in_out_degrees(self) -> Dict[int, List[int]]:
        """
        Returns the in and out degrees of all the vertices in the eDAG.
        The output is in the format of a dictionary that maps the
        the ID of a vertex to a list containing two integers [`in`, `out`], 
        where `in` and `out` denote the in and out degrees respectively.
        """
        res: Dict[int, List[int]] = {}
        for vertex, (in_vertices, out_vertices) in self.adj_list.items():
            res[vertex.id] = [len(in_vertices), len(out_vertices)]
        return res


    def filter_vertices(self, cond: Callable[[Vertex], bool]) -> None:
        """
        Given a certain condition, iteratively removes all nodes from the eDAG 
        which do not satisfy the condition, i.e. when called, the function
        returns False. After a vertex is removed, the dependencies of its
        predecessors and successors are maintained. For instance, if A -> B,
        B -> C, and B is removed, then the edge A -> C will be added.
        """
        filtered_vertices = set()
        for vertex in self.vertices:
            # If condition is not satisfied
            if not cond(vertex):
                # Removes vertex v from the dictionary
                del self.id_to_vertex[vertex.id]
                # Retrieves the set of vertices on which v
                # depend and the set that depend on v
                assert(vertex in self.adj_list)
                in_vertices, out_vertices = self.adj_list[vertex]
                for in_vertex in in_vertices:
                    # Removes v from the out set of in_vertices
                    self.adj_list[in_vertex][EDag._out].remove(vertex)
                for out_vertex in out_vertices:
                    # Removes v from the in set of out_vertices
                    self.adj_list[out_vertex][EDag._in].remove(vertex)
                for in_vertex in in_vertices:
                    for out_vertex in out_vertices:
                        # Adds an edge between each pair of 
                        # in_vertex and out_vertex
                        self.add_edge(in_vertex, out_vertex)
                # Removes vertex v from the adjacency list entirely
                del self.adj_list[vertex]
            else:
                filtered_vertices.add(vertex)

        self.vertices = filtered_vertices

    def get_depth(self) -> int:
        """
        Calculates the depth, i.e. in this case diameter of the eDAG by 
        exploring the graph from each starting vertex to each end vertex
        and keeping track of the longest distance.
        """
        def find_depth(curr: Vertex, depth: int) -> int:
            """
            A helper function that performs depth-first-search recursively
            to find the depth of the eDAG starting from a certain vertex.
            """
            out_vertices = self.adj_list[curr][EDag._out]
            if len(out_vertices) == 0:
                # Base case: checks if the current vertex is the end vertex
                # i.e. out degree is 0
                return depth

            max_depth = 0
            for out_vertex in out_vertices:
                max_depth = max(max_depth, find_depth(out_vertex, depth + 1))
            return max_depth
        
        depth = 0
        for vertex in self.get_starting_vertices():
            depth = max(depth, find_depth(vertex, 0))

        return depth
    
    def get_work(self, cond: Optional[Callable[[Vertex], bool]] = None) -> int:
        """
        Counts the number of vertices which satisfy the given condition. If
        a condition is not provided, the output will simply be the total number
        of vertices in the eDAG.
        """
        if cond is None:
            return len(self.vertices)

        work_count = 0
        for vertex in self.vertices:
            if cond(vertex):
                work_count += 1
        return work_count

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
                if vertex.is_mem_load and vertex.cache_hit:
                    # The vertex color should be green if cache hit
                    color = "green"
                else:
                    color = "red"
                
                graph.node(f"{vertex.id}", str(vertex),
                            style="filled", color=color)
            else:
                graph.node(f"{vertex.id}", str(vertex))
        graph.edges(self.edges())
        return graph