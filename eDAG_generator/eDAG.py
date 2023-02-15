from __future__ import annotations
from enum import Enum, auto
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional, Set, Tuple, Callable, Union
from graphviz import Digraph


class OpType(Enum):
    """Types of operations for a single vertex"""
    # Load from memory
    LOAD_MEM = auto()
    # Stores to a memory address
    STORE_MEM = auto()
    # Load an immediate value
    LOAD_IMM = auto()
    # Arithmetic operations between registers, e.g. add, sub
    ARITHMETIC = auto()
    # Arithmetic operations involving immediate values, e.g. addi
    ARITHMETIC_IMM = auto()
    # Move
    MOVE = auto()
    # Branch
    BRANCH = auto()


class Vertex:
    """
    A class representation of a single vertex in an execution DAG.
    It is made of a single assembly instruction that belongs to
    an arbitrary ISA.
    """
    def __init__(self, id: int, opcode: str, operands: List[str],
                target: Optional[str], dependencies: Set[str],
                op_type: OpType, is_comm_assoc: bool = False,
                # Optional attributes
                cpu: Optional[int] = None,
                insn_addr: Optional[None] = None,
                data_addr: Optional[str] = None,
                data_size: Optional[int] = 0) -> None:
        """
        @param id: a unique number given to each vertex by the parser.
        @param opcode: opcode of the assembly instruction, e.g. add, sub, mv.
        @param operands: a list of operands represented as strings.
        @param target: target register or memory location, i.e. the register 
        or memory location whose value is updated. It can be None for 
        certain instructions. As an example, the target for `sd a1,-64(s0)` is
        the memory location `-64(s0)`.
        @param dependencies: a set of registers or memory locations on which
        this vertex depends. As an example, the command `sd a1,-64(s0)` depends
        on the registers a1 and s0, whereas the command `ld a5,-40(s0)` depends
        on the memory location denoted by `-40(s0)`.
        @param op_type: Type of operation performed in this vertex.
        @param is_comm_assoc: Indicates whether the operation performed by
        the instruction is both commutative and associative. Used by
        the subgraph optimizer.
        @param cpu: ID of the CPU on which the instruction was executed.
        @param insn_addr: Virtual memory address where the instruction contained
        in this vertex is stored.
        @param data_addr: Data address of the virtual memory accessed in 
        this vertex. Note that this argument should be None unless the opcode 
        type is a memory access.
        @param data_size: Size of data movement in bytes when the vertex
        represents a memory access operation. If it is not memory access,
        this argument should 0.
        TODO: can later extend `data_size` to non-memory-access operations.
        """
        self.id = id
        self.opcode = opcode
        self.operands = tuple(operands)
        self.target = target
        self.dependencies = dependencies
        self.op_type = op_type
        self.is_comm_assoc = is_comm_assoc
        self.cpu = cpu
        self.insn_addr = insn_addr
        self.data_addr = data_addr
        self.data_size = data_size
        # Given that this vertex represents a memory access
        # instruction, this argument indicates whether the data stored in the
        # memory address is already in cache according to a predefined
        # cache model. This variable should be set at a later stage
        # if a cache model is used.
        self.cache_hit = False

    @property
    def is_mem_acc(self) -> bool:
        """
        A vertex contains a memory access operation if it is
        either a LOAD_MEM or STORE_MEM.
        """
        return self.op_type == OpType.LOAD_MEM or \
            self.op_type == OpType.STORE_MEM

    @property
    def asm(self) -> str:
        """
        Returns the assembly instruction as a string
        """
        return f"{self.opcode} {','.join(self.operands)}"
    
    def full_str(self) -> str:
        """
        Converts all the available information in the vertex into a string.
        If all the optional attributes are provided, the returned string
        should be exactly the same as when the instruction was parsed.
        """
        res = self.asm
        if self.insn_addr is not None:
            res = f"{self.insn_addr} {res}"
        if self.cpu is not None:
            res = f"{self.cpu} {res}"
        if self.data_addr is not None:
            res = f"{res} {self.data_addr}"
        return res

    def __str__(self) -> str:
        return f"{self.id}: {self.asm}"
    
    def __repr__(self) -> str:
        return f"{self.id}: {self.asm}"

    def __hash__(self) -> int:
        return hash((self.id, self.opcode, self.operands))


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
        
        # A list of disjoint subgraphs contained in the eDAG.
        # Note that this list will not be filled automatically while
        # constructing the eDAG. split_disjoint_subgraphs() have to
        # be invoked to retrieve them
        self.disjoint_subgraphs: List[EDag] = []

    @property
    def sorted_vertices(self) -> List[Vertex]:
        """
        Sorts the vertices in the eDAG by their ID, i.e. the order
        in which they were added / parsed, and returns them as a list.
        """
        return [vertex for (_, vertex) in sorted(self.id_to_vertex.items())]

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
        # Ensures that both the source and target vertices have already
        # been added to the eDAG
        assert(source in self.vertices)
        assert(target in self.vertices)
        self.adj_list[source][EDag._out].add(target)
        self.adj_list[target][EDag._in].add(source)

    def get_starting_vertices(self, id_only: bool = False) \
        -> Union[Set[Vertex], Set[int]]:
        """
        Retrieves a set of starting vertices, i.e. vertices whose
        in degree is 0. If `id_only` is True, only IDs of the vertices
        will be returned.
        """
        res = set()

        in_out_degrees = self.get_in_out_degrees()
        for vertex_id, (in_degree, _) in in_out_degrees.items():
            if in_degree == 0:
                if id_only:
                    res.add(vertex_id)
                else:
                    res.add(self.id_to_vertex[vertex_id])
        return res

    def get_end_vertices(self, id_only: bool = False) \
        -> Union[Set[Vertex], Set[int]]:
        """
        Retrieves a set of end vertices, i.e. vertices whose out degree is 0.
        If `id_only` is True, only IDs of the vertices will be returned.
        """
        res = set()
        in_out_degrees = self.get_in_out_degrees()
        for vertex_id, (_, out_degree) in in_out_degrees.items():
            if out_degree == 0:
                if id_only:
                    res.add(vertex_id)
                else:
                    res.add(self.id_to_vertex[vertex_id])
        return res

    def get_vertex_id_adj_list(self) -> Dict[int, List[Set[int]]]:
        """
        Iterates over the current adjacency list and converts all `Vertex`
        objects to their corresponding IDs for faster processing.
        """
        adj_list = {}
        for vertex, (in_vertices, out_vertices) in self.adj_list.items():
            in_ids = set(map(lambda v: v.id, in_vertices))
            out_ids = set(map(lambda v: v.id, out_vertices))
            adj_list[vertex.id] = [in_ids, out_ids]
        return adj_list

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
                self.remove_vertex(vertex, False)
        
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
    

    def remove_vertex(self, vertex: Vertex, maintain_deps: bool = True) -> None:
        """
        Removes the given vertex from the eDAG. If `maintain_deps` is
        set to True, all dependencies of the removed vertex are maintained.
        For instance, if A -> B -> C, and B is removed, then the graph
        after B's removal would be A -> C. If `maintain_deps` is False,
        there will be no dependency between vertex A and C.
        """
        # Removes vertex v from the dictionary
        del self.id_to_vertex[vertex.id]

        # Retrieves the set of vertices on which v depends
        # and the set of vertices that depends on v
        assert(vertex in self.adj_list)
        in_vertices, out_vertices = self.adj_list[vertex]
        for in_vertex in in_vertices:
            # Removes v from the out set of in_vertices
            self.adj_list[in_vertex][EDag._out].remove(vertex)
        for out_vertex in out_vertices:
            # Removes v from the in set of out_vertices
            self.adj_list[out_vertex][EDag._in].remove(vertex)
        if maintain_deps:
            # Adds an edge between each pair of in_vertex and
            # out_vertex if `maintain_deps` is True
            for in_vertex in in_vertices:
                for out_vertex in out_vertices:
                    self.add_edge(in_vertex, out_vertex)
        # Removes vertex v from the adjacency list and the vertex set entirely
        del self.adj_list[vertex]
        assert(vertex in self.vertices)
        self.vertices.remove(vertex)

    def remove_subgraph(self, subgraph: EDag) -> None:
        """
        Removes all vertices that belong to the given subgraph in the eDAG
        """
        for vertex_to_remove in subgraph.vertices:
            self.remove_vertex(vertex_to_remove)

    def get_subgraph(self, vertex: Vertex) -> EDag:
        """
        Given a vertex v, returns a sub-eDAG that contains all nodes
        that are connected to v. Note that "connected" in this case means
        all vertices that have a valid path to v and all vertices to which
        v has a valid path.
        """
        assert(vertex in self.vertices and vertex in self.adj_list)
        sub_eDag = EDag()

        curr = vertex
        visited = set()
        to_visit = { curr }

        # Builds a set that contains all the vertices connected to the
        # given vertex v
        while len(to_visit) > 0:
            curr = to_visit.pop()
            if curr in visited:
                continue
            in_vertices, out_vertices = self.adj_list[curr]
            to_visit.update(in_vertices)
            to_visit.update(out_vertices)
            visited.add(curr)

        id_to_vertex_copy = self.id_to_vertex.copy()
        adj_list_copy = self.adj_list.copy()
        unvisited = self.vertices.difference(visited)
        for vertex in unvisited:
            del id_to_vertex_copy[vertex.id]
            del adj_list_copy[vertex]
        
        sub_eDag.vertices = visited
        sub_eDag.id_to_vertex = id_to_vertex_copy
        sub_eDag.adj_list = adj_list_copy

        return sub_eDag
    
    def filter_vertices(self, cond: Callable[[Vertex], bool]) -> None:
        """
        Given a certain condition, iteratively removes all nodes from the eDAG 
        which do not satisfy the condition, i.e. when called, the function
        returns False. After a vertex is removed, the dependencies of its
        predecessors and successors are maintained. For instance, if A -> B,
        B -> C, and B is removed, then the edge A -> C will be added.
        """
        while True:
            # Iteratively removes all the vertices that
            # do not satisfy the given condition.
            changed = False

            tmp_vertices = self.vertices.copy()
            for vertex in tmp_vertices:
                # If condition is not satisfied
                if not cond(vertex):
                    self.remove_vertex(vertex)
                    changed = True

            # Will break out of the loop if no vertex is removed
            # in this iteration
            if not changed:
                break

    def get_depth(self) -> int:
        """
        Calculates the depth, i.e. in this case diameter of the eDAG by 
        exploring the graph from each starting vertex to each end vertex
        and keeping track of the longest distance.
        """
        # Calculates the depth using an iterative and DP approach
        # to guarantee maximum efficiency and to avoid stack overflow
        dp = defaultdict(int)
        adj_list = self.get_vertex_id_adj_list()
        end_vertices = self.get_end_vertices(True)
        prev_level = set()
        curr_level = set()
        for curr in end_vertices:
            # Marks the depth of the end vertex as 1
            dp[curr] = 1
            in_vertices, _ = adj_list[curr]
            curr_level.update(in_vertices)
        # Visits the eDAG from bottom to top
        while curr_level:
            # Collects vertices from the level above while
            # traversing through the current level
            for curr in curr_level:
                in_vertices, out_vertices = adj_list[curr]
                for out_vertex in out_vertices:
                    # Using `if` is faster than `max()`
                    new_val = 1 + dp[out_vertex]
                    # dp[curr] = max(dp[curr], 1 + dp[curr])
                    dp[curr] = dp[curr] if dp[curr] > new_val else new_val
                # Adds all `in_vertices` to the level above since
                # they will be visited in the next iteration
                prev_level.update(in_vertices)
            curr_level = prev_level
            prev_level = set()

        # Iterates through the starting node to find the maximum depth
        depth = 0
        for start_vertex in self.get_starting_vertices(True):
            depth = max(depth, dp[start_vertex])
        return depth
    
    def get_work(self, cond: Optional[Callable[[Vertex], bool]] = None) -> int:
        """
        Counts the number of vertices which satisfy the given condition. If
        a condition is not provided, the output will simply be the total number
        of vertices in the eDAG.
        """
        if cond is None:
            return len(self.vertices)

        # FIXME: Can probably use Python `filter()`
        work_count = 0
        for vertex in self.vertices:
            if cond(vertex):
                work_count += 1
        return work_count

    def split_disjoint_subgraphs(self) -> None:
        """
        Splits the eDAG into disjoint subgraphs. After invoking this function,
        the disjoint subgraphs in this eDAG can be accessed in the list
        `self.disjoint_subgraphs`.
        
        === WARNING ===
        
        1. Note that invoking this function will create an additional copy of 
        every vertex in the graph, it is recommended that this function be
        called after the eDAG has been sanitized or simplified.
        2. After subgraphs have been partitioned, invoking the function
        `self.remove_subgraph()` will render the list of disjoint subgraphs
        in valid, and this function will need to be called again to
        re-partition the eDAG.
        """
        # Clears the current list of disjoint subgraphs
        self.disjoint_subgraphs = []
        visited = set()
        # Constructs a subgraph for each starting node, i.e. vertices
        # with in-degree 0
        for start_vertex in self.get_starting_vertices():
            if start_vertex not in visited:
                subgraph = self.get_subgraph(start_vertex)
                visited.update(subgraph.vertices)
                # Adds the subgraph to self.disjoint_subgraphs
                self.disjoint_subgraphs.append(subgraph)

    def to_asm(self, full_str: bool = True) -> List[str]:
        """
        Converts the entire eDAG back into a list of assembly instructions
        according to the sequential order in which they were parsed.
        If `full_str`is set to True, the instructions returned will be
        the same as they were stored in the trace file, i.e. including
        cpu, instruction address, etc.
        Returns a list containing all the instructions.
        """
        # Sorts the vertices by their IDs so that the order in which
        # they were parsed / added
        sorted_vertices = self.sorted_vertices
        if full_str:
            asm = [vertex.full_str() for vertex in sorted_vertices]
        else:
            asm = [vertex.asm for vertex in sorted_vertices]
        return asm

    def visualize(self, highlight_mem_acc: bool = True, 
                large_graph_thresh: int = 1000) -> Digraph:
        """
        Converts the eDAG to an graphviz.Digraph, which can then be
        rendered and saved as a PDF file.
        @param highlight_mem_acc: If set to True, vertices that perform memory
        accesses will be highlighted.
        @param large_graph_thresh: An integer which marks the threshold above
        which a graph will be considered large and the layout engine used for
        graphviz will be changed to sfdp instead of dot
        """
        # Uses sfdp as the layout engine for large graph
        engine = "sfdp" if len(self.vertices) > large_graph_thresh else None
        graph = Digraph(engine=engine, graph_attr={"overlap": "scale"})
        for vertex in self.vertices:
            if highlight_mem_acc and vertex.is_mem_acc:
                if vertex.op_type == OpType.LOAD_MEM and vertex.cache_hit:
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