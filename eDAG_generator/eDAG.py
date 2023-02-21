from __future__ import annotations
from enum import Enum, auto
from tqdm import tqdm
import numpy as np
import bz2
import networkx as nx
from array import array
import igraph
import pickle
from collections import defaultdict
from multiprocessing import Pool, Process, Manager, JoinableQueue
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
                insn_addr: Optional[str] = None,
                data_addr: Optional[str] = None,
                data_size: int = 0) -> None:
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
        # Estimated number of cycles that will take for the instruction
        # contained in this vertex to finish based on the pre-defined
        # CPU model. It is set to None at initialization time
        self.cycles = None
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
        self.vertices: Set[int] = set()
        # A dictionary that maps the ID of the vertex to the
        # corresponding object. This is only present for
        # easier access to specific vertex
        self.id_to_vertex: Dict[int, Vertex] = {}
        # An adjacency list that maps each vertex to two sets
        # of vertices. The first set will be the vertices
        # on which it depends, while the second set will be
        # the vertices it points to, i.e. vertices that depend on it
        # Note that for performance reasons, only the IDs of the
        # vertices will appear this dictionary
        self.adj_list: Dict[int, List[Set[int]]] = {}
        
        # A list of disjoint subgraphs contained in the eDAG.
        # Note that this list will not be filled automatically while
        # constructing the eDAG. split_disjoint_subgraphs() have to
        # be invoked to retrieve them
        self.disjoint_subgraphs: List[EDag] = []
        # Keeps track of all vertices that have been removed
        self.removed_vertices = set()

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
            self.vertices.add(vertex.id)
            self.id_to_vertex[vertex.id] = vertex

        if vertex not in self.adj_list:
            self.adj_list[vertex.id] = [set(), set()]

    def add_edge(self, source: Vertex, target: Vertex) -> None:
        """
        Adds a connection between the given source and target vertices,
        and the relationship between the vertices is that the target vertex
        depends on the source, as in the source precedes the target.
        """
        # Ensures that both the source and target vertices have already
        # been added to the eDAG
        assert(source.id in self.vertices)
        assert(target.id in self.vertices)
        self.adj_list[source.id][EDag._out].add(target.id)
        self.adj_list[target.id][EDag._in].add(source.id)

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

    def edges(self, use_str: bool = True) \
        -> Union[List[Tuple[str, str]], List[Tuple[int, int]]]:
        """
        Converts the adjacency list to a list of directed edges between 
        two vertices. Note that vertices are represented with their IDs.
        For instance, an edge of ('1', '2') denotes a directed edge from 
        vertex 1 to vertex 2.
        @param use_str: If True, will return the vertex IDs as strings.
        """
        edges = []
        for source, (_, out_vertices) in self.adj_list.items():
            for target in out_vertices:
                if use_str:
                    edges.append((f"{source}", f"{target}"))
                else:
                    edges.append((source, target))
        return edges

    def remove_unconnected_vertices(self) -> None:
        """
        Removes all the unconnected vertices that have no dependencies
        from the eDAG. "No dependencies" means that a vertex does not
        have any edges connected to it, i.e. both in and out degrees of the
        vertex are 0.
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
            res[vertex] = [len(in_vertices), len(out_vertices)]
        return res
    
    def topological_sort(self, reverse: bool = False) -> List[int]:
        """
        Returns one topological sort of the vertices of an eDAG in a list
        containing their IDs.
        Implementation from
        https://www.geeksforgeeks.org/python-program-for-topological-sorting/
        TODO Re-indexing of the Python library is annoying and it causes
        an error in this function when vertices are removed.
        @param reverse: If True, all predecessors will be on the right as
        opposed to on the left.
        """
        if len(self.removed_vertices) > 0:
            raise RuntimeError("[ERROR] Topological sort can only be computed if no vertices have been removed")
        # Uses the igraph library to get the topological sort
        graph = igraph.Graph(edges=self.edges(False), directed=True)
        # graph.add_vertices(map(str, self.vertices))
        # graph.add_edges(self.edges())
        # vertices_to_remove = [v.index for v in graph.vs if v.index not in self.vertices]
        # graph.delete_vertices(vertices_to_remove)
        path = graph.topological_sorting(mode="in" if reverse else "out")
        assert(len(path) == len(self.vertices))
        return path

    def remove_vertex(self, vertex: Vertex, maintain_deps: bool = True) -> None:
        """
        Removes the given vertex from the eDAG. If `maintain_deps` is
        set to True, all dependencies of the removed vertex are maintained.
        For instance, if A -> B -> C, and B is removed, then the graph
        after B's removal would be A -> C. If `maintain_deps` is False,
        there will be no dependency between vertex A and C.
        """
        vertex_id = vertex.id
        # Removes vertex v from the dictionary
        del self.id_to_vertex[vertex_id]
        
        # Retrieves the set of vertices on which v depends
        # and the set of vertices that depends on v
        assert(vertex_id in self.adj_list)
        in_vertices, out_vertices = self.adj_list[vertex_id]
        for in_vertex in in_vertices:
            # Removes v from the out set of in_vertices
            self.adj_list[in_vertex][EDag._out].remove(vertex_id)
        for out_vertex in out_vertices:
            # Removes v from the in set of out_vertices
            self.adj_list[out_vertex][EDag._in].remove(vertex_id)
        if maintain_deps:
            # Adds an edge between each pair of in_vertex and
            # out_vertex if `maintain_deps` is True
            for in_id in in_vertices:
                in_vertex = self.id_to_vertex[in_id]
                for out_id in out_vertices:
                    out_vertex = self.id_to_vertex[out_id]
                    self.add_edge(in_vertex, out_vertex)
        # Removes vertex v from the adjacency list and the vertex set entirely
        del self.adj_list[vertex_id]
        assert(vertex_id in self.vertices)
        self.vertices.remove(vertex_id)
        self.removed_vertices.add(vertex_id)

    def remove_subgraph(self, subgraph: EDag) -> None:
        """
        Removes all vertices that belong to the given subgraph in the eDAG
        """
        for vertex_id in subgraph.vertices:
            vertex_to_remove = self.id_to_vertex[vertex_id]
            self.remove_vertex(vertex_to_remove)

    def get_subgraph(self, vertex: Vertex) -> EDag:
        """
        Given a vertex v, returns a sub-eDAG that contains all nodes
        that are connected to v. Note that "connected" in this case means
        all vertices that have a valid path to v and all vertices to which
        v has a valid path.
        """
        vertex_id = vertex.id
        assert(vertex_id in self.vertices and vertex_id in self.adj_list)
        sub_eDag = EDag()

        curr = vertex_id
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
            del id_to_vertex_copy[vertex]
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
            for vertex_id in tmp_vertices:
                vertex = self.id_to_vertex[vertex_id]
                # If condition is not satisfied
                if not cond(vertex):
                    self.remove_vertex(vertex)
                    changed = True

            # Will break out of the loop if no vertex is removed
            # in this iteration
            if not changed:
                break
    
    def get_longest_path(self, id_only: bool = True) -> List[int]:
        """
        Computes the longest path in the eDAG in a list of consecutive vertices
        that should be followed.
        @param id_only: If True, only IDs of the vertices will be returned.
        """
        depth, dp = self.get_depth(True)
        path = []
        curr = None
        curr_depth = 0
        # Finds the vertex that starts the longest path
        for vertex in self.get_starting_vertices(True):
            if dp[vertex] > curr_depth:
                curr = vertex
                curr_depth = dp[vertex]
        assert curr is not None
        path.append(curr)
        # Follows the starting node until the end is reached
        while True:
            _, out_vertices = self.adj_list[curr]
            curr_depth = -1
            for out_vertex in out_vertices:
                if dp[out_vertex] > curr_depth:
                    curr = out_vertex
                    curr_depth = dp[out_vertex]
            if curr_depth == -1:
                # Reached the end
                break
            else:
                # Appends the current vertex to the path
                path.append(curr if id_only else self.id_to_vertex[curr])
        assert depth == len(path)
        return path
    
    def get_vertex_rank(self) -> Dict[int, Set[int]]:
        """
        Calculates the maximum distance between every vertex and an
        arbitrary starting vertex in the eDAG, which is called its rank,
        i.e. the depth of the vertex.
        Returns a dictionary that maps each rank to a set of vertices
        which belong to that rank. Note that only the IDs of the
        vertices will be contained in the dictionary.
        FIXME The term rank comes from how layout engines are implemented and 
        is probably not the best term in this case.
        """
        # Computes the topological sort of the vertices
        topo_sorted = self.topological_sort(reverse=False)
        # Uses an array to keep track of the rank of each vertex
        dp = array('L', [0] * len(topo_sorted))
        for vertex in topo_sorted:
            _, out_vertices = self.adj_list[vertex]
            for out in out_vertices:
                new_val = dp[vertex] + 1
                dp[out] = dp[out] if dp[out] > new_val else new_val
        
        res = defaultdict(set)
        # Iterates through the dp array to construct the sets
        # containing vertices belonging to a specific rank
        for vertex, rank in enumerate(dp):
            res[rank].add(vertex)
        
        return res


    def get_depth(self, return_dp: bool = False) \
         -> Union[int, Tuple[int, array]]:
        """
        Calculates the depth, i.e. in this case diameter of the eDAG, by 
        exploring the graph from each starting vertex to each end vertex
        and keeping track of the longest distance.
        @param return_dp: If True, the array that is used to store the
        intermediate values during the depth computation will also be
        returned in a tuple along with the depth itself.
        """
        # Calculates the depth using an iterative and DP approach
        # to guarantee maximum efficiency and to avoid stack overflow
        # dp = defaultdict(int)

        # Computes the topologically sorted list of vertices
        topo_sorted = self.topological_sort(reverse=True)
        # array.array is faster than np.array
        dp = array('L', [0] * len(topo_sorted))
        depth = 0

        for vertex in tqdm(topo_sorted):
            _, out_vertices = self.adj_list[vertex]
            for out_vertex in out_vertices:
                new_val = dp[out_vertex] + 1
                # `if` statement is faster than `max()`
                dp[vertex] = dp[vertex] if dp[vertex] > new_val else new_val
            depth = depth if depth > dp[vertex] else dp[vertex]
        
        if return_dp:
            return (depth + 1, dp)
        
        return depth + 1

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
        for vertex_id in self.vertices:
            if cond(self.id_to_vertex[vertex_id]):
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

    def save(self, save_path: str, use_compression: bool = False) -> None:
        """
        Persists the current eDAG object to the given path
        through serialization.
        """
        if not use_compression:
            # Saves the eDAG to a pickle file
            with open(save_path, "wb") as pickle_file:
                pickle.dump(self, pickle_file)
        else:
            # Uses bz2 to compress the data file
            with bz2.BZ2File(save_path, "wb") as compressed_file:
                pickle.dump(self, compressed_file)

    @staticmethod
    def load(save_path: str, use_compression: bool = False) -> EDag:
        """
        Loads the persisted eDAG object from the given path.
        """
        if not use_compression:
            with open(save_path, "rb") as pickle_file:
                return pickle.load(pickle_file)
        else:
            with bz2.BZ2File(save_path, "rb") as compressed_file:
                return pickle.load(compressed_file)
