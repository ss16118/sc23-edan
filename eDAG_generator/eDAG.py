from __future__ import annotations
from enum import Enum, auto
from tqdm import tqdm
import numpy as np
import bz2
import networkx as nx
from array import array
import igraph
import pickle
from functools import lru_cache
from collections import defaultdict
from multiprocessing import Pool, Process, Manager, JoinableQueue
from typing import List, Dict, Optional, Set, Tuple, Callable, Union


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
    # Jump
    JUMP = auto()
    # Return
    RETURN = auto()
    # Uncategorized operation
    UNCATEGORIZED = auto()
    

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
                imm_val: Optional[int] = None,
                sec_target: Optional[str] = None,
                cpu: Optional[int] = None,
                data_addr: Optional[str] = None,
                data_size: int = 0) -> None:
        """
        @param id: A unique number given to each vertex by the parser.
        @param opcode: Opcode of the assembly instruction, e.g. add, sub, mv.
        @param operands: A list of operands represented as strings.
        @param target: Target register or memory location, i.e. the register 
        or memory location whose value is updated. It can be None for 
        certain instructions. As an example, the target for `sd a1,-64(s0)` is
        the memory location `-64(s0)`.
        @param dependencies: A set of registers or memory locations on which
        this vertex depends. As an example, the command `sd a1,-64(s0)` depends
        on the registers a1 and s0, whereas the command `ld a5,-40(s0)` depends
        on the memory location denoted by `-40(s0)`.
        @param op_type: Type of operation performed in this vertex.
        @param is_comm_assoc: Indicates whether the operation performed by
        the instruction is both commutative and associative. Used by
        the subgraph optimizer.
        @param imm_val: Immediate value, only exists if the instruction
        has a operand that is an immediate value.
        @param sec_target: Second target.
        @param cpu: ID of the CPU on which the instruction was executed.
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
        self.operands = operands
        self.target = target
        self.dependencies = dependencies
        self.op_type = op_type
        self.is_comm_assoc = is_comm_assoc

        self.imm_val = imm_val
        # The second target of the instruction, only present
        # for very few instructions
        self.sec_target = sec_target
        self.cpu = cpu
        # self.insn_addr = insn_addr
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
    
    def has_same_insn(self, v: Vertex) -> bool:
        """
        Returns true if the given vertex contains the exact
        same instructions (i.e. opcode and operands) as the
        current object, False otherwise.
        """
        return self.opcode == v.opcode and self.operands == v.operands

    def __hash__(self) -> int:
        # return hash((self.id, self.opcode, self.operands))
        return hash(self.id)


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
        # Dependencies between memory access vertices induced by
        # limited number of memory request issue slots
        self.mem_slot_deps = {}

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

    def add_edge(self, source: int, target: int) -> None:
        """
        Adds a connection between the given source and target vertices,
        and the relationship between the vertices is that the target vertex
        depends on the source, as in the source precedes the target.
        """
        # Ensures that both the source and target vertices have already
        # been added to the eDAG
        assert(source in self.vertices and target in self.vertices)
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
    
    # Uses caching to make sure that consecutive calls to this
    # function can be executed quickly
    @lru_cache(maxsize=2)
    def topological_sort(self, reverse: bool = False,
                         mem_acc_only: bool = False) -> List[int]:
        """
        Returns one topological sort of the vertices of an eDAG in a list
        containing their IDs.
        TODO Re-indexing of the Python library is annoying and it causes
        an error in this function when vertices have been removed.
        @param reverse: If True, all predecessors will be on the right as
        opposed to on the left.
        @param mem_acc_only: If True, the returned list will only
        contain memory access vertices.
        """
        if len(self.removed_vertices) > 0:
            raise RuntimeError("[ERROR] Topological sort can only be computed if no vertices have been removed")
        # Uses the igraph library to get the topological sort
        graph = igraph.Graph(n=len(self.vertices), edges=self.edges(False), directed=True)

        if self.mem_slot_deps:
            graph.add_edges(self.mem_slot_deps.items())
        # graph.add_vertices(map(str, self.vertices))
        # graph.add_edges(self.edges())
        # vertices_to_remove = [v.index for v in graph.vs if v.index not in self.vertices]
        # graph.delete_vertices(vertices_to_remove)
        path = graph.topological_sorting(mode="in" if reverse else "out")
        assert(len(path) == len(self.vertices))

        if mem_acc_only:
            # Filters out non-memory access vertices
            path = [v for v in path if self.id_to_vertex[v].is_mem_acc and \
                        not self.id_to_vertex[v].cache_hit]
        return path

    def limit_issue_slots(self, num_slots: Optional[int] = None) -> None:
        """
        Traverses through all the memory access vertices and determine the
        dependencies between them as per the given number of memory issue
        slots as well as the time at which each of them is executed. For
        example, if vertex 1, 2, and 3 are executed in the same time frame,
        and there are only two issue slots, then vertex 3 will have to
        be dependent on vertex 1.
        
        If `num_slots` is None, it will assume
        that infinite issue slots are present and previously generated
        memory slot dependencies will be removed.
        """
        # Clears the issue slot dependencies
        self.mem_slot_deps.clear()

        if num_slots is None:
            return
        
        num_iter = 0
        prev_dp = None

        # Obtains the topological sort of the memory
        # access vertices in the eDAG
        mem_vertices = self.topological_sort(False, True)

        # Repeats until no changes occur
        while True:
            _, dp = self.get_vertex_depth()
            num_iter += 1
            if prev_dp == dp:
                break
            prev_dp = dp

            slots = [None for _ in range(num_slots)]
            # Keeps track of the time at which the request
            # at each slot will be finished
            slot_time = [None for _ in range(num_slots)]
            curr_slot = 0
            
            for v_id in mem_vertices:
                vertex = self.id_to_vertex[v_id]
                if slots[curr_slot] is not None:
                    # Checks if the previous slot can be freed
                    if slot_time[curr_slot] >= dp[v_id]:
                        # If not, creates a dependency between the memory vertices
                        self.mem_slot_deps[slots[curr_slot]] = v_id

                slots[curr_slot] = v_id
                slot_time[curr_slot] = dp[v_id] + vertex.cycles
                curr_slot = (curr_slot + 1) % num_slots
        
        print(f"[DEBUG] Number of iterations: {num_iter}")
    
    def get_vertex_depth(self) -> Tuple[float, array]:
        """
        Returns the depth of the graph as well as an array that contains
        the maximum number of cycles it takes to reach each vertex from
        any starting vertex.

        For instance, in an eDAG where there are three vertices
        0 -> 1, 2 -> 1, meaning that vertex 1 depends both on vertex 0 and
        vertex 2. The number of CPU cycles it takes to compute the vertices
        are [10, 20, 5]. Then, the returned array will be [0, 10, 0],
        since vertex 1 needs wait for both vertex 0 and 2 to complete.
        """
        topo_sorted = self.topological_sort(reverse=False)
        res = array('f', [0] * len(topo_sorted))
        depth = 0
        for v_id in topo_sorted:
            _, out_vertices = self.adj_list[v_id]
            vertex = self.id_to_vertex[v_id]
            new_val = res[v_id] + vertex.cycles
            for out in out_vertices:
                res[out] = res[out] if res[out] > new_val else new_val
            
            if v_id in self.mem_slot_deps:
                out = self.mem_slot_deps[v_id]
                res[out] = res[out] if res[out] > new_val else new_val
            
            depth = depth if depth > new_val else new_val
        
        return depth, res

    def remove_edge(self, source: int, target: int) -> None:
        """
        Removes the edge between the vertices whose IDs are
        specified by `source` and `target`.
        """
        assert(source in self.vertices and target in self.vertices)
        # Removes the target vertex from the out-set of the source
        _, src_outs = self.adj_list[source]
        src_outs.remove(target)
        # Removes the source vertex from the in-set of the target
        tar_ins, _ = self.adj_list[target]
        tar_ins.remove(source)

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
    
    def get_vertices(self, cond: Callable[[Vertex], bool],
                     id_only: bool = True) -> Set[Union[int, Vertex]]:
        """
        Retrieves all vertices in the eDAG that satisfy the given condition.
        Returns the result as a set. If `id_only` is True, only the IDs
        of the vertices will be returned.
        """
        res = set()
        # FIXME Can probably use higher-order functions like `map()`
        # and `reduce()`
        for vertex_id in self.vertices:
            vertex = self.id_to_vertex[vertex_id]
            if cond(vertex):
                res.add(vertex_id if id_only else vertex)
        return res

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
    
    def get_longest_path(self, id_only: bool = True,
                         mem_acc_only: bool = False,
                         dp: Optional[array] = None) -> Tuple[float, List[int]]:
        """
        Computes the longest path in the eDAG in a list of consecutive vertices
        that should be followed.
        @param id_only: If True, only IDs of the vertices will be returned.
        @param mem_acc_only: If True, only non-cache-hit memory access vertices
        will be taken into account when computing the longest path.
        @param dp: If present will forgo the invocation of `get_depth()`.
        
        @return a tuple containing two items, the first item being 
        a float representing the depth and the second item is a list containing
        the vertices of the longest path.
        """
        if dp is None:
            depth, dp = self.get_depth(True, mem_acc_only)
        
        path = []
        curr = None
        curr_depth = 0
        # Finds the vertex that starts the longest path
        # FIXME Can probably use reduce
        for vertex in self.get_starting_vertices(True):
            if dp[vertex] > curr_depth:
                curr = vertex
                curr_depth = dp[vertex]
        assert curr is not None
        depth = curr_depth
        path.append(curr)
        # Follows the starting node until the end is reached
        while True:
            _, out_vertices = self.adj_list[curr]
            curr_depth = -1
            if curr in self.mem_slot_deps:
                target = self.mem_slot_deps[curr]
                if dp[target] > curr_depth:
                    curr = target
                    curr_depth = dp[target]
            
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
            
        return depth, path

    def get_vertex_rank(self, interval: float = 1) -> Dict[int, Set[int]]:
        """
        Calculates the maximum distance between every vertex and an
        arbitrary starting vertex in the eDAG, which is called its rank,
        i.e. the depth of the vertex.
        @param interval: The time interval to use in CPU cycles to split
        the critical path into different ranks.

        @return a dictionary that maps each rank to a set of vertices
        which belong to that rank. Note that only the IDs of the
        vertices will be contained in the dictionary.
        FIXME The term rank comes from how layout engines are implemented and 
        is probably not the best term in this case.
        """
        res = defaultdict(set)
        _, dp = self.get_vertex_depth()
        
        for v_id, cycles in enumerate(dp):
            rank = cycles // interval if interval != 1 else cycles
            # Adds the vertex ID to a set that contains
            # all vertices belonging to the same rank
            res[rank].add(v_id)
        
        return res

    def get_depth(self, return_dp: bool = False, mem_acc_only: bool = False) \
         -> Union[int, Tuple[int, array]]:
        """
        Calculates the depth, i.e. in this case diameter of the eDAG, by 
        exploring the graph from each starting vertex to each end vertex
        and keeping track of the longest distance. Note that
        if a CPU model has been used, each vertex will be weighted
        by its corresponding number of CPU cycles. For instance, 
        if a vertex takes 200 cycles to execute, it will contribute
        to the length of the path by 200 units as opposed to only 1 unit.

        @param return_dp: If True, the array that is used to store the
        intermediate values during the depth computation will also be
        returned in a tuple along with the depth itself.
        @param mem_acc_only: If True, will only consider the number of
        non-cache-hit memory access vertices when calculating the depth.
        """
        # Calculates the depth using an iterative and DP approach
        # to guarantee maximum efficiency and to avoid stack overflow
        # dp = defaultdict(int)

        # Computes the topologically sorted list of vertices
        topo_sorted = self.topological_sort(reverse=True)
        # array.array is faster than np.array
        dp = array('f', [0] * len(topo_sorted))
        depth = 0

        # TODO Refactor code
        if not mem_acc_only:
            for vertex_id in tqdm(topo_sorted):
                vertex = self.id_to_vertex[vertex_id]
                cycles = vertex.cycles
                _, out_vertices = self.adj_list[vertex_id]
                if not out_vertices and \
                    vertex_id not in self.mem_slot_deps:
                    dp[vertex_id] = cycles
                else:
                    for out_vertex in out_vertices:
                        new_val = dp[out_vertex] + cycles
                        # `if` statement is faster than `max()`
                        dp[vertex_id] = \
                            dp[vertex_id] if dp[vertex_id] > new_val else new_val
                    
                    if vertex_id in self.mem_slot_deps:
                        new_val = dp[self.mem_slot_deps[vertex_id]] + cycles
                        dp[vertex_id] = \
                            dp[vertex_id] if dp[vertex_id] > new_val else new_val

                depth = depth if depth > dp[vertex_id] else dp[vertex_id]

            if return_dp:
                return (depth, dp)
            
            return depth
        else:
            for vertex_id in tqdm(topo_sorted):
                vertex = self.id_to_vertex[vertex_id]
                val = int(vertex.is_mem_acc and not vertex.cache_hit)
                _, out_vertices = self.adj_list[vertex_id]
                # If the vertex is an end vertex that has no outgoing edges
                if not out_vertices and vertex_id not in self.mem_slot_deps:
                    dp[vertex_id] = val
                else:
                    for out_vertex in out_vertices:
                        new_val = dp[out_vertex] + val
                        dp[vertex_id] = \
                            dp[vertex_id] if dp[vertex_id] > new_val else new_val
                    if vertex_id in self.mem_slot_deps:
                        new_val = dp[self.mem_slot_deps[vertex_id]] + val
                        dp[vertex_id] = \
                            dp[vertex_id] if dp[vertex_id] > new_val else new_val
                
                depth = depth if depth > dp[vertex_id] else dp[vertex_id]

            if return_dp:
                return (depth, dp)

            return depth

    def get_work(self, cond: Optional[Callable[[Vertex], bool]] = None) -> int:
        """
        Counts the number of vertices which satisfy the given condition. If
        a condition is not provided, the output will simply be the total number
        of vertices in the eDAG.

        FIXME Can probably use higher-order functions, e.g. map, filter
        """
        if cond is None:
            work = 0
            for vertex_id in self.vertices:
                vertex = self.id_to_vertex[vertex_id]
                work += vertex.cycles
            return work

        work_count = 0
        for vertex_id in self.vertices:
            vertex = self.id_to_vertex[vertex_id]
            if cond(vertex):
                work_count += vertex.cycles
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
    
    @staticmethod
    def merge_subgraphs(subgraphs: List[EDag], disjoint: bool = True) -> EDag:
        """
        Creates a new `EDag` object from all the vertices and
        edges of the given list of subgraphs.
        @param disjoint: If True, would assume that the given subgraphs are
        all disjoint, meaning that they do not share vertices or edges, and
        this would significantly accelerate the merging process.
        """
        merged = EDag()

        for subgraph in subgraphs:
            # Merges vertices
            merged.vertices.update(subgraph.vertices)
            merged.id_to_vertex.update(subgraph.id_to_vertex)
            # Merges edges
            if disjoint:
                merged.adj_list.update(subgraph.adj_list)
            else:
                for v_id, (in_vs, out_vs) in subgraph.adj_list.items():
                    if v_id in merged.adj_list:
                        merged[v_id][EDag._in].update(in_vs)
                        merged[v_id][EDag._out].update(out_vs)
                    else:
                        merged[v_id] = [in_vs, out_vs]
        return merged