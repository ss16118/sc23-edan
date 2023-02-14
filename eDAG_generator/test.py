from eDAG import EDag, Vertex, OpType
from edag_sanitizer import *
from riscv_subgraph_optimizer import *

if __name__ == "__main__":
    # Example eDAG:
    # 1 -> 4
    # 2 -> 4
    # 3 -> 4
    # 4 -> 6
    # 5 -> 6
    # 6 -> 8
    # 7 -> 8
    # 8 -> 10
    # 9 -> 10
    # 10 -> []
    eDag = EDag()
    v1 = Vertex(1, "ld", ["s3", "0(s0)"], None, set(), OpType.LOAD_MEM)
    v2 = Vertex(2, "ld", ["s1", "0(s0)"], None, set(), OpType.LOAD_MEM)
    v3 = Vertex(3, "ld", ["s2", "0(s0)"], None, set(), OpType.LOAD_MEM)
    v4 = Vertex(4, "add", ["s1", "s1", "s2"], None, set(), OpType.ARITHMETIC, True)
    v5 = Vertex(5, "ld", ["s2", "0(s0)"], None, set(), OpType.LOAD_MEM)
    v6 = Vertex(6, "add", ["s1", "s1", "s2"], None, set(), OpType.ARITHMETIC, True)
    v7 = Vertex(7, "ld", ["s2", "0(s0)"], None, set(), OpType.LOAD_MEM)
    v8 = Vertex(8, "add", ["s1", "s1", "s2"], None, set(), OpType.ARITHMETIC, True)
    v9 = Vertex(9, "ld", ["s2", "0(s0)"], None, set(), OpType.LOAD_MEM)
    v10 = Vertex(10, "add", ["s1", "s1", "s2"], None, set(), OpType.ARITHMETIC, True)
    for v in [v2, v3, v4, v5, v6, v7, v8, v9, v10]:
        eDag.add_vertex(v)
    # eDag.add_edge(v1, v4)
    eDag.add_edge(v2, v4)
    eDag.add_edge(v3, v4)
    eDag.add_edge(v4, v6)
    eDag.add_edge(v5, v6)
    eDag.add_edge(v6, v8)
    eDag.add_edge(v7, v8)
    eDag.add_edge(v8, v10)
    eDag.add_edge(v9, v10)

    print(eDag.get_in_out_degrees())
    print(eDag.get_depth())
    # eDag.remove_single_vertices()
    # eDag.filter_vertices(lambda v: v.is_mem_acc)
    sanitizer = RiscvEDagSanitizer()
    sanitizer.sanitize_edag(eDag)

    optimizer = RiscvSubgraphOptimizer()
    optimizer.optimize(eDag)

    graph = eDag.visualize()

    graph.render("../eDAG/test", view=True)