from eDAG import EDag, Vertex


if __name__ == "__main__":
    # Example eDAG
    # 0 -> 1
    # 1 -> 2, 3
    # 2 -> 3, 4
    # 3 -> 4
    # 5 -> []
    eDag = EDag()
    v0 = Vertex(0, "sw", ["s0", "0(s1)"], None, set(), True)
    eDag.add_vertex(v0)
    v1 = Vertex(1, "li", ["s2", "1"], None, set(), False)
    eDag.add_vertex(v1)
    v2 = Vertex(2, "lw", ["s1", "0(s1)"], None, set(), True)
    eDag.add_vertex(v2)
    v3 = Vertex(3, "sw", ["s0", "0(s0)"], None, set(), True)
    eDag.add_vertex(v3)
    v4 = Vertex(4, "li", ["s1", "0"], None, set(), False)
    eDag.add_vertex(v4)
    v5 = Vertex(5, "sw", ["s0", "64(s1)"], None, set(), True)

    eDag.add_edge(v0, v1)
    eDag.add_edge(v1, v2)
    eDag.add_edge(v1, v3)
    eDag.add_edge(v2, v3)
    eDag.add_edge(v2, v4)
    eDag.add_edge(v3, v4)
    print(eDag.get_in_out_degrees())
    # eDag.remove_single_vertices()
    # eDag.filter_vertices(lambda v: v.is_mem_acc)

    graph = eDag.visualize()
    graph.render("../eDAG/test", view=True)