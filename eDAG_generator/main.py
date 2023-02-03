import os
import argparse
from edag_generator import EDagGenerator, ISA
from cache_model import SingleLevelSetAssociativeCache
from riscv_subgraph_optimizer import RiscvSubgraphOptimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trace file cleanup",
                                     description="`Clean up the instruction trace file produced by gdb")
    parser.add_argument("-f", "--trace-file", dest="trace_file_path",
                        help="File path to the instruction trace file")
    parser.add_argument("-g", "--graph-file", dest="graph_file", default=None,
                        help="Path to the visualization of the eDAG")
    parser.add_argument("-m", "--only-mem-acc", dest="only_mem_acc", default=False,
                        action='store_true',
                        help="If set, only vertices with memory accesses will be displayed")
    parser.add_argument("-l", "--highlight-mem-acc", dest="highlight_mem_acc",
                        default=False, action="store_true",
                        help="If set, memory access vertices will be highlighted as red")
    parser.add_argument("-r", "--remove-single-vertices", dest="remove_single_vertices",
                        default=False, action="store_true",
                        help="If set, single vertices without any connections will be removed")
    parser.add_argument("-s", "--simplified", dest="simplified", 
                        default=False, action="store_true",
                        help="Simplifies the eDAG to the bare minimum for theoretical work-depth analysis")
    parser.add_argument("-o", "--optimize", dest="optimize_subgraph",
                        default=False, action="store_true",
                        help="If set, will attempt optimize the eDAG in terms of work and depth according to pre-defined heuristics")
    parser.add_argument("-c", "--use-cache-model", dest="use_cache_model",
                        default=False, action="store_true",
                        help="If set, a predefined LRU cache model will be used")
    args = parser.parse_args()

    if args.trace_file_path is None:
        print("[ERROR] Path to the trace file must be provided")
        exit(-1)

    if args.graph_file is None:
        filename, file_extension = os.path.splitext(args.trace_file_path)
        graph_file = filename + "_eDAG"
    else:
        graph_file = args.graph_file
    
    if args.use_cache_model:
        cache = SingleLevelSetAssociativeCache()
    else:
        cache = None
    # Initializes eDAG generator
    generator = EDagGenerator(args.trace_file_path, ISA.RISC_V, args.only_mem_acc, 
                            args.remove_single_vertices, args.simplified,
                            cache_model=cache)
    eDag = generator.generate()
    if args.optimize_subgraph:
        optimizer = RiscvSubgraphOptimizer()
        optimizer.optimize(eDag)
    
    print(f"Work : {eDag.get_work()}")
    print(f"Depth: {eDag.get_depth()}")
    print(f"Parallelism: {eDag.get_work() / eDag.get_depth()}")
    
    graph = eDag.visualize(args.highlight_mem_acc)
    graph.render(graph_file, view=True)