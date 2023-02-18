import os
import argparse
import cProfile
from time import time
from edag_generator import EDagGenerator, ISA
from cache_model import SingleLevelSetAssociativeCache
from riscv_parser import RiscvParser
from metrics import *
from riscv_subgraph_optimizer import RiscvSubgraphOptimizer


insn_cycles = [
    # Memory load
    ({ "lb", "lh", "lw", "ld", "lhu", "lbu", "fld", "flw" }, 200.0),
    # Load immediate
    ({ "li", "lui" }, 1.0),
    (RiscvParser.store_instructions, 200.0),
    (RiscvParser.mv_instructions, 1.0),
    (RiscvParser.add_instructions, 1.0),
    (RiscvParser.sub_instructions, 1.0),
    (RiscvParser.mul_instructions, 8.0),
    (RiscvParser.div_instructions, 20.0),
    (RiscvParser.rem_instructions, 20.0),
    (RiscvParser.atomic_op_instructions, 1.0),
    (RiscvParser.comp_and_set_instructions, 1.0),
    (RiscvParser.bit_op_instructions, 1.0),
    (RiscvParser.uncond_jump_instructions, 3.0),
    (RiscvParser.cond_br_2ops_instructions, 3.0),
    (RiscvParser.cond_br_3ops_instructions, 3.0),
    (RiscvParser.fp_2ops_instructions, 1.0),
    (RiscvParser.fp_3ops_instructions, 20.0),
    (RiscvParser.fp_4ops_instructions, 1.0),
]

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
    parser.add_argument("--reuse-histogram", dest="reuse_histogram",
                        default=False, action="store_true",
                        help="If set, will generate the reuse distance histogram based on the given trace")
    parser.add_argument("--bandwidth", dest="calc_bandwidth",
                        default=False, action="store_true",
                        help="If set, will estimate the bandwidth utilization")
    parser.add_argument("-c", "--use-cache-model", dest="use_cache_model",
                        default=False, action="store_true",
                        help="If set, a predefined LRU cache model will be used")
    args = parser.parse_args()

    if args.trace_file_path is None:
        print("[ERROR] Path to the trace file must be provided")
        exit(-1)

    # if args.graph_file is None:
    #     filename, file_extension = os.path.splitext(args.trace_file_path)
    #     graph_file = filename + "_eDAG"
    # else:
    #     graph_file = args.graph_file
    
    if args.use_cache_model:
        cache = SingleLevelSetAssociativeCache()
    else:
        cache = None

    if args.reuse_histogram:
        # Generates the reuse distance histogram based on the given trace
        print("[INFO] Generating reuse distance histogram")
        reuse_distance = ReuseDistance(args.trace_file_path, RiscvParser())
        reuse_distance.plot_histogram("../examples/histo.pdf")

    # Initializes eDAG generator
    print("[INFO] Generating eDAG")
    generator = EDagGenerator(args.trace_file_path, ISA.RISC_V, args.only_mem_acc, 
                            args.remove_single_vertices, args.simplified,
                            cache_model=cache)
    eDag = generator.generate()
    # for asm in eDag.to_asm(False):
    #     print(asm)
    
    if args.optimize_subgraph:
        optimizer = RiscvSubgraphOptimizer()
        optimizer.optimize(eDag)
    
    if args.calc_bandwidth:
        bandwidth_util = BandwidthUtilization(eDag)
        bandwidth = bandwidth_util.compute_bandwidth(insn_cycles)
        print(f"Bandwidth utilization: {bandwidth / 10 ** 6} MB/s")

    print("[INFO] Calculating eDAG work")
    work = eDag.get_work()
    print(f"Work : {work}")
    print("[INFO] Calculating eDAG depth")
    # cProfile.run('depth = eDag.get_depth()')

    # start = time()
    depth = eDag.get_depth()
    # print(f"[DEBUG] Time taken: {time() - start}")
    print(f"Depth: {depth}")
    print(f"Parallelism: {work / depth}")
    # print(f"Sort: {eDag.topological_sort(reverse=True)}")
    if args.remove_single_vertices:
        eDag.remove_single_vertices()

    if args.graph_file is not None:
        graph = eDag.visualize(args.highlight_mem_acc)
        graph.render(args.graph_file, view=True)