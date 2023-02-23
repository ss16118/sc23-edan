import os
import argparse
import cProfile
from pathlib import Path
from time import time
from edag_generator import EDagGenerator, ISA
from cache_model import SingleLevelSetAssociativeCache
from cpu_model import CPUModel
from riscv_parser import RiscvParser
from metrics import *
from utils import *
from riscv_subgraph_optimizer import RiscvSubgraphOptimizer



compress = True

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
    parser.add_argument("-l", "--load-file", dest="load_path", default=None,
                        help="If set, will try to load the eDAG object directly from the given file")
    parser.add_argument("-g", "--graph-file", dest="graph_file", default=None,
                        help="Path to which the visualization of the eDAG will be saved")
    parser.add_argument("-m", "--only-mem-acc", dest="only_mem_acc",
                        default=False, action='store_true',
                        help="If set, only vertices with memory accesses will be displayed")
    parser.add_argument("--highlight-mem-acc", dest="highlight_mem_acc",
                        default=False, action="store_true",
                        help="If set, memory access vertices will be highlighted")
    parser.add_argument("-r", "--remove-unconnected-vertices", dest="remove_unconnected_vertices",
                        default=False, action="store_true",
                        help="If set, unconnected vertices without any connections will be removed")
    parser.add_argument("-s", "--save-path", dest="save_path", default=None,
                        help="If set, will save the original generated eDAG to the given path")
    parser.add_argument("--sanitize", dest="sanitize", 
                        default=False, action="store_true",
                        help="If set, will try to simplify the eDAG")
    parser.add_argument("-o", "--optimize", dest="optimize_subgraph",
                        default=False, action="store_true",
                        help="If set, will attempt optimize the eDAG in terms of work and depth according to pre-defined heuristics")
    parser.add_argument("--reuse-histogram", dest="reuse_histogram",
                        default=False, action="store_true",
                        help="If set, will generate the reuse distance histogram based on the given trace")
    parser.add_argument("--bandwidth", dest="calc_bandwidth",
                        default=False, action="store_true",
                        help="If set, bandwidth related metrics will be computed")
    parser.add_argument("--cpu", dest="use_cpu_model",
                        default=False, action="store_true",
                        help="If set, a predefined CPU model will be used")
    parser.add_argument("--cache", dest="use_cache_model",
                        default=False, action="store_true",
                        help="If set, a predefined LRU cache model will be used")
    parser.add_argument("--work-depth", dest="calc_work_depth",
                        default=False, action="store_true",
                        help="If set, will calculate the work and depth of the eDAG")
    args = parser.parse_args()

    if args.trace_file_path is None and args.load_path is None:
        print("[ERROR] Either provide the path to the trace file or "
              "the path to the saved eDAG")
        exit(-1)
    
    if args.use_cache_model:
        print("[INFO] Using cache model")
        cache = SingleLevelSetAssociativeCache()
    else:
        cache = None

    if args.use_cpu_model:
        print("[INFO] Using default CPU model")
        cpu_model = CPUModel(insn_cycles, frequency=3 * G)
    else:
        cpu_model = None

    if args.load_path:
        assert(args.trace_file_path is None)
        filename_root = Path(args.load_path).stem
        print(f"[INFO] Loading eDAG from {args.load_path}")
        eDag = EDag.load(args.load_path, compress)
        
    if args.trace_file_path:
        assert(args.load_path is None)
        filename_root = Path(args.trace_file_path).stem
        # Initializes eDAG generator
        print("[INFO] Generating eDAG")
        generator = EDagGenerator(args.trace_file_path, ISA.RISC_V,
                    args.only_mem_acc, args.sanitize,
                    cache_model=cache, cpu_model=cpu_model)
        eDag = generator.generate()

    if args.save_path:
        save_path = args.save_path
        eDag.save(save_path, compress)
        print(f"[INFO] Saved generated eDAG to {save_path}")
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File size of {save_path}: {file_size / K:.2f} KB")

    longest_path = None
    if args.calc_work_depth:
        print("[INFO] Calculating eDAG work")
        work = eDag.get_work()
        print(f"Work : {work}")
        print("[INFO] Calculating eDAG depth")
        # cProfile.run('depth = eDag.get_depth()')
        start = time()
        longest_path = eDag.get_longest_path(True)
        depth = len(longest_path)
        print(f"[DEBUG] Time taken: {time() - start}")
        print(f"Depth: {depth}")
        print(f"Parallelism: {work / depth:.2f}")


    critical_path = None
    if args.calc_bandwidth:
        print("[INFO] Calculating the average bandwidth utilization")
        assert cpu_model is not None
        bandwidth_metric = BandwidthUtilization(eDag, cpu_model.frequency)
        # Obtains some intermediate values that will help with
        # the calculation of other metrics
        critical_path_cycles, dp = \
            bandwidth_metric.get_critical_path_cycles(True)
        # Computes the average bandwidth
        avg_bandwidth = bandwidth_metric.get_avg_bandwidth(critical_path_cycles)
        print(f"Average bandwidth utilization: {avg_bandwidth / M:.2f} MB/s")

        # Retrieves the critical path itself
        critical_path = bandwidth_metric.get_critical_path(dp)

        # Computes the data movement over time
        bins, data_movement = \
            bandwidth_metric.get_data_movement_over_time(1, False)
        # fig_path = f"../tmp/{filename_root}_dm.png"
        visualize_data_movement_over_time(bins, data_movement, False)

    if args.optimize_subgraph:
        print("[INFO] Optimizing subgraph")
        optimizer = RiscvSubgraphOptimizer()
        optimizer.optimize(eDag)

    if args.remove_unconnected_vertices:
        eDag.remove_unconnected_vertices()

    if args.graph_file is not None:
        highlight = args.highlight_mem_acc
        print("[INFO] Generating graph")
        vertex_rank = eDag.get_vertex_rank()
        # vertex_rank = None
        graph = visualize_eDAG(eDag, highlight, vertex_rank=vertex_rank)
        # Highlights the longest path if it exists
        if longest_path:
            visualize_path(eDag, graph, highlight, longest_path, "blue")
        if critical_path:
            visualize_path(eDag, graph, highlight, critical_path)
        graph.render(args.graph_file, view=True)

    if args.reuse_histogram:
        # Generates the reuse distance histogram based on the given trace
        print("[INFO] Generating reuse distance histogram")
        reuse_distance_metric = ReuseDistance(eDag)
        reuse_histograms = \
            reuse_distance_metric.get_all_reuse_histograms()
        for i, reuse_histogram in enumerate(reuse_histograms):
            save_filename = f"../histos/{filename_root}_histo_{i}.png"
            visualize_reuse_histogram(reuse_histogram, save_filename)

