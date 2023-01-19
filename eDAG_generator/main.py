import os
import argparse
from riscv_parser import RiscvParser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trace file cleanup",
                                     description="`Clean up the instruction trace file produced by gdb")
    parser.add_argument("-f", "--trace-file", dest="trace_file_path",
                        help="File path to the instruction trace file")
    parser.add_argument("-o", "--graph-file", dest="graph_file", default=None,
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

    args = parser.parse_args()
    if args.trace_file_path is None:
        print("[ERROR] Path to the trace file must be provided")
        exit(-1)

    if args.graph_file is None:
        filename, file_extension = os.path.splitext(args.trace_file_path)
        graph_file = filename + "_eDAG"
    else:
        graph_file = args.graph_file

    parser = RiscvParser(args.trace_file_path, args.only_mem_acc,
                         args.remove_single_vertices)
    eDag = parser.parse()
    graph = eDag.visualize(args.highlight_mem_acc)
    graph.render(graph_file, view=True)