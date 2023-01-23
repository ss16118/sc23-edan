import os
import argparse
from typing import Optional


def cleanup_trace_file(trace_file_path: str, start_delimiter: str = "<main>:",
                       new_filename: Optional[str] = None) -> None:
    """
    Iterates through every line in the given trace file and
    removes redundant lines. If the name of the post-cleanup trace file
    is not given, the default filename will be the old filename plus "_cleanup"
    at the end.
    """
    if new_filename is None:
        filename, file_extension = os.path.splitext(trace_file_path)
        new_file_path = filename + "_cleanup" + file_extension
    else:
        file_dir, _ = os.path.split(trace_file_path)
        new_file_path = file_dir + "/" + new_filename
    
    trace_file = open(trace_file_path, "r")
    new_file = open(new_file_path, "w")
    start = False
    for line in trace_file:
        # Inserts assembly instructions into the new cleaned-up
        # trace file only when 'start' is set to true, that
        # is when the `start_delimiter` is detected
        if start_delimiter in line:
            start = True
        elif "<__GI_exit>:" in line:
            # When the program exits from main, the relevant code stops
            break
        
        if start and line.startswith("=>") and not line == "":
            new_file.write(line[3:])
            
    trace_file.close()
    new_file.close()    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trace file cleanup",
                                     description="`Clean up the instruction trace file produced by gdb")
    parser.add_argument("-f", "--trace-file",
                        dest="trace_file_path",
                        help="File path to the instruction trace file")
    parser.add_argument("-s", "--start-delimiter",
                        dest="start_delimiter", default="<main>:",
                        help="A string whose appearance marks the start of the actual trace")
    parser.add_argument("-c", "--new-filename",
                        dest="new_filename",
                        help="The new filename of the trace file post clean up")

    args = parser.parse_args()
    if args.trace_file_path is None:
        print("[ERROR] Path to the trace file must be provided")
        exit(-1)
    cleanup_trace_file(args.trace_file_path, args.start_delimiter, args.new_filename)
