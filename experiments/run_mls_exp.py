# ====================================================
# Performs the experiment to measure the memory
# latency sensitivity of a set of kernels by
# first tracing them and then generating their
# corresponding eDAG.
# ====================================================
import os
import sys
import subprocess
import magic
from time import time
from pathlib import Path

PROJECT_ROOT_DIR = "/home/sishen/workspace/samsung"
HOME_DIR = f"{PROJECT_ROOT_DIR}/experiments"
QEMU_DIR = "/home/sishen/workspace/qemu/build"
TRACE_PLUGIN = f"{QEMU_DIR}/contrib/plugins/libexeclog.so,ffilter=kernel"
TRACE_CMD = \
    f'{QEMU_DIR}/qemu-riscv64 -plugin "{TRACE_PLUGIN}" -d plugin'
CD_CMD1 = f"cd {PROJECT_ROOT_DIR}/eDAG_generator"
CD_CMD2 = f"cd {PROJECT_ROOT_DIR}/experiments"
MLS_CMD = "python3 main.py --cache --work-depth"

binary_dir = sys.argv[1]
# Sets the default log directory
log_dir = os.path.join(HOME_DIR, "logs")
if len(sys.argv) > 2:
    log_dir = os.path.join(HOME_DIR, sys.argv[2])

# Sets the default trace file directory
trace_dir = os.path.join(HOME_DIR, "tmp")
if len(sys.argv) > 3:
    trace_dir = os.path.join(HOME_DIR, sys.argv[3])

print(f"[INFO] log directory: {log_dir}")
print(f"[INFO] trace directory: {trace_dir}")
    
mime = magic.Magic(mime=True)

# Traces all RISC-V binaries in a given directory
for subdir, dirs, files in os.walk(binary_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        file_type = mime.from_file(file_path)
        # Checks if the file is a binary executable
        if file_type == "application/x-executable":
            print(f"[INFO] Kernel: {file}")
            log_file = os.path.join(log_dir, f"{file}_log.out")
            if os.path.exists(log_file):
                os.remove(log_file)
            log = open(log_file, "a")
            trace_file = os.path.join(trace_dir, f"{file}_trace.out")
            # If trace file does not exist, generate trace file
            if not os.path.exists(trace_file):
                os.system(CD_CMD2)
                output = f"[INFO] Tracing {file}..."
                log.write(f"{output}\n")
                cmd = f"{TRACE_CMD} -D {trace_file} {file_path}"
                start = time()
                os.system(cmd)
                time_taken = time() - start
                output = f"[INFO] Trace completed, time taken: {time_taken:.5f}"
                print(output)
                log.write(f"{output}\n")
            else:
                print(f"[INFO] Trace file found for kernel {file}")

            # Generates eDAG from the trace file
            print(f"[INFO] Generating and analyzing eDAG for binary: {file}")
            cmd = f"{CD_CMD1} && {MLS_CMD} -f {trace_file} > {log_file}"
            print(cmd)
            os.system(cmd)
            print(f"[INFO] Analysis completed for {file}")
            log.close()
            
    



