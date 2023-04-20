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


kernel_configs = {
    "2mm": ["NI", "NJ", "NK", "NL"],
    "3mm": ["NI", "NJ", "NK", "NL", "NM"],
    "atax": ["NX", "NY"],
    "bicg": ["NX", "NY"],
    "cholesky": ["N"],
    "doitgen": ["NQ", "NR", "NP"],
    "gemm": ["NI", "NJ", "NK"],
    "gemver": ["N"],
    "gesummv": ["N"],
    "mvt": ["N"],
    "symm": ["NI", "NJ"],
    "syr2k": ["NI", "NJ"],
    "syrk": ["NI", "NJ"],
    "trisolv": ["N"],
    "trmm": ["NI"]
}

data_sizes = list(range(2, 33))
# data_sizes = [64, 128]

PROJECT_ROOT_DIR = "/home/sishen/workspace/samsung"
POLYBENCH_ROOT_DIR = "/home/sishen/workspace/polybench-c-3.2/"
POLYBENCH_KERNELS_DIR = f"{POLYBENCH_ROOT_DIR}linear-algebra/kernels"
HOME_DIR = f"{PROJECT_ROOT_DIR}/experiments"
QEMU_DIR = "/home/sishen/workspace/qemu/build"
TRACE_PLUGIN = f"{QEMU_DIR}/contrib/plugins/libexeclog.so,ffilter=kernel"
TRACE_CMD = \
    f'{QEMU_DIR}/qemu-riscv64 -plugin "{TRACE_PLUGIN}" -d plugin'
CD_CMD1 = f"cd {PROJECT_ROOT_DIR}/eDAG_generator"
CD_CMD2 = f"cd {PROJECT_ROOT_DIR}/experiments"
PYTHON_CMD = "python3 main.py --work-depth --mls"

binary_dir = os.path.join(HOME_DIR, "case_study_kernels")
if len(sys.argv) > 1:
    binary_dir = os.path.join(HOME_DIR, sys.argv[1])
# Sets the default log directory
log_dir = os.path.join(HOME_DIR, "logs")
if len(sys.argv) > 2:
    log_dir = os.path.join(HOME_DIR, sys.argv[2])

# Sets the default trace file directory
trace_dir = os.path.join(HOME_DIR, "tmp")
if len(sys.argv) > 3:
    trace_dir = os.path.join(HOME_DIR, sys.argv[3])

print(f"[INFO] binary directory: {binary_dir}")
print(f"[INFO] log directory: {log_dir}")
print(f"[INFO] trace directory: {trace_dir}")

def check_err(err_code):
    if err_code != 0:
        print(f"[ERROR] An error occurred, error code: {err_code}")
        sys.exit(-1)


COMPILE_CMD = "riscv64-unknown-linux-gnu-gcc -static -fno-inline -O2 " + \
    "-I " + POLYBENCH_ROOT_DIR + "utilities " + \
    POLYBENCH_ROOT_DIR + "/utilities/polybench.c"

mime = magic.Magic(mime=True)

for data_size in data_sizes:
    print("==========================")
    print(f"==== Data size {data_size} ====")
    print("==========================")
    # Compiles all kernels for a specific data size
    print(f"[INFO] Compiling kernels for size: {data_size}")
    for subdir, dirs, files in os.walk(POLYBENCH_KERNELS_DIR):
        for file in files:
            file_name = file.split(".")[0]
            if ".h" in file or \
               file_name not in kernel_configs:
                continue

            file_path = os.path.join(subdir, file)
            binary_path = os.path.join(binary_dir, f"{file_name}_{data_size}")
            if not os.path.exists(binary_path):
                print(f"[INFO] Compiling binary for {file_name}")
                # Adds the data size definitions
                cmd = f"{COMPILE_CMD}"
                for flag in kernel_configs[file_name]:
                    cmd += f" -D{flag}={data_size}"
                cmd += f" -o {binary_path} {file_path} -lm"
                print(cmd)
                ret = os.system(cmd)
                check_err(ret)
            else:
                print(f"[INFO] Binary already exists for {file_name}")

    print(f"[INFO] Compliation finished for data size {data_size}")

    for subdir, dirs, files in os.walk(binary_dir):
        for file in files:
            if not file.endswith(f"_{data_size}"):
                continue
            file_path = os.path.join(subdir, file)
            file_type = mime.from_file(file_path)
            print(f"[INFO] Kernel {file}")
            log_file = os.path.join(log_dir, f"{file}_log.out")
            if os.path.exists(log_file):
                os.remove(log_file)
            log = open(log_file, "a")
            trace_file = os.path.join(trace_dir, f"{file}_trace.out")
            # If trace file does not exist, generate trace file
            if not os.path.exists(trace_file):
                ret = os.system(CD_CMD2)
                check_err(ret)
                output = f"[INFO] Tracing {file}..."
                log.write(f"{output}\n")
                cmd = f"{TRACE_CMD} -D {trace_file} {file_path}"
                start = time()
                ret = os.system(cmd)
                check_err(ret)
                time_taken = time() - start
                output = f"[INFO] Trace completed, time taken: {time_taken:.5f}"
                print(output)
                log.write(f"{output}\n")
            else:
                print(f"[INFO] Trace file found for kernel {file}")


            # Generates eDAG from the trace file
            print(f"[INFO] Generating and analyzing eDAG for binary: {file}")
            cmd = f"{CD_CMD1} && {PYTHON_CMD} -f {trace_file} > {log_file}"
            print(cmd)
            ret = os.system(cmd)
            check_err(ret)
            print(f"[INFO] Analysis completed for {file}")
            log.close()
            
    



