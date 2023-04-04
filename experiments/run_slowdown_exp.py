# ====================================================
# Performs the experiment to measure the slowdown of
# a set of kernels when traced in QEMU.
# ====================================================
import os
import sys
import subprocess
import magic
from time import time
from pathlib import Path


QEMU_DIR = "/home/blueadmin/workspace/qemu/build"
TRACE_PLUGIN = f"{QEMU_DIR}/contrib/plugins/libexeclog.so"
TRACE_FILE = "trace.out"
CMD = \
    f'{QEMU_DIR}/qemu-riscv64 -plugin "{TRACE_PLUGIN}" -d plugin -D "{TRACE_FILE}"'

binary_dir = sys.argv[1]
n_iters = int(sys.argv[2])

mime = magic.Magic(mime=True)

# Traces all RISC-V binaries in a given directory
for subdir, dirs, files in os.walk(binary_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        file_type = mime.from_file(file_path)
        # Checks if the file is a binary executable
        if file_type == "application/x-executable":
            print(f"[INFO] Kernel: {file}")
            cmd = f"{CMD} {file_path}"
            log_file = f"{file}_log.out"
            if os.path.exists(log_file):
                os.remove(log_file)
            file = open(log_file, "a")
            for i in range(n_iters):
                if os.path.exists(TRACE_FILE):
                    os.remove(TRACE_FILE)
                start = time()
                os.system(cmd)
                time_taken = time() - start
                output = f"Trial {i + 1}: {time_taken:.5f}"
                print(output)
                file.write(f"{output}\n")
            file.close()
            
    



