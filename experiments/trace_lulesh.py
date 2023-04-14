# ====================================================
# Produces the trace file for LULESH2.0
# ====================================================
import os
import sys
from time import time


QEMU_DIR = "/home/blueadmin/workspace/qemu/build"
TRACE_PLUGIN = f"{QEMU_DIR}/contrib/plugins/libexeclog.so,trace_marks=1,fexclude=default"
TRACE_CMD = \
    f'{QEMU_DIR}/qemu-riscv64 -plugin "{TRACE_PLUGIN}" -d plugin'


binary = sys.argv[1]

trace_file = "trace.out"
if len(sys.argv) > 2:
    trace_file = sys.argv[2]

print(f"[INFO] trace file: {trace_file}")

# Tracing
print("[INFO] Tracing started")
trace_cmd = f'{TRACE_CMD} -D {trace_file} {binary} -i 10 -s 10'
print(trace_cmd)
start = time()
os.system(trace_cmd)
print(f"[INFO] Time taken: {time() - start:.3f} s")
