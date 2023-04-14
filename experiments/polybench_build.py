import sys
import os
import subprocess
from pathlib import Path

dataset_type = sys.argv[1]

# cross-compile polybench for riscv

thispath = os.path.dirname(os.path.realpath(__file__))
polybench_root = thispath + "/polybench-c-3.2/"
binary_dir = thispath + "/kernels/"

if not Path(binary_dir).is_dir():
    os.mkdir(binary_dir)

for subdir, dirs, files in os.walk(polybench_root):
    if "utilities" in subdir:
        continue

    for file in files:
        if ".h" in file:
            continue
        
        cmd = "riscv64-unknown-linux-gnu-gcc -static -O3 " + \
            " -I " + polybench_root + "/utilities/ " + \
            polybench_root + "/utilities/polybench.c " + \
            os.path.join(subdir, file) + \
            " -D" + dataset_type + \
            " -o " + binary_dir + file[:-2] + \
            " -lm "

        print(cmd)
        os.system(cmd)
