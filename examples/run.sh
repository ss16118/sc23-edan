#!/bin/bash

# A bash script for collecting the trace of simple C programs complied to RISC-V
src=$1
target=$2

# Preparation
source ./setup.sh
sudo pkill --signal 9 qemu-riscv64
rm trace.out

make compile src=${src} target=${target}
echo Compiled ${src}

# Starts the QEMU process
PORT=1234
echo Start QEMU
nohup qemu-riscv64 -g ${PORT} ${target} > /dev/null 2>&1 &

# Starts gdb
GDB=/opt/riscv/bin/riscv64-unknown-linux-gnu-gdb
echo Start GDB
CMD1="target remote :${PORT}" # Connects to QEMU
CMD2="b kernel"               # Sets a break point at function "kernel"
CMD3="continue"               # Starts running the program
CMD4="set logging enabled on" # Enables logging
CMD5="trace-riscv"            # Starts collecting trace
CMD6="quit"
$GDB -ex "$CMD1" -ex "$CMD2" -ex "$CMD3" -ex "$CMD4" -ex $CMD5 -ex $CMD6 $target
