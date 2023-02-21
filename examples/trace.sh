#!/bin/bash

# A bash script for collecting the trace of simple C programs compiled to RSIC-V

#Prepraration

# Checks the number of arguments passed in
if [ $# -eq 0 ]
then
    echo "Need to pass in at least one argument to run the script"
    exit 1
fi

src=$1
target=$2

if [ -z "$3" ]
then
    # If the path to the trace log is not specified by the user
    trace_log="/home/blueadmin/workspace/samsung/examples/trace.out"
else
    trace_log="$3"
fi

# Performs cross compilation
make compile src=${src} target=${target}
echo Compiled ${src}

# Collects trace with QEMU
echo Collecting trace
QEMU="/home/blueadmin/workspace/qemu/build/qemu-riscv64"
QEMU_DIR="/home/blueadmin/workspace/qemu/build/"
# Assumes that the function which will be traced is `kernel()`
TRACE_PLUGIN="${QEMU_DIR}/contrib/plugins/libexeclog.so,ffilter=kernel"
QEMU_FLAGS="-L /opt/riscv/sysroot/"
time ${QEMU} ${QEMU_FLAGS} \
     -plugin "${TRACE_PLUGIN}" -d plugin \
     -D "${trace_log}" ${target}

echo Trace collected to ${trace_log}
ls -l --block-size=M ${trace_log}
