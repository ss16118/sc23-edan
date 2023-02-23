#!/bin/bash

# Checks the number of arguments passed in
if [ $# -eq 0 ]
then
    echo "Need to pass in at least one argument to run the script"
    exit 1
fi

binary=$1
if [ -z "$2" ]
then
    # If the path to the trace log is not specified by the user
    trace_log="/home/blueadmin/workspace/samsung/examples/trace.out"
else
    trace_log="$2"
fi

rm ${trace_log}
QEMU_FLAGS="-L /opt/riscv/sysroot/"

echo "Running ${binary} in QEMU without tracing"
QEMU_WITHOUT_TRACING=qemu-riscv64
time ${QEMU_WITHOUT_TRACING} ${QEMU_FLAGS} ${binary}

echo "Running ${binary} in QEMU with tracing"
QEMU_DIR="/home/blueadmin/workspace/qemu/build/"
QEMU_WITH_TRACING="${QEMU_DIR}qemu-riscv64"
TRACE_PLUGIN="${QEMU_DIR}/contrib/plugins/libexeclog.so"
time ${QEMU_WITH_TRACING} ${QEMU_FLAGS} \
     -plugin "${TRACE_PLUGIN}" -d plugin \
     -D "${trace_log}" ${binary}
echo "Trace file saved to ${trace_log}"
