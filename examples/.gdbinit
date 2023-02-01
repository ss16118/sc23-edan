set auto-load safe-path /
set solib-search-path /opt/riscv/sysroot/lib
set logging redirect on
set logging file /dev/null
set height 0
source trace.py


define trace-asm
  while 1
    x/i $pc
    stepi
  end
end
