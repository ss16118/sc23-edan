import re
import time

class TraceRiscv(gdb.Command):
    reg_offset_pattern = re.compile(r"-?\d+\((\w+\d*)\)")
    
    def __init__(self):
        super().__init__(
            'trace-riscv',
            gdb.COMMAND_BREAKPOINTS,
            gdb.COMPLETE_NONE,
            False
        )
    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if argv:
            gdb.write('Does not take any arguments.\n')
        else:
            done = False
            thread = gdb.inferiors()[0].threads()[0]
            last_path = None
            last_line = None
            while thread.is_valid():
                frame = gdb.selected_frame()
                pc = frame.pc()
                # Retrieves the current instruction
                asm = frame.architecture().disassemble(pc)[0]['asm']
                out = f"{hex(pc)} {asm}"
                print(out)
                tokens = asm.split()
                if len(tokens) >= 2:
                    # Checks if the current instruction loads or stores to
                    # a memory location
                    operands = asm.split()[1].split(',')
                    if len(operands) >= 2:
                        matches = TraceRiscv.reg_offset_pattern.match(operands[1])
                        if matches is not None:
                            reg = matches.group(1)
                            print(f"{reg}: {hex(frame.read_register(reg))}")
                sal = frame.find_sal()
                symtab = sal.symtab
                if symtab and "libc" in symtab.fullname():
                    print(symtab.fullname(), sal.line)
                    break
                gdb.execute('si', to_string=True)
            gdb.execute('continue', to_string=True)
TraceRiscv()
