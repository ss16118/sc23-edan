import re
import time

class TraceRiscv(gdb.Command):
    """
    Implementation from:
    https://stackoverflow.com/questions/8841373/displaying-each-assembly-instruction-executed-in-gdb
    """
    reg_offset_pattern = re.compile(r"(-?\d+)\((\w+\d*)\)")
    
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
            # FIXME: Seems like the script only works for single-threaded programs
            # Needs to verify what multi-threaded programs look like
            thread = gdb.inferiors()[0].threads()[0]
            # Stores each line of assembly in the list first, prints
            # everything at the end
            asm_trace = []
            while thread.is_valid():
                frame = gdb.selected_frame()
                pc = frame.pc()
                # Retrieves the current instruction
                asm = frame.architecture().disassemble(pc)[0]['asm']
                out = f"{hex(pc)} {asm}"
                tokens = asm.split()
                reg_val = None
                if len(tokens) >= 2:
                    # Checks if the current instruction loads or stores to
                    # a memory location
                    operands = asm.split()[1].split(',')
                    if len(operands) >= 2:
                        matches = TraceRiscv.reg_offset_pattern.match(operands[1])
                        if matches is not None:
                            offset = int(matches.group(1))
                            reg = matches.group(2)
                            reg_val = hex(frame.read_register(reg) + offset)
                if reg_val is not None:
                    out = f"{out} {reg_val}"

                asm_trace.append(out)
                            
                # sal = frame.find_sal()
                # symtab = sal.symtab
                # TODO: This is probably not the best way to identify end of a program
                # if symtab and "libc" in symtab.fullname():
                #     break
                gdb.execute('si', to_string=True)
            # gdb.execute('continue', to_string=True)
            
            with open("tmp.out", "w") as trace_file:
                trace_file.write("\n".join(asm_trace))


class TraceX86(gdb.Command):
    """
    Implementation from:
    https://stackoverflow.com/questions/8841373/displaying-each-assembly-instruction-executed-in-gdb
    """
    reg_offset_pattern = re.compile(r"(-?\d+)\((\w+\d*)\)")
    
    def __init__(self):
        super().__init__(
            'trace-x86',
            gdb.COMMAND_BREAKPOINTS,
            gdb.COMPLETE_NONE,
            False
        )
    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if argv:
            gdb.write('Does not take any arguments.\n')
        else:
            gdb.execute("set logging on")
            gdb.execute("start")
            # FIXME: Seems like the script only works for single-threaded programs
            # Needs to verify what multi-threaded programs look like
            thread = gdb.inferiors()[0].threads()[0]
            # Stores each line of assembly in the list first, prints
            # everything at the end
            asm_trace = []
            while thread.is_valid():
                frame = gdb.selected_frame()
                pc = frame.pc()
                # Retrieves the current instruction
                asm = frame.architecture().disassemble(pc)[0]['asm']
                out = f"{hex(pc)} {asm}"

                asm_trace.append(out)
                            
                # sal = frame.find_sal()
                # symtab = sal.symtab
                # TODO: This is probably not the best way to identify end of a program
                # if symtab and "libc" in symtab.fullname():
                #     break
                gdb.execute('si', to_string=True)
            # gdb.execute('continue', to_string=True)
            with open("trace.out", "w") as trace_file:
                trace_file.write("\n".join(asm_trace))


class TraceAarch64(gdb.Command):
    """
    Implementation from:
    https://stackoverflow.com/questions/8841373/displaying-each-assembly-instruction-executed-in-gdb
    """
    def __init__(self):
        super().__init__(
            'trace-aarch64',
            gdb.COMMAND_BREAKPOINTS,
            gdb.COMPLETE_NONE,
            False
        )
    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if argv:
            gdb.write('Does not take any arguments.\n')
        else:
            # FIXME: Seems like the script only works for single-threaded programs
            # Needs to verify what multi-threaded programs look like
            thread = gdb.inferiors()[0].threads()[0]
            # Stores each line of assembly in the list first, prints
            # everything at the end
            asm_trace = []
            while thread.is_valid():
                frame = gdb.selected_frame()
                pc = frame.pc()
                # Retrieves the current instruction
                asm = frame.architecture().disassemble(pc)[0]['asm']
                out = f"{hex(pc)} {asm}"

                asm_trace.append(out)
                            
                # sal = frame.find_sal()
                # symtab = sal.symtab
                # TODO: This is probably not the best way to identify end of a program
                # if symtab and "libc" in symtab.fullname():
                #     break
                gdb.execute('si', to_string=True)
            # gdb.execute('continue', to_string=True)
            
            with open("trace.out", "w") as trace_file:
                trace_file.write("\n".join(asm_trace))


TraceRiscv()
TraceX86()
TraceAarch64()
