# Samsung GRO Work Package 1
### Quick Start

##### Install Dependencies

To collect trace and generate eDAG from scratch, it needs to be ensured that the following toolchains and dependencies are installed:
- __RISC-V GNU Compiler Toolchain__: It is used to cross-compile C and C++ programs, installation instructions can be found in its [github repo](https://github.com/riscv-collab/riscv-gnu-toolchain).
- __QEMU__: After cross compilation, the generated RISC-V binary needs to be run on QEMU in order for the trace to be collected. In this case, as a proof-of-concept implementation, I only tried running them in QEMU user mode, and the full-system emulation needs to be tested later. Although the precompiled QEMU binary can be used to collect the trace, it does not have the TCG plugins enabled by default, meaning that one would have to rely on gdb to to trace programs. Therefore, it is preferable to build QEMU from source while enabling plugins. Detailed instructions can be found in the [github repo](https://github.com/qemu/qemu) and the [webpage about plugins](https://qemu.readthedocs.io/en/latest/devel/tcg-plugins.html). In short, to build RSIC-V QEMU (user mode) from source, type in the following commands:
  ```console
    > cd qemu
    > ./configure --enable-plugins --target-list=riscv64-linux-user
    > make
  ```
- To run the eDAG generator in Python, simply install all the libraries listed in __requirements.txt__.

#### Tracing

If you simply want to check the eDAG generation and does not want to go through the process of compiling and tracing programs, you can just use the example traces included in the `examples` directory and skip ahead to the next section. However, the example traces are quite small, and does not work well as stress tests for the program. Therefore, to obtain traces for larger programs / kernels, tracing has to be done from scratch.

To trace a program, first of all, copy __trace/qemu_execlog.c__ to __qemu/contrib/plugins/execlog.c__. Assuming that QEMU has already been built and the "build" directory exists:

```console
> cp trace/qemu_execlog.c <qemu-directory>/contrib/plugins/execlog.c
> cd <qemu-directory>/build/contrib/plugins && make
```
Then, generate the binary for any program with the RISC-V cross compiler. As a simple example:
```console
> cd examples
> riscv64-unknown-linux-gnu-gcc -o sum sum.c
> cd ../trace
> ./trace.sh ../examples/sum trace.out
```
Note that by default, the trace.sh script traces the entire program, including all the setup code, invocation of library functions and syscalls. To trace specific functions, change the line 29 in the trace script to `TRACE_PLUGIN="${QEMU_DIR}/contrib/plugins/libexeclog.so,ffilter=<func1-name>,ffilter=<func2-name>,..."`, where `<func1-name>`, `<func2-name>` etc., can be replaced by the names of the functions that need to be traced. It is highly recommended to add these function filters to the trace script so as to remove all the irrelevant assembly instructions, which will greatly reduce the computation time for eDAG generation.

#### Generating eDAGs from traces

Once the trace has been generated, simply run the following command to generate its corresponding eDAG and save its visualization as a PDF:
```console
> cd eDAG_generator
> python3 main.py -f "../trace/trace.out" -g sum_eDAG --highlight-mem-acc
```
Different metrics can be calculated for the eDAG as per the specified flags. For instance, if you want to compute the work and depth of an eDAG:
```console
> python3 main.py -f "../trace/trace.out" --work-depth
```
In the cases where the trace file is large and it is time-consuming to construct the entire eDAG, you can save the compressed eDAG as a pickle file with:
```console
> python3 main.py -f "../trace/trace.out" -s sum_eDAG.pkl
```
so that next time it can be loaded directly
```console
> python3 main.py -l sum_eDAG.pkl --work-depth
```

### Commandline Options for eDAG Generation

The commandline options for the Python script are as follows:

| Options             | Alternative                   | Description                                                                                           | Default |
|---------------------|-------------------------------|-------------------------------------------------------------------------------------------------------|---------|
| -h                  | --help                        | show this help message and exit                                                                       |         |
| -f                  | --trace-file                  | File path to the instruction trace file                                                               |         |
| -l                  | --load-file                   | If set, will try to load the eDAG object directly from the given file                                 |         |
| -g                  | --graph-file                  | Path to which the visualization of the eDAG will be saved                                             | None    |
| -m                  | --only-mem-acc                | If set, only vertices with memory accesses will be displayed                                          | False   |
| --highlight-mem-acc |                               | If set, memory access vertices will be highlighted when the eDAG is saved to PDF                      | False   |
| -r                  | --remove-unconnected-vertices | If set, unconnected vertices without any connections will be removed                                  | False   |
| -s                  | --save-path                   | If set, will save the original generated eDAG to the given path                                       | None    |
| --sanitize          |                               | If set, will try to simplify the eDAG                                                                 | False   |
| -o                  | --optimize                    | If set, will attempt optimize the eDAG in terms of work and depth according to pre-defined heuristics | False   |
| --reuse-histogram   |                               | If set, will generate the reuse distance histogram based on the given trace                           | False   |
| --bandwidth         |                               | If set, bandwidth related metrics will be computed                                                    | False   |
| --cpu               |                               | If set, a predefined CPU model will be used                                                           | False   |
| --cache             |                               | If set, a predefined LRU cache model will be used                                                     | False   |
| --work-depth        |                               | If set, will calculate the work and depth of the eDAG                                                 | False   |

Few things to note:
- The `--trace-file` and `--load-file` options are mutually exclusive. In essence, you can choose either to create the eDAG from the trace or re-load it from a save file.
- `--bandwidth` has to be used along with `--cpu` while `--cpu` can be used alone. [Needs to be fixed]
- When used with `-g`, `--bandwidth` will highlight the critical path in orange while `--work-depth` will highlight the critical path in blue.
- A lot of experimental features such as `--sanitize`, `-r`, `-o` are not compatible with a lot of the current features (specifically the _igraph_ library used for topological sort), so errors will likely occur when they are used alongside other flags.
- Reuse histogram is a feature that is still under construction.

### Program Tracing

### eDAG Parsing

### Metrics Calculation


### TODO