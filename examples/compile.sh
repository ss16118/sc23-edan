CC=riscv64-unknown-linux-gnu-gcc
CC_FLAGS="-O3 -g"
LD_FLAGS=""

src=$1
target=$2

# Compile to object files
${CC} ${CC_FLAGS} -c ${src}
# Linking
${CC} ${LD_FLAGS} -o ${target} "${target}.o"
