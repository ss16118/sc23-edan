#!/bin/bash

# Check if directory is provided as a command-line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 directory"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$1" ]; then
    echo "Error: directory $1 not found."
    exit 1
fi

# Change to the specified directory
cd "$1" || exit

# Loop through every file in the directory
for file in *; do
    # Check if the file is executable and a binary file
    if [ -x "$file" ] && [ -f "$file" ]; then
        # Execute the binary file
	outfile=${file}.out
	rm -f $outfile
	for i in {1..10}; do
	    { time ./$file; } 2>> $outfile
	done
    fi
done
