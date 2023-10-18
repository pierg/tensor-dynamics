#!/bin/bash

# Initialize variables with default values
DATA_DIR=""
CONFIGS=""

# Check if we have the mandatory argument
if [ "$#" -lt 1 ]; then
    echo "Error: DATA_DIR argument is mandatory."
    echo "Usage: docker run <image> DATA_DIR=<data-dir> [CONFIGS='<config1> <config2> ...']"
    exit 1
fi

# Parse arguments
for arg in "$@"
do
    case $arg in
        DATA_DIR=*)
        DATA_DIR="${arg#*=}"
        ;;
        CONFIGS=*)
        CONFIGS="${arg#*=}"
        ;;
        *)
        # You can decide to do something with unexpected arguments or ignore them
        ;;
    esac
done

# Check if DATA_DIR was provided and is not empty
if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR not provided."
    exit 1
fi

# Run the make command depending on whether CONFIGS is provided
if [ -n "$CONFIGS" ]; then
    # If CONFIGS is provided, include it in the make command
    make run DATA_DIR="$DATA_DIR" CONFIGS="$CONFIGS"
else
    # If CONFIGS is not provided, run the make command without it
    make run DATA_DIR="$DATA_DIR"
fi
