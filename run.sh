#!/bin/bash

# Initialize variables with default values
DATA_DIR="/data"  # Set default value
CONFIGS=""

# Note: No need to check for mandatory arguments now, since we have a default value

# Parse arguments
for arg in "$@"
do
    case $arg in
        DATA_DIR=*)
        DATA_DIR="${arg#*=}"  # Override if provided
        ;;
        CONFIGS=*)
        CONFIGS="${arg#*=}"  # Override if provided
        ;;
        *)
        # Ignore unexpected arguments
        ;;
    esac
done

# No need to check if DATA_DIR is empty, as we're ensuring it's either the default or provided value.

# Run the make command depending on whether CONFIGS is provided
if [ -n "$CONFIGS" ]; then
    # If CONFIGS is provided, include it in the make command
    make run DATA_DIR="$DATA_DIR" CONFIGS="$CONFIGS"
else
    # If CONFIGS is not provided, run the make command without it
    make run DATA_DIR="$DATA_DIR"
fi
