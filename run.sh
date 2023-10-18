#!/bin/bash

# Initialize variables with default values
DATA_DIR="/data"  # Set default value
CONFIGS=""
LAUNCH_BASH=0  # By default, do not launch bash

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
        -c)
        LAUNCH_BASH=1  # Set flag to launch bash
        ;;
        *)
        # Ignore unexpected arguments
        ;;
    esac
done

if [ $LAUNCH_BASH -eq 1 ]; then
    # If the "-c" argument was passed, launch an interactive bash shell
    /bin/bash
    exit  # After bash shell is closed, exit the script
fi

# Proceed if not launching bash (the rest of your script remains the same)
# Run the make command depending on whether CONFIGS is provided
if [ -n "$CONFIGS" ]; then
    # If CONFIGS is provided, include it in the make command
    . .venv/bin/activate && python src/main.py --data_dir=$DATA_DIR $CONFIGS
else
    # If CONFIGS is not provided, run the make command without it
    . .venv/bin/activate && python src/main.py --data_dir=$DATA_DIR
fi
