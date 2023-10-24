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

# No need for a second shebang (#!/bin/bash) here.

if [ $LAUNCH_BASH -eq 1 ]; then
    # If the "-c" argument was passed, launch an interactive bash shell
    exec /bin/bash  # 'exec' replaces the current process (the script) with the new command (bash)
else
    # Custom operations like checking or starting services can be placed here.

    # Proceed if not launching bash
    # Run the python command depending on whether CONFIGS is provided
    if [ -n "$CONFIGS" ]; then
        # If CONFIGS is provided, include it in the python command
        
        # Double-quote variables to prevent globbing and word splitting, in case of spaces or special characters in variables.
        python src/main.py --data_dir="$DATA_DIR" $CONFIGS
    else
        # If CONFIGS is not provided, run the python command without it
        python src/main.py --data_dir="$DATA_DIR"
    fi
fi
