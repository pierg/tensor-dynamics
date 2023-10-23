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

#!/bin/bash

# Assuming LAUNCH_BASH and other variables are set earlier in the script or are passed as environment variables

if [ $LAUNCH_BASH -eq 1 ]; then
    # If the "-c" argument was passed, launch an interactive bash shell
    exec /bin/bash  # 'exec' replaces the current process (the script) with the new command (bash)
else
    if ! pgrep -x "tensorboard" > /dev/null; then
        echo "Starting TensorBoard..."
        tensorboard --logdir ./logs &  # This starts tensorboard in the background
    else
        echo "TensorBoard is already running."
    fi
    
    # Proceed if not launching bash (the rest of your script remains the same)
    # Run the python command depending on whether CONFIGS is provided
    if [ -n "$CONFIGS" ]; then
        # If CONFIGS is provided, include it in the python command
        
        python src/main.py --data_dir=$DATA_DIR $CONFIGS
    else
        # If CONFIGS is not provided, run the python command without it
        python src/main.py --data_dir=$DATA_DIR
    fi
fi
