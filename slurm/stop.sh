#!/bin/bash

# Function to get the job ID based on the provided index
get_job_id_by_index() {
    # Run the queue command and find the appropriate job. Adjust this line depending on how you retrieve your job queue (e.g., squeue for SLURM)
    JOBID=$(./queue.sh | awk -v idx="$1" 'NR == idx+1 {print $1}')
    echo $JOBID
}

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Error: No index provided."
    echo "Usage: $0 <index>"
    exit 1
fi

# Check if the argument is a number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: The provided index is not a number."
    exit 1
fi

INDEX=$1

# Get the corresponding job ID
JOBID=$(get_job_id_by_index $INDEX)

# Check if a job ID was found
if [ -z "$JOBID" ]; then
    echo "No job found at index $INDEX"
    exit 1
fi

# Cancel the job. Change 'scancel' to the appropriate command if not using SLURM.
scancel $JOBID
echo "Sent cancel request for job $JOBID."
