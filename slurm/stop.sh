#!/bin/bash

# Function to get the job ID based on the provided index
get_job_id_by_index() {
    # Run the queue command and find the appropriate job. Adjust this line depending on how you retrieve your job queue (e.g., squeue for SLURM)
    JOBID=$(./queue.sh | awk -v idx="$1" 'NR == idx+1 {print $1}')
    echo $JOBID
}

# If no argument is provided, we'll stop all jobs.
if [ -z "$1" ]; then
    echo "No index provided, stopping all jobs."
    # Cancel all jobs. This command may vary depending on your job scheduler.
    # For SLURM, you might do something like this:
    scancel -u $(whoami)  # This cancels all jobs belonging to the current user
    exit 0  # Exit after stopping all jobs
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
