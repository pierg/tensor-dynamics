#!/bin/bash

# Script to parallelize GPU jobs for neural network configurations

# Define the path to the configurations file
CONFIG_FILE="../config/configurations.toml"

# Check if the configuration file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file does not exist."
    exit 1
fi

# Extract configuration identifiers from the TOML file
config_ids=$(grep '^\[[^.]*\]$' "$CONFIG_FILE" | tr -d '[]')

# Validate the presence of configurations
if [[ -z "$config_ids" ]]; then
    echo "Error: No configurations found in the file."
    exit 1
fi

# Loop through each configuration and submit a separate job
while IFS= read -r id; do
    echo "Submitting configuration $id"
    sbatch ./submit-gpu.sh "$id"
    if [[ $? -ne 0 ]]; then
        echo "Error occurred while submitting configuration $id"
        # Uncomment to stop execution on first error
        # exit 1
    fi
done <<< "$config_ids"

echo "All configurations have been submitted."
exit 0
