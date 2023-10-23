#!/bin/bash

# Define the path to the configurations file
CONFIG_FILE="../config/configurations.toml"

# Check if the configuration file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file does not exist."
    exit 1
fi

# Extract configuration identifiers from the TOML file. We're looking for lines that match '[*]'
# where * is the identifier, then stripping away the square brackets.
config_ids=$(grep '^\[.*\]$' "$CONFIG_FILE" | tr -d '[]')

# Check if we found any configurations
if [[ -z "$config_ids" ]]; then
    echo "Error: No configurations found in the file."
    exit 1
fi

# Loop through each configuration identifier and execute the submit command
while IFS= read -r id; do
    echo "Submitting configuration $id"
    # Call the submit script with the configuration ID as a parameter
    ./submit.sh "$id"  # adjusted here
    # Check the exit status of the submit command and handle any errors
    if [[ $? -ne 0 ]]; then
        echo "An error occurred while submitting configuration $id"
        # Uncomment the next line if you want the script to stop on the first error
        # exit 1
    fi
done <<< "$config_ids"

echo "All configurations have been submitted."

# Exit the script successfully
exit 0
