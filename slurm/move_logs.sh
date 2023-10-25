#!/bin/bash

# Define the directory where files should be moved
TARGET_DIR="logs"

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
  # If the directory doesn't exist, create it
  mkdir "$TARGET_DIR"
fi

# Find all .out and .err files in the current directory and move them to the target directory
# The loop handles filenames with spaces
for file in *.out *.err; do
  # Check if the glob gets expanded to existing files.
  # If not, file will be exactly the pattern as a string, which should not be moved.
  [ -e "$file" ] || continue
  
  # If conditions are met, move files
  mv -- "$file" "$TARGET_DIR"
done
