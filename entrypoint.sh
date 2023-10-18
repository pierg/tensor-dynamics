#!/bin/bash

# Pull the latest changes from the repository
# Note: Replace 'main' with the relevant branch if your default branch is not 'main'
git fetch origin main
git reset --hard origin/main

# Check if run.sh is executable, if not, add the permission
if [ ! -x run.sh ]; then
    chmod +x run.sh
fi

# Forward all script arguments to run.sh
./run.sh "$@"
