#!/bin/bash

# Pull the latest changes from the repository
# Note: Replace 'main' with the relevant branch if your default branch is not 'main'
git fetch origin main
if [ $? -ne 0 ]; then
    echo "Failed to fetch the latest changes from the origin. Exiting."
    exit 1
fi

git reset --hard origin/main
if [ $? -ne 0 ]; then
    echo "Failed to reset the branch to the latest changes from the origin. Exiting."
    exit 1
fi

# Check if run.sh is executable, if not, add the permission
if [ ! -x "run.sh" ]; then
    chmod +x run.sh
fi

# Forward all script arguments to run.sh
./run.sh "$@"
