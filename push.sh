#!/bin/bash

git add --a


# Source the secrets file
source .secrets

git fetch origin

# Merge the changes from the remote branch (e.g., main) into your local branch
# The strategy-option 'ours' is used to keep local changes in conflict.
# Note: Replace 'main' with the relevant branch if it's different in your setup.
git merge -X ours origin/main

# Check if the GITHUB_TOKEN was loaded
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Failed to load the GITHUB_TOKEN from the .secrets file"
    exit 1
fi

# Adding all changes (it should be '.' or '--all', not '--a')
git add .

# Committing the changes
git commit -m "update"


git remote set-url origin https://$GITHUB_TOKEN@github.com/pierg/neural_networks.git


# Pushing the changes
git push origin main