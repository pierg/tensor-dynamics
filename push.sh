#!/bin/bash

git add --a


# Source the secrets file
source .secrets

# Check if the GITHUB_TOKEN was loaded
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Failed to load the GITHUB_TOKEN from the .secrets file"
    exit 1
fi

# Adding all changes (it should be '.' or '--all', not '--a')
git add .

# Committing the changes
git commit -m "update"

# Pushing the changes
git push origin main