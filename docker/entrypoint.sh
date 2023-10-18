#!/bin/bash

# Expect the GITHUB_TOKEN environment variable to be set
# You should pass this token when you run the container
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: The GITHUB_TOKEN environment variable is not set."
    exit 1
fi

# Clone the private repository
# Do not print the token in logs
echo "Cloning repository..."
git clone https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/your-username/your-repo.git /app/repo || exit 2
echo "Repository cloned successfully."

# Move to the repository directory
cd /app/repo

# Install project dependencies via Poetry
echo "Installing dependencies..."
poetry install --no-dev

# Run the command passed to the docker run
# This relies on your run.sh script being in your repository and correctly handling the commands
echo "Running command..."
exec "/app/repo/run.sh" "$@"
