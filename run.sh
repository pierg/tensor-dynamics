#!/bin/bash

# Simple command router for the arguments passed to the Docker container

case "$1" in
    command1)
        python script1.py
        ;;
    command2)
        python script2.py
        ;;
    *)
        echo "Usage: $0 {command1|command2}"
        exit 1
esac
