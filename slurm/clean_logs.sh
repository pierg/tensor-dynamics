#!/bin/bash

find . -type f -name "*.out" -exec rm -f {} \;
echo "Log files have been deleted."