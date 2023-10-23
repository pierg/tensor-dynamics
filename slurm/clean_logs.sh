#!/bin/bash

find . -type f -name "*.out" -exec rm -f {} \;
find . -type f -name "*.err" -exec rm -f {} \;
echo "Log files have been deleted."