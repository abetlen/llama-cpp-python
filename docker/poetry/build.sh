#!/bin/bash

# To run the script with OpenBLAS support, you can do:
#   export USE_OPENBLAS=true
#   ./build.sh
# And to run without OpenBLAS:
#   unset USE_OPENBLAS
#   ./build.sh

if [ -v USE_OPENBLAS ]; then
    docker build --build-arg USE_OPENBLAS=true -t llama-cpp-python:latest .
else
    docker build -t llama-cpp-python:latest .
fi
