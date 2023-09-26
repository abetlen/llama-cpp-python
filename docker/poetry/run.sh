#!/bin/bash

# Check if make is installed
if ! command -v make &> /dev/null; then
    echo "Error: make is not installed."
    exit 1
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "Error: uvicorn is not installed."
    exit 1
fi

# Set default HOST and PORT if not set
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

make build
uvicorn --factory llama_cpp.server.app:create_app --host $HOST --port $PORT
