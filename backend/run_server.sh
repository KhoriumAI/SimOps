#!/bin/bash
# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Current Directory: $(pwd)"
echo "Files in directory:"
ls -F

if [ -f "api_server.py" ]; then
    echo "Starting api_server.py..."
    # Use the venv if it exists in the parent directory
    if [ -d "../venv" ]; then
        ../venv/bin/python3 api_server.py
    else
        python3 api_server.py
    fi
else
    echo "ERROR: api_server.py not found in $SCRIPT_DIR"
fi
