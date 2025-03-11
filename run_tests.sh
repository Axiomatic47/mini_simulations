#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")"

# Create necessary directories if they don't exist
mkdir -p tests
mkdir -p config
mkdir -p outputs/data
mkdir -p outputs/plots

# Set up Python environment
if [ -d "venv" ]; then
    echo "Using existing virtual environment..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Create __init__.py files for proper package structure
touch config/__init__.py
touch tests/__init__.py

# Run tests using standard unittest module
echo "Running tests..."
python -m unittest discover -s tests

# Exit with the same status as the Python script
exit $?