#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")"

# Create necessary directories if they don't exist
mkdir -p tests
mkdir -p config
mkdir -p outputs/data
mkdir -p outputs/plots
mkdir -p simulations

# Set up Python environment
if [ -d "venv" ]; then
    echo "Using existing virtual environment..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    # Check if pip install succeeds
    if ! pip install -r requirements.txt; then
        echo "Error installing requirements. Please check requirements.txt and your internet connection."
        exit 1
    fi
    echo "Virtual environment created and dependencies installed."
fi

# Create __init__.py files for proper package structure
touch config/__init__.py
touch tests/__init__.py
touch simulations/__init__.py

# Check if specific test is provided
if [ $# -eq 1 ]; then
    test_file=$1

    # Check if the file exists
    if [ ! -f "$test_file" ]; then
        echo "Error: Test file $test_file does not exist"
        exit 1
    fi

    echo "Running specific test: $test_file"
    python -m unittest $test_file
    exit_code=$?
else
    # Run all tests using standard unittest module
    echo "Running all tests..."
    python -m unittest discover -s tests
    exit_code=$?
fi

# Print success/failure message
if [ $exit_code -eq 0 ]; then
    echo -e "\n\033[0;32m✅ All tests passed successfully!\033[0m"
else
    echo -e "\n\033[0;31m❌ Some tests failed. Please check the output above for details.\033[0m"
fi

# Exit with the same status as the Python script
exit $exit_code