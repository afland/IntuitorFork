#!/bin/bash

set -e

echo "Running MATH Benchmark Tests..."

# Check if we're in the correct directory
if [ ! -f "test_benchmark.py" ]; then
    echo "Error: test_benchmark.py not found. Please run this script from the math_benchmark directory."
    exit 1
fi

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Step 1: Testing imports and setup..."
python test_benchmark.py

if [ $? -eq 0 ]; then
    echo -e "\nStep 2: Running benchmark with test config..."
    echo "Using test configuration with small model and 3 samples..."
    
    # Backup original config and use test config
    if [ -f "config.yaml" ]; then
        cp config.yaml config.yaml.bak
    fi
    cp config_test.yaml config.yaml
    
    python benchmark.py
    
    # Restore original config
    if [ -f "config.yaml.bak" ]; then
        mv config.yaml.bak config.yaml
    fi
    
    echo "Test run completed successfully!"
else
    echo "Setup tests failed. Please fix the issues before running the benchmark."
    exit 1
fi 