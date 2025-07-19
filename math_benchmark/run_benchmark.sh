#!/bin/bash

set -e

echo "Starting MATH Benchmark with Self-Certainty..."

# Check if we're in the correct directory
if [ ! -f "benchmark.py" ]; then
    echo "Error: benchmark.py not found. Please run this script from the math_benchmark directory."
    exit 1
fi

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found."
    exit 1
fi

echo "Configuration:"
echo "=============="
cat config.yaml
echo "=============="
echo ""

# Run the benchmark
echo "Running benchmark..."
python benchmark.py

echo "Benchmark completed!" 