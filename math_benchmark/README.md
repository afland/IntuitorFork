# MATH Benchmark with Self-Certainty (Standalone Version)

This benchmark evaluates models on the MATH dataset and computes self-certainty scores for each response using vLLM for inference and PyTorch for logits recomputation. This is a standalone implementation that **does not depend on verl** or **flash-attn**.

## Structure

- `benchmark.py`: Main benchmark script (standalone implementation)
- `math_utils.py`: Standalone math evaluation utilities
- `test_benchmark.py`: Test script to verify setup
- `config.yaml`: Configuration for the benchmark
- `config_test.yaml`: Test configuration with smaller model
- `run_benchmark.sh`: Shell script to run the benchmark
- `test_run.sh`: Shell script to run tests

## Features

- **No verl dependency**: Standalone implementation using only standard libraries
- **No flash-attn dependency**: Avoids ABI compatibility issues
- Uses vLLM for fast generation
- Computes self-certainty using separate forward pass (same logic as training)
- Evaluates correctness using MATH dataset scoring logic
- Saves results in JSON format with all requested fields

## Dependencies

The standalone version only requires:
- PyTorch
- transformers
- vllm
- datasets
- omegaconf
- tqdm
- numpy

## Usage

### Testing the Setup
First, test that everything is working correctly:
```bash
cd math_benchmark
bash test_run.sh
```

### Running the Full Benchmark
```bash
cd math_benchmark
bash run_benchmark.sh
```

Note: Make sure to update the model path in `config.yaml` to the model you want to benchmark.

## Output

The benchmark saves results to `results.json` with the following fields for each sample:
- `question`: The math problem
- `gt_response`: Ground truth answer
- `model_response`: Model's generated response
- `is_model_response_correct`: Boolean indicating if the response is correct
- `model_response_self_certainty`: Self-certainty score (mean over response tokens)

## Key Implementation Details

- **Math Evaluation**: Uses the same normalization and equivalence checking logic as the original MATH dataset
- **Self-Certainty**: Computed as `logsumexp(logits) - mean(logits)` for each token, then averaged over response tokens
- **Two-Stage Process**: 
  1. vLLM generates responses quickly
  2. Separate model forward pass computes logits and self-certainty scores
- **Memory Efficient**: Processes in batches and handles variable-length sequences properly 