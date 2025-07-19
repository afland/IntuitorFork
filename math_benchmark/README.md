# MATH Benchmark with Self-Certainty

This benchmark evaluates models on the MATH dataset and computes self-certainty scores for each response using the same logic as the verl-intuitor training.

## Structure

- `benchmark.py`: Main benchmark script
- `config.yaml`: Configuration for the benchmark
- `run_benchmark.sh`: Shell script to run the benchmark

## Features

- Uses vLLM for fast generation
- Computes self-certainty using separate forward pass (same as training)
- Evaluates correctness using MATH dataset scoring logic
- Saves results in JSON format with all requested fields

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