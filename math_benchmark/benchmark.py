#!/usr/bin/env python3
"""
MATH Benchmark with Self-Certainty

This script benchmarks models on the MATH dataset and computes self-certainty scores
without relying on verl dependencies.
"""

import json
import logging
import os
import sys
from typing import Dict, List, Any
import warnings

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Import our standalone math utilities
from math_utils import last_boxed_only_string, remove_boxed, is_equiv

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("vllm").setLevel(logging.ERROR)


def extract_solution(solution_str: str) -> str:
    """Extract the solution from a boxed string."""
    return remove_boxed(last_boxed_only_string(solution_str))


def compute_math_score(solution_str: str, ground_truth: str) -> bool:
    """Compute if the solution is correct using MATH dataset logic."""
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            return is_equiv(answer, ground_truth)
    except Exception as e:
        print(f"Error computing score: {e}")
    return False


def create_math_prompt(problem: str) -> str:
    """Create a prompt for the MATH dataset following the same format as training."""
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    return f"{problem} {instruction}"


def self_certainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate self-certainty from logits: logsumexp - mean."""
    return torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities from logits and labels."""
    # Ensure we have the right shape
    assert logits.shape[:-1] == labels.shape, f"Shape mismatch: {logits.shape} vs {labels.shape}"
    
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather the log probabilities for the target labels
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


class MATHBenchmark:
    def __init__(self, config_path: str):
        """Initialize the benchmark with configuration."""
        self.config = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.path,
            trust_remote_code=self.config.model.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize vLLM engine for generation
        self.llm = self._init_vllm()
        
        # Initialize model for self-certainty computation
        self.model = self._init_model()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _init_vllm(self) -> LLM:
        """Initialize vLLM engine for generation."""
        return LLM(
            model=self.config.model.path,
            tensor_parallel_size=self.config.rollout.tensor_model_parallel_size,
            dtype=self.config.rollout.dtype,
            enforce_eager=self.config.rollout.enforce_eager,
            gpu_memory_utilization=self.config.rollout.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            max_num_batched_tokens=self.config.rollout.max_num_batched_tokens,
            enable_chunked_prefill=self.config.rollout.enable_chunked_prefill,
            disable_log_stats=self.config.rollout.disable_log_stats,
            trust_remote_code=self.config.model.trust_remote_code,
        )
    
    def _init_model(self):
        """Initialize model for self-certainty computation."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            torch_dtype=getattr(torch, self.config.rollout.dtype),
            trust_remote_code=self.config.model.trust_remote_code,
            device_map="auto"
        )
        model.eval()
        return model
    
    def _load_dataset(self):
        """Load the MATH dataset."""
        dataset = datasets.load_dataset(
            self.config.data.dataset_name, 
            split=self.config.data.split,
            trust_remote_code=True
        )
        
        if self.config.data.max_samples:
            dataset = dataset.select(range(min(self.config.data.max_samples, len(dataset))))
            
        return dataset
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses using vLLM."""
        sampling_params = SamplingParams(
            temperature=self.config.rollout.temperature,
            top_k=self.config.rollout.top_k,
            top_p=self.config.rollout.top_p,
            max_tokens=self.config.rollout.max_tokens,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def compute_self_certainty_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute self-certainty scores for a batch of responses."""
        # Prepare input sequences (prompt + response)
        input_sequences = []
        response_lengths = []
        
        for prompt, response in zip(prompts, responses):
            # Tokenize prompt and response separately
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Combine prompt and response
            full_sequence = prompt_tokens + response_tokens
            input_sequences.append(full_sequence)
            response_lengths.append(len(response_tokens))
        
        # Find max length for padding
        max_length = max(len(seq) for seq in input_sequences)
        
        # Pad sequences and create attention masks
        padded_sequences = []
        attention_masks = []
        
        for seq in input_sequences:
            padding_length = max_length - len(seq)
            # Pad on the left
            padded_seq = [self.tokenizer.pad_token_id] * padding_length + seq
            attention_mask = [0] * padding_length + [1] * len(seq)
            
            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_sequences, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)
        
        # Compute logits using the model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Compute self-certainty for response tokens only
        self_certainty_scores = []
        for i, response_length in enumerate(response_lengths):
            if response_length > 0:
                # Get logits for response tokens
                # Response tokens start at position: total_length - response_length
                total_length = attention_mask[i].sum().item()
                start_pos = total_length - response_length
                end_pos = total_length
                
                # Extract response logits (exclude the last token since we can't predict beyond it)
                response_logits = logits[i, start_pos:end_pos-1]  # [response_length-1, vocab_size]
                
                if response_logits.size(0) > 0:
                    # Compute self-certainty for each token and take the mean
                    certainty = self_certainty_from_logits(response_logits)  # [response_length-1]
                    mean_certainty = certainty.mean().item()
                    self_certainty_scores.append(mean_certainty)
                else:
                    self_certainty_scores.append(0.0)
            else:
                self_certainty_scores.append(0.0)
        
        return self_certainty_scores
    
    def run_benchmark(self) -> List[Dict[str, Any]]:
        """Run the complete benchmark."""
        results = []
        batch_size = self.config.output.batch_size
        
        print(f"Running benchmark on {len(self.dataset)} samples...")
        
        for i in tqdm(range(0, len(self.dataset), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(self.dataset))
            batch_data = self.dataset[i:batch_end]
            
            # Prepare prompts
            if isinstance(batch_data['problem'], list):
                problems = batch_data['problem']
                ground_truths = batch_data['solution']
            else:
                problems = [batch_data['problem']]
                ground_truths = [batch_data['solution']]
            
            prompts = [create_math_prompt(problem) for problem in problems]
            
            # Generate responses
            responses = self.generate_responses(prompts)
            
            # Compute self-certainty scores
            self_certainty_scores = self.compute_self_certainty_batch(prompts, responses)
            
            # Evaluate correctness and compile results
            for j, (problem, gt_solution, response, certainty) in enumerate(
                zip(problems, ground_truths, responses, self_certainty_scores)
            ):
                # Extract ground truth answer
                gt_answer = extract_solution(gt_solution)
                
                # Check if response is correct
                is_correct = compute_math_score(response, gt_answer)
                
                result = {
                    "question": problem,
                    "gt_response": gt_answer,
                    "model_response": response,
                    "is_model_response_correct": is_correct,
                    "model_response_self_certainty": certainty
                }
                results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        output_file = self.config.output.results_file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary statistics
        correct_count = sum(1 for r in results if r["is_model_response_correct"])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        avg_certainty = sum(r["model_response_self_certainty"] for r in results) / total_count
        
        print(f"\nBenchmark Results:")
        print(f"Total samples: {total_count}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average self-certainty: {avg_certainty:.4f}")
        print(f"Results saved to: {output_file}")


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    benchmark = MATHBenchmark(config_path)
    results = benchmark.run_benchmark()
    benchmark.save_results(results)


if __name__ == "__main__":
    main() 