#!/usr/bin/env python3
"""
Test script for MATH Benchmark to verify setup and imports.
"""

import os
import sys

# Add the parent directory to the path to import from verl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import vllm
        print(f"✓ vLLM {vllm.__version__}")
    except ImportError as e:
        print(f"✗ vLLM import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        from omegaconf import OmegaConf
        print("✓ OmegaConf")
    except ImportError as e:
        print(f"✗ OmegaConf import failed: {e}")
        return False
    
    # Test verl imports
    try:
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv
        print("✓ VERL math utils")
    except ImportError as e:
        print(f"✗ VERL math utils import failed: {e}")
        return False
    
    try:
        from verl.utils.torch_functional import self_certainty_from_logits
        print("✓ VERL torch functional")
    except ImportError as e:
        print(f"✗ VERL torch functional import failed: {e}")
        return False
    
    try:
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        print("✓ VERL DataParallelPPOActor")
    except ImportError as e:
        print(f"✗ VERL DataParallelPPOActor import failed: {e}")
        return False
    
    try:
        from verl import DataProto
        print("✓ VERL DataProto")
    except ImportError as e:
        print(f"✗ VERL DataProto import failed: {e}")
        return False
    
    return True

def test_config():
    """Test config loading."""
    print("\nTesting config...")
    
    try:
        from omegaconf import OmegaConf
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = OmegaConf.load(config_path)
        print("✓ Config loaded successfully")
        print(f"  Model: {config.model.path}")
        print(f"  Dataset: {config.data.dataset_name}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset...")
    
    try:
        import datasets
        from omegaconf import OmegaConf
        
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = OmegaConf.load(config_path)
        
        # Load just a small sample to test
        dataset = datasets.load_dataset(
            config.data.dataset_name, 
            split=f"{config.data.split}[:5]",
            trust_remote_code=True
        )
        print(f"✓ Dataset loaded successfully ({len(dataset)} samples)")
        print(f"  Sample problem: {dataset[0]['problem'][:100]}...")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_math_functions():
    """Test math evaluation functions."""
    print("\nTesting math functions...")
    
    try:
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv
        
        # Test with a simple example
        test_solution = "The answer is \\boxed{42}"
        boxed_string = last_boxed_only_string(test_solution)
        answer = remove_boxed(boxed_string)
        is_correct = is_equiv(answer, "42")
        
        print(f"✓ Math functions work correctly")
        print(f"  Extracted answer: {answer}")
        print(f"  Is correct: {is_correct}")
        return True
    except Exception as e:
        print(f"✗ Math functions failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MATH Benchmark Setup Test")
    print("=" * 30)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_config():
        all_passed = False
    
    if not test_dataset():
        all_passed = False
    
    if not test_math_functions():
        all_passed = False
    
    print("\n" + "=" * 30)
    if all_passed:
        print("✓ All tests passed! Benchmark is ready to run.")
    else:
        print("✗ Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 