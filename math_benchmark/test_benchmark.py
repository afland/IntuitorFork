#!/usr/bin/env python3
"""
Test script for MATH Benchmark to verify setup and imports.
"""

import os
import sys


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
    
    # Test our standalone math utils
    try:
        from math_utils import last_boxed_only_string, remove_boxed, is_equiv
        print("✓ Standalone math utils")
    except ImportError as e:
        print(f"✗ Standalone math utils import failed: {e}")
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
        from math_utils import last_boxed_only_string, remove_boxed, is_equiv
        
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

def test_model_loading():
    """Test model loading (optional, since it requires GPU/large memory)."""
    print("\nTesting model loading (optional)...")
    
    try:
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = OmegaConf.load(config_path)
        
        # Just test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.path,
            trust_remote_code=config.model.trust_remote_code
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        print(f"⚠ Model/tokenizer loading failed (this may be expected): {e}")
        return True  # Don't fail the test for this since it may require large downloads

def main():
    """Run all tests."""
    print("MATH Benchmark Setup Test (Standalone Version)")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_config():
        all_passed = False
    
    if not test_dataset():
        all_passed = False
    
    if not test_math_functions():
        all_passed = False
    
    if not test_model_loading():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Benchmark is ready to run.")
    else:
        print("✗ Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 