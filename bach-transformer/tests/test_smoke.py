#!/usr/bin/env python3

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MiniMusicTransformer
from src.data import load_jsb
from src.utils import PAD, BOS, EOS, SEP, REST


def test_model_forward():
    """Test model forward pass with random input."""
    print("Testing model forward pass...")
    
    vocab_size = 133
    seq_len = 16
    batch_size = 2
    
    # Create model
    model = MiniMusicTransformer(vocab_size=vocab_size, max_len=64)
    model.eval()
    
    # Random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    print(f"✓ Model forward pass successful: {logits.shape}")
    return True


def test_model_generation():
    """Test model generation."""
    print("Testing model generation...")
    
    model = MiniMusicTransformer(vocab_size=133, max_len=64)
    model.eval()
    
    # Start with BOS token
    start_tokens = torch.tensor([[BOS]], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    
    assert generated.shape[0] == 1, "Batch size should be 1"
    assert generated.shape[1] == 11, "Should have 1 start + 10 new tokens"
    
    print(f"✓ Model generation successful: {generated.shape}")
    print(f"  Generated tokens: {generated.squeeze().tolist()}")
    return True


def test_data_loader():
    """Test data loader if file exists."""
    print("Testing data loader...")
    
    data_files = [
        "data/raw/jsb-chorales-16th.json",
        "data/raw/Jsb16thSeparated.json"
    ]
    
    found_file = None
    for filepath in data_files:
        if os.path.exists(filepath):
            found_file = filepath
            break
    
    if not found_file:
        print("⚠ No data file found, skipping data loader test")
        return True
    
    try:
        sequences, vocab, ivocab = load_jsb(found_file)
        
        assert len(sequences) > 0, "Should load at least one sequence"
        assert vocab['vocab_size'] == 133, "Vocab size should be 133"
        assert len(ivocab) == 133, "Inverse vocab should have 133 entries"
        
        # Check first few sequences
        for i, seq in enumerate(sequences[:3]):
            assert isinstance(seq, list), "Sequence should be a list"
            assert len(seq) > 0, "Sequence should not be empty"
            assert seq[0] == BOS, "Sequence should start with BOS"
            if len(seq) > 1:
                assert seq[-1] == EOS, "Sequence should end with EOS"
        
        print(f"✓ Data loader successful: {len(sequences)} sequences loaded")
        print(f"  Vocab size: {vocab['vocab_size']}")
        print(f"  First sequence length: {len(sequences[0])}")
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False


def test_token_constants():
    """Test token constant values."""
    print("Testing token constants...")
    
    assert PAD == 0, f"PAD should be 0, got {PAD}"
    assert BOS == 1, f"BOS should be 1, got {BOS}"
    assert EOS == 2, f"EOS should be 2, got {EOS}"
    assert SEP == 3, f"SEP should be 3, got {SEP}"
    assert REST == 4, f"REST should be 4, got {REST}"
    
    print("✓ Token constants are correct")
    return True


def main():
    """Run all smoke tests."""
    print("=== Bach Transformer Smoke Tests ===\n")
    
    tests = [
        test_token_constants,
        test_model_forward,
        test_model_generation,
        test_data_loader,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
