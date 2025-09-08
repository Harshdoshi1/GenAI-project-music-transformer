#!/usr/bin/env python3

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_json_load():
    """Test loading the JSON file directly."""
    try:
        with open('data/raw/Jsb16thSeparated.json', 'r') as f:
            data = json.load(f)
        print(f"✓ JSON loaded successfully")
        print(f"  Keys: {list(data.keys())}")
        
        if 'test' in data:
            test_data = data['test']
            print(f"  Test data length: {len(test_data)}")
            if test_data:
                first_chorale = test_data[0]
                print(f"  First chorale: {len(first_chorale)} timesteps")
                if first_chorale:
                    print(f"  First timestep: {first_chorale[0]} (SATB)")
        return data
    except Exception as e:
        print(f"✗ JSON load failed: {e}")
        return None

def test_loader():
    """Test the JSB loader."""
    try:
        from src.data.jsb_loader import load_jsb, split_train_valid
        
        print("\n--- Testing JSB Loader ---")
        sequences, vocab, ivocab = load_jsb('data/raw/Jsb16thSeparated.json')
        
        print(f"✓ Loaded {len(sequences)} sequences")
        print(f"  Vocab size: {vocab['vocab_size']}")
        
        if sequences:
            seq_lengths = [len(seq) for seq in sequences]
            print(f"  Sequence lengths - min: {min(seq_lengths)}, max: {max(seq_lengths)}, avg: {sum(seq_lengths)/len(seq_lengths):.1f}")
            print(f"  First sequence preview: {sequences[0][:20]}...")
            
            # Test train/valid split
            train_seqs, valid_seqs = split_train_valid(sequences, valid_ratio=0.1)
            print(f"  Train/valid split: {len(train_seqs)} train, {len(valid_seqs)} valid")
            
        return True
    except Exception as e:
        print(f"✗ Loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test the PyTorch dataset."""
    try:
        from src.data.jsb_loader import load_jsb, JSBChoralesDataset
        
        print("\n--- Testing Dataset ---")
        sequences, vocab, ivocab = load_jsb('data/raw/Jsb16thSeparated.json')
        
        if sequences:
            dataset = JSBChoralesDataset(sequences[:5], max_len=512)  # Test with first 5
            print(f"✓ Dataset created with {len(dataset)} items")
            
            # Test getting an item
            item = dataset[0]
            print(f"  First item shape: {item.shape}")
            print(f"  First item preview: {item[:10]}")
            
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== JSB Loader Test ===")
    
    # Test 1: JSON loading
    data = test_json_load()
    
    if data:
        # Test 2: Loader function
        loader_ok = test_loader()
        
        if loader_ok:
            # Test 3: Dataset class
            test_dataset()
    
    print("\n=== Test Complete ===")
