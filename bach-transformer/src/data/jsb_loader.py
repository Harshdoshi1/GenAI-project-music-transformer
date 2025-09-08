import json
import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

# Special tokens
PAD = 0
BOS = 1
EOS = 2
SEP = 3
REST = 4


def load_jsb(path: str) -> Tuple[List[List[int]], Dict, Dict]:
    """Load JSB chorales from jsb-chorales-16th.json or Jsb16thSeparated.json.
    
    Returns:
        sequences: List of integer sequences (SATB+SEP encoding, ≤2048 length)
        vocab: Dict with token mappings and vocab_size
        ivocab: Inverse vocab mapping (int -> str)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Build vocabulary: PAD=0, BOS=1, EOS=2, SEP=3, REST=4, then pitches 0-127 -> 5-132
    vocab = {
        "PAD": PAD,
        "BOS": BOS, 
        "EOS": EOS,
        "SEP": SEP,
        "REST": REST,
    }
    
    # Add pitch tokens (MIDI 0-127 -> tokens 5-132)
    for pitch in range(128):
        vocab[f"PITCH_{pitch}"] = pitch + 5
    
    vocab["vocab_size"] = 133  # 5 special + 128 pitches
    
    # Inverse vocab
    ivocab = {v: k for k, v in vocab.items() if isinstance(v, int)}
    
    sequences = []
    
    # Handle different JSON formats
    chorales_data = []
    if isinstance(data, list):
        chorales_data = data
    elif isinstance(data, dict):
        # For separated datasets, combine train/valid/test if they exist
        if any(key in data for key in ["train", "valid", "test"]):
            for key in ["train", "valid", "test"]:
                if key in data and isinstance(data[key], list):
                    chorales_data.extend(data[key])
        else:
            # Try other common keys
            for key in ["chorales", "data"]:
                if key in data and isinstance(data[key], list):
                    chorales_data = data[key]
                    break
        if not chorales_data:
            raise ValueError(f"Could not find chorales data in JSON structure. Keys: {list(data.keys())}")
    else:
        raise ValueError("Unsupported JSON format")
    
    for chorale in chorales_data:
        seq = _encode_chorale_satb_sep(chorale, vocab)
        if seq and len(seq) <= 2048:  # Filter empty and overly long sequences
            sequences.append(seq)
    
    return sequences, vocab, ivocab


def _encode_chorale_satb_sep(chorale, vocab: Dict) -> List[int]:
    """Convert chorale to SATB+SEP sequence.
    
    Expected format: List of timesteps, where each timestep is [S, A, T, B] pitches.
    Order per timestep: S, A, T, B, SEP
    """
    # The data format is: list of timesteps, each timestep is [S, A, T, B]
    if not isinstance(chorale, list) or len(chorale) == 0:
        return []
    
    # Check if it's the expected format: list of [S,A,T,B] per timestep
    if not isinstance(chorale[0], list) or len(chorale[0]) != 4:
        return []
    
    sequence = [BOS]  # Start with BOS token
    
    for timestep in chorale:
        if len(timestep) != 4:
            continue
            
        # Add S, A, T, B for this timestep
        for voice_idx in range(4):
            note = timestep[voice_idx]
            
            if note is None or note < 0:
                # Rest or invalid note
                sequence.append(REST)
            elif 0 <= note <= 127:
                # Valid MIDI pitch -> token
                sequence.append(note + 5)  # Pitch tokens start at 5
            else:
                # Invalid pitch, treat as rest
                sequence.append(REST)
        
        # Add separator after SATB
        sequence.append(SEP)
    
    sequence.append(EOS)  # End with EOS token
    return sequence


def split_train_valid(seqs: List[List[int]], valid_ratio: float = 0.1, seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
    """Split sequences into train and validation sets."""
    random.seed(seed)
    seqs_copy = seqs.copy()
    random.shuffle(seqs_copy)
    
    valid_size = max(1, int(len(seqs_copy) * valid_ratio))
    valid_seqs = seqs_copy[:valid_size]
    train_seqs = seqs_copy[valid_size:]
    
    return train_seqs, valid_seqs


class JSBChoralesDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len: int = 512):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Truncate if too long
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        return torch.tensor(seq, dtype=torch.long)

    @staticmethod
    def collate_fn(batch: List[torch.Tensor]):
        # Pad to max length in batch
        max_len = max(x.size(0) for x in batch)
        padded = torch.full((len(batch), max_len), PAD, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, : x.size(0)] = x
        # Inputs and targets (shifted)
        inp = padded[:, :-1]
        tgt = padded[:, 1:]
        return inp, tgt


def main():
    """Load JSB data from data/raw/ and print statistics."""
    data_dir = "data/raw"
    
    # Try to find JSB files
    possible_files = [
        "jsb-chorales-16th.json",
        "Jsb16thSeparated.json"
    ]
    
    found_file = None
    for filename in possible_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            found_file = filepath
            break
    
    if not found_file:
        print(f"No JSB dataset found in {data_dir}/")
        print(f"Expected files: {possible_files}")
        return
    
    print(f"Loading JSB data from: {found_file}")
    
    try:
        sequences, vocab, ivocab = load_jsb(found_file)
        
        print(f"Loaded {len(sequences)} chorales")
        print(f"Vocabulary size: {vocab['vocab_size']}")
        
        if sequences:
            seq_lengths = [len(seq) for seq in sequences]
            print(f"Sequence lengths - min: {min(seq_lengths)}, max: {max(seq_lengths)}, avg: {sum(seq_lengths)/len(seq_lengths):.1f}")
            
            # Show first few tokens of first sequence
            if len(sequences[0]) > 0:
                print(f"First sequence preview: {sequences[0][:20]}...")
        
        # Test train/valid split
        train_seqs, valid_seqs = split_train_valid(sequences, valid_ratio=0.1)
        print(f"Train/valid split: {len(train_seqs)} train, {len(valid_seqs)} valid")
        
    except Exception as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    main()
