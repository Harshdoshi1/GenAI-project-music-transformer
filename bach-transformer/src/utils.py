import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Token definitions
PAD = 0
BOS = 1
EOS = 2
SEP = 3
REST = 4

# Legacy aliases for backward compatibility
PAD_ID = PAD
BOS_ID = BOS
EOS_ID = EOS
SEP_ID = SEP
REST_ID = REST
VOCAB_SIZE = 133  # 5 special tokens + 128 pitches


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Checkpoint:
    model_state: dict
    optimizer_state: dict
    epoch: int
    best_val_loss: float


def save_checkpoint(path: str, chkpt: Checkpoint):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(chkpt.__dict__, path)


def load_checkpoint(path: str) -> Checkpoint:
    data = torch.load(path, map_location="cpu")
    return Checkpoint(**data)


def count_params(model) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_midi(token_ids, ivocab, path: str):
    """Simple MIDI conversion: map pitches 36-96 to NOTE_ON/OFF with fixed 8th-note duration."""
    import pretty_midi
    
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Piano
    
    time = 0.0
    dur = 0.5  # 8th note duration in seconds
    velocity = 80
    
    for token in token_ids:
        if isinstance(token, torch.Tensor):
            token = token.item()
        
        # Skip special tokens
        if token in [PAD, BOS, EOS, SEP, REST]:
            if token == SEP:
                time += dur  # Advance time on separator
            continue
            
        # Convert pitch tokens (5-132) back to MIDI (0-127)
        if 5 <= token <= 132:
            pitch = token - 5
            # Only use pitches in range 36-96 (C2 to C6)
            if 36 <= pitch <= 96:
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=time,
                    end=time + dur
                )
                inst.notes.append(note)
        
        time += dur
    
    midi.instruments.append(inst)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    midi.write(path)


def subsequent_mask(sz: int) -> torch.Tensor:
    """Mask out subsequent positions (causal mask). Shape: (sz, sz) with -inf on masked positions (for adding to attn weights)."""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask
