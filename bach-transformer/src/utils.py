import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Token definitions
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NOTE_OFFSET = 3  # MIDI 0..127 -> token 3..130
VOCAB_SIZE = NOTE_OFFSET + 128  # 131


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


def subsequent_mask(sz: int) -> torch.Tensor:
    """Mask out subsequent positions (causal mask). Shape: (sz, sz) with -inf on masked positions (for adding to attn weights)."""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask
