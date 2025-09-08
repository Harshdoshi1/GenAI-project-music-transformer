import json
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from ..utils import BOS_ID, EOS_ID, NOTE_OFFSET, PAD_ID


def _flatten_jsb_chorale(chorale) -> List[int]:
    """Flatten a JSB chorale structure to a simple note stream.
    Supports formats:
    - List[List[int]]: sequence of notes per timestep (monophonic) -> pass through
    - List[List[List[int]]]: voices x timesteps x possibly-chords -> interleave voices per timestep
    - Dict with key 'notes' -> directly use list of ints
    Any values outside 0..127 are dropped.
    """
    if isinstance(chorale, dict) and "notes" in chorale:
        seq = chorale["notes"]
        return [int(n) for n in seq if 0 <= int(n) <= 127]

    if not chorale:
        return []

    # If it's already a flat list
    if isinstance(chorale[0], int):
        return [int(n) for n in chorale if 0 <= int(n) <= 127]

    # If it's list of timesteps with chord lists
    if isinstance(chorale[0], list) and chorale and chorale[0] and isinstance(chorale[0][0], int):
        # Take the first note from each chord
        out = []
        for chord in chorale:
            if chord:
                n = int(chord[0])
                if 0 <= n <= 127:
                    out.append(n)
        return out

    # If it's voices x timesteps
    if isinstance(chorale[0], list) and chorale and chorale[0] and isinstance(chorale[0][0], list):
        voices = chorale
        T = min(len(v) for v in voices)
        out = []
        for t in range(T):
            for v in range(len(voices)):
                step = voices[v][t]
                if isinstance(step, list) and step:
                    n = int(step[0])
                elif isinstance(step, int):
                    n = int(step)
                else:
                    continue
                if 0 <= n <= 127:
                    out.append(n)
        return out

    # Fallback: flatten recursively
    out = []
    def _rec(x):
        if isinstance(x, list):
            for y in x:
                _rec(y)
        elif isinstance(x, int) and 0 <= x <= 127:
            out.append(int(x))
    _rec(chorale)
    return out


def load_sequences(json_path: str) -> List[List[int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seqs: List[List[int]] = []
    if isinstance(data, list):
        for item in data:
            seqs.append(_flatten_jsb_chorale(item))
    elif isinstance(data, dict):
        # Could be {'chorales': [...]} or similar
        for key in ("chorales", "train", "valid", "test", "data"):
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    seqs.append(_flatten_jsb_chorale(item))
        if not seqs and "notes" in data:
            seqs.append(_flatten_jsb_chorale(data))
    else:
        raise ValueError("Unsupported JSON structure for JSB dataset")

    # Filter empty
    return [s for s in seqs if len(s) > 0]


def encode_sequence(notes: Sequence[int], max_len: int) -> torch.Tensor:
    # map MIDI 0..127 -> tokens 3..130, wrap with BOS/EOS, clip/pad
    tokens = [BOS_ID] + [NOTE_OFFSET + int(n) for n in notes if 0 <= int(n) <= 127] + [EOS_ID]
    tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)


class JSBChoralesDataset(Dataset):
    def __init__(self, json_path: str, max_len: int = 512):
        self.seqs = load_sequences(json_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = encode_sequence(self.seqs[idx], self.max_len)
        return x

    @staticmethod
    def collate_fn(batch: List[torch.Tensor]):
        # Pad to max length in batch
        max_len = max(x.size(0) for x in batch)
        padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, : x.size(0)] = x
        # Inputs and targets (shifted)
        inp = padded[:, :-1]
        tgt = padded[:, 1:]
        return inp, tgt
