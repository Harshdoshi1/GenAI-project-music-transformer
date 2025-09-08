#!/usr/bin/env python3

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_jsb, split_train_valid
from src.models import MiniMusicTransformer
from src.utils import PAD, set_seed, get_device, subsequent_mask, count_params


class SimpleDataset(Dataset):
    def __init__(self, sequences, seq_len=128):
        self.chunks = []
        for seq in sequences:
            for i in range(0, len(seq) - seq_len, seq_len // 2):
                chunk = seq[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.chunks.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


def quick_train():
    """Quick training for demo purposes."""
    print("🎵 Bach Transformer - Quick Training")
    print("===================================")
    
    # Setup
    set_seed(42)
    device = get_device()
    seq_len = 128
    batch_size = 8
    epochs = 2
    
    # Find data file
    data_files = ["data/raw/jsb-chorales-16th.json", "data/raw/Jsb16thSeparated.json"]
    data_path = None
    for f in data_files:
        if os.path.exists(f):
            data_path = f
            break
    
    if not data_path:
        print("❌ No data file found!")
        return False
    
    print(f"📚 Loading data from {data_path}")
    
    # Load data
    sequences, vocab, ivocab = load_jsb(data_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Quick split
    train_seqs, val_seqs = split_train_valid(sequences[:100], valid_ratio=0.2)  # Use first 100 for speed
    
    # Create datasets
    train_ds = SimpleDataset(train_seqs, seq_len)
    val_ds = SimpleDataset(val_seqs, seq_len)
    print(f"Created {len(train_ds)} train chunks, {len(val_ds)} val chunks")
    
    if len(train_ds) == 0:
        print("❌ No training data available!")
        return False
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None
    
    # Model
    model = MiniMusicTransformer(
        vocab_size=vocab['vocab_size'],
        d_model=128,  # Smaller for speed
        n_heads=4,
        n_layers=2,   # Fewer layers for speed
        max_len=seq_len,
    ).to(device)
    
    print(f"Model parameters: {count_params(model):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab['vocab_size'],
        'epoch': epochs,
    }, "checkpoints/model.pt")
    
    with open("outputs/vocab.json", 'w') as f:
        json.dump({'vocab': vocab, 'ivocab': ivocab}, f, indent=2)
    
    print("✅ Quick training completed!")
    print("📁 Model saved to: checkpoints/model.pt")
    print("📁 Vocab saved to: outputs/vocab.json")
    
    return True


def main():
    try:
        success = quick_train()
        if success:
            print("\n🚀 You can now run:")
            print("   python app.py  # Web interface")
            print("   python -m src.generate --prime_tokens '60,64,67,3'")
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
