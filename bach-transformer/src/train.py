import argparse
import json
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .data import load_jsb, split_train_valid
from .models import MiniMusicTransformer
from .utils import PAD, set_seed, get_device, subsequent_mask, count_params


class ChunkedSequenceDataset(Dataset):
    """Dataset that chunks long sequences into fixed-length segments."""
    
    def __init__(self, sequences, seq_len):
        self.seq_len = seq_len
        self.chunks = []
        
        for seq in sequences:
            # Split sequence into chunks of seq_len + 1 (for input/target pairs)
            for i in range(0, len(seq) - seq_len, seq_len):
                chunk = seq[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.chunks.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = chunk[:-1]  # inputs
        y = chunk[1:]   # targets (shifted by 1)
        return x, y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/raw/jsb-chorales-16th.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save", type=str, default="checkpoints/model.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab_cache", type=str, default="outputs/vocab.json")
    p.add_argument("--val_split", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    
    print(f"Loading data from {args.data}")
    
    # Load sequences
    sequences, vocab, ivocab = load_jsb(args.data)
    print(f"Loaded {len(sequences)} sequences, vocab size: {vocab['vocab_size']}")
    
    # Split train/valid
    train_seqs, val_seqs = split_train_valid(sequences, valid_ratio=args.val_split, seed=args.seed)
    print(f"Train/valid split: {len(train_seqs)} train, {len(val_seqs)} valid")
    
    # Create chunked datasets
    train_ds = ChunkedSequenceDataset(train_seqs, args.seq_len)
    val_ds = ChunkedSequenceDataset(val_seqs, args.seq_len)
    print(f"Created {len(train_ds)} train chunks, {len(val_ds)} valid chunks")
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = MiniMusicTransformer(
        vocab_size=vocab['vocab_size'],
        max_len=args.seq_len,
    ).to(device)
    
    print(f"Model parameters: {count_params(model):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            
            # Create causal mask
            seq_len = x.size(1)
            mask = subsequent_mask(seq_len).to(device)
            
            # Forward pass
            logits = model(x, attn_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]")
        
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                
                seq_len = x.size(1)
                mask = subsequent_mask(seq_len).to(device)
                
                logits = model(x, attn_mask=mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                
                val_loss += loss.item()
                val_pbar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader)
        val_perplexity = math.exp(val_loss)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ppl={val_perplexity:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {val_loss:.4f}")
            
            # Save model
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'vocab_size': vocab['vocab_size'],
            }, args.save)
            
            # Save vocab
            os.makedirs(os.path.dirname(args.vocab_cache), exist_ok=True)
            with open(args.vocab_cache, 'w') as f:
                json.dump({'vocab': vocab, 'ivocab': ivocab}, f, indent=2)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save}")
    print(f"Vocab saved to: {args.vocab_cache}")


if __name__ == "__main__":
    main()
