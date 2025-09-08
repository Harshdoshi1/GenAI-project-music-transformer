import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data.jsb_loader import JSBChoralesDataset
from .models import MiniMusicTransformer
from .utils import Checkpoint, PAD_ID, get_device, save_checkpoint, set_seed, subsequent_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    dataset = JSBChoralesDataset(args.data_path, max_len=args.max_len)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=JSBChoralesDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=JSBChoralesDataset.collate_fn)

    model = MiniMusicTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        total_loss = 0.0
        for inp, tgt in pbar:
            inp = inp.to(device)
            tgt = tgt.to(device)
            T = inp.size(1)
            mask = subsequent_mask(T).to(device)

            logits = model(inp, attn_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]"):
                inp = inp.to(device)
                tgt = tgt.to(device)
                T = inp.size(1)
                mask = subsequent_mask(T).to(device)
                logits = model(inp, attn_mask=mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                args.checkpoint,
                Checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optim.state_dict(),
                    epoch=epoch,
                    best_val_loss=best_val,
                ),
            )
        print(f"Epoch {epoch} done. Train loss: {total_loss / max(1, len(train_loader)):.4f} | Val loss: {val_loss:.4f} | Best: {best_val:.4f}")


if __name__ == "__main__":
    main()
