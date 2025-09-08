import argparse
import json
import os

import torch

from .models import MiniMusicTransformer
from .utils import get_device, save_midi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    p.add_argument("--vocab", type=str, default="outputs/vocab.json")
    p.add_argument("--prime_tokens", type=str, default="60,64,67,3", help="CSV of prime tokens")
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--out_midi", type=str, default="outputs/sample.mid")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    
    # Load vocab
    print(f"Loading vocab from {args.vocab}")
    with open(args.vocab, 'r') as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    ivocab = vocab_data['ivocab']
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = MiniMusicTransformer(vocab_size=checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Parse prime tokens
    if args.prime_tokens:
        prime_list = [int(x.strip()) for x in args.prime_tokens.split(',') if x.strip()]
    else:
        prime_list = []
    
    # Convert MIDI notes to pitch tokens (add 5 offset)
    prime_tokens = []
    for token in prime_list:
        if 0 <= token <= 127:  # MIDI pitch
            prime_tokens.append(token + 5)  # Convert to pitch token
        elif token in [0, 1, 2, 3, 4]:  # Special tokens
            prime_tokens.append(token)
        else:
            print(f"Warning: Invalid token {token}, skipping")
    
    # Add BOS if not present
    if not prime_tokens or prime_tokens[0] != 1:  # BOS = 1
        prime_tokens = [1] + prime_tokens
    
    print(f"Prime tokens: {prime_tokens}")
    
    # Generate
    idx = torch.tensor(prime_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Generating {args.max_new} new tokens...")
    with torch.no_grad():
        generated = model.generate(
            idx, 
            max_new_tokens=args.max_new, 
            temperature=args.temperature, 
            top_k=args.top_k
        )
    
    # Convert to list
    tokens = generated.squeeze(0).cpu().tolist()
    print(f"Generated sequence length: {len(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}")
    
    # Save MIDI
    print(f"Saving MIDI to {args.out_midi}")
    save_midi(tokens, ivocab, args.out_midi)
    print(f"Generated MIDI saved to: {args.out_midi}")


if __name__ == "__main__":
    main()
