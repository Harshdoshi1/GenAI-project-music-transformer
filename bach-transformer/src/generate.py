import argparse
import os

import numpy as np
import pretty_midi
import torch

from .models import MiniMusicTransformer
from .utils import BOS_ID, EOS_ID, NOTE_OFFSET, PAD_ID, VOCAB_SIZE, get_device, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_midi", type=str, default="outputs/sample.mid")
    p.add_argument("--seed_tokens", type=int, nargs="*", default=None, help="Optional list of starting MIDI notes (0-127)")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    return p.parse_args()


def tokens_to_midi(tokens: np.ndarray, path: str):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    time = 0.0
    dur = 0.25  # quarter of a beat for demo
    velocity = 80
    for t in tokens:
        if t == PAD_ID or t == BOS_ID:
            continue
        if t == EOS_ID:
            break
        if t >= NOTE_OFFSET:
            pitch = int(t - NOTE_OFFSET)
            if 0 <= pitch <= 127:
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=time, end=time + dur)
                inst.notes.append(note)
                time += dur
    midi.instruments.append(inst)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    midi.write(path)


def main():
    args = parse_args()
    device = get_device()

    # Build a model with default sizes (must match training sizes for best results)
    model = MiniMusicTransformer().to(device)
    chkpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(chkpt.model_state)
    model.eval()

    if args.seed_tokens:
        seed = [BOS_ID] + [NOTE_OFFSET + int(n) for n in args.seed_tokens]  # BOS + notes
    else:
        seed = [BOS_ID]

    idx = torch.tensor(seed, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    out = model.generate(idx, max_new_tokens=args.max_len, temperature=args.temperature, top_k=args.top_k)
    tokens = out.squeeze(0).detach().cpu().numpy()

    tokens_to_midi(tokens, args.out_midi)
    print(f"Wrote {args.out_midi}")


if __name__ == "__main__":
    main()
