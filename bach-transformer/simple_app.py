#!/usr/bin/env python3

import os
import sys
import subprocess

def install_if_missing():
    """Install packages if missing."""
    packages = {
        'torch': 'torch',
        'gradio': 'gradio', 
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'pretty_midi': 'pretty_midi'
    }
    
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install dependencies
install_if_missing()

import json
import torch
import torch.nn as nn
import gradio as gr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pretty_midi

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Token constants
PAD, BOS, EOS, SEP, REST = 0, 1, 2, 3, 4

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_midi(tokens, path):
    """Simple MIDI conversion."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    
    time = 0.0
    dur = 0.5
    velocity = 80
    
    for token in tokens:
        if isinstance(token, torch.Tensor):
            token = token.item()
        
        if token in [PAD, BOS, EOS, SEP, REST]:
            if token == SEP:
                time += dur
            continue
            
        if 5 <= token <= 132:
            pitch = token - 5
            if 36 <= pitch <= 96:
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=time, end=time + dur)
                inst.notes.append(note)
        
        time += dur
    
    midi.instruments.append(inst)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    midi.write(path)

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=133, d_model=128, n_heads=4, n_layers=2, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    
    def forward(self, idx, attn_mask=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x, mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
        self.eval()
        for _ in range(max_new_tokens):
            if idx.size(1) > self.max_len:
                idx = idx[:, -self.max_len:]
            T = idx.size(1)
            mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float("-inf"))
            
            logits = self(idx, attn_mask=mask)[:, -1, :] / max(1e-6, temperature)
            
            if top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

def load_data():
    """Load JSB data if available."""
    data_files = ["data/raw/jsb-chorales-16th.json", "data/raw/Jsb16thSeparated.json"]
    
    for filepath in data_files:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            chorales_data = []
            if isinstance(data, list):
                chorales_data = data
            elif isinstance(data, dict):
                for key in ["train", "valid", "test"]:
                    if key in data and isinstance(data[key], list):
                        chorales_data.extend(data[key])
            
            sequences = []
            for chorale in chorales_data[:50]:  # Use first 50 for demo
                if isinstance(chorale, list) and len(chorale) > 0 and isinstance(chorale[0], list) and len(chorale[0]) == 4:
                    sequence = [BOS]
                    for timestep in chorale:
                        for note in timestep:
                            if 0 <= note <= 127:
                                sequence.append(note + 5)
                            else:
                                sequence.append(REST)
                        sequence.append(SEP)
                    sequence.append(EOS)
                    if len(sequence) <= 512:
                        sequences.append(sequence)
            
            return sequences
    return []

class SimpleDataset(Dataset):
    def __init__(self, sequences, seq_len=64):
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
    """Quick training."""
    print("🎵 Quick Training Started...")
    set_seed(42)
    device = get_device()
    
    sequences = load_data()
    if not sequences:
        return False, "No data found. Please add JSB dataset to data/raw/"
    
    print(f"Loaded {len(sequences)} sequences")
    
    dataset = SimpleDataset(sequences, seq_len=64)
    if len(dataset) == 0:
        return False, "No training data created"
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = MiniTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    model.train()
    for epoch in range(2):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/2")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/simple_model.pt")
    
    print("✅ Training completed!")
    return True, "Training successful"

def generate_music(prime_tokens_str, max_new, temperature, top_k):
    """Generate music."""
    try:
        device = get_device()
        
        # Load or train model
        model = MiniTransformer().to(device)
        if os.path.exists("checkpoints/simple_model.pt"):
            model.load_state_dict(torch.load("checkpoints/simple_model.pt", map_location=device))
            status = "Using existing model"
        else:
            success, msg = quick_train()
            if not success:
                return None, f"Training failed: {msg}"
            model.load_state_dict(torch.load("checkpoints/simple_model.pt", map_location=device))
            status = "Trained new model"
        
        model.eval()
        
        # Parse prime tokens
        prime_tokens = [BOS]
        if prime_tokens_str.strip():
            for token_str in prime_tokens_str.split(','):
                token = int(token_str.strip())
                if 0 <= token <= 127:
                    prime_tokens.append(token + 5)
                elif token in [0, 1, 2, 3, 4]:
                    prime_tokens.append(token)
        
        # Generate
        idx = torch.tensor(prime_tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(idx, max_new_tokens=int(max_new), temperature=float(temperature), top_k=int(top_k))
        tokens = generated.squeeze(0).cpu().tolist()
        
        # Save MIDI
        output_path = "outputs/generated.mid"
        os.makedirs("outputs", exist_ok=True)
        save_midi(tokens, output_path)
        
        return output_path, f"✅ {status}. Generated {len(tokens)} tokens. MIDI saved!"
        
    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}\n{traceback.format_exc()}"

def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(title="Bach Transformer") as demo:
        gr.Markdown("# 🎵 Bach Transformer Music Generator")
        
        # Check for data
        data_files = ["data/raw/jsb-chorales-16th.json", "data/raw/Jsb16thSeparated.json"]
        data_exists = any(os.path.exists(f) for f in data_files)
        
        if not data_exists:
            gr.Markdown("## ⚠️ No Dataset Found")
            gr.Markdown("Please add JSB dataset to `data/raw/` directory:")
            gr.Markdown("- `jsb-chorales-16th.json` or `Jsb16thSeparated.json`")
        else:
            gr.Markdown("## 🎼 Generate Music")
            
            with gr.Row():
                with gr.Column():
                    prime_tokens = gr.Textbox(
                        label="Prime Tokens (CSV)",
                        value="60,64,67,3",
                        placeholder="60,64,67,3"
                    )
                    max_new = gr.Slider(16, 256, value=64, step=16, label="Max New Tokens")
                    temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                    top_k = gr.Slider(0, 50, value=0, step=1, label="Top-K")
                    
                    generate_btn = gr.Button("🎵 Generate", variant="primary")
                
                with gr.Column():
                    status = gr.Textbox(label="Status", lines=5, interactive=False)
                    download = gr.File(label="Generated MIDI", interactive=False)
            
            generate_btn.click(
                fn=generate_music,
                inputs=[prime_tokens, max_new, temperature, top_k],
                outputs=[download, status]
            )
    
    return demo

def main():
    """Main function."""
    print("🎵 Starting Bach Transformer Web Interface...")
    demo = create_interface()
    demo.launch(server_name="localhost", server_port=7860, share=False)

if __name__ == "__main__":
    main()
