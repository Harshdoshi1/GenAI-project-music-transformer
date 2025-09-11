#!/usr/bin/env python3

import os
import sys
import subprocess
import json

def install_dependencies():
    """Install required packages."""
    packages = ["torch", "gradio", "numpy", "tqdm", "pretty_midi"]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install {package}")

def copy_dataset():
    """Copy dataset if available."""
    import shutil
    
    src_path = r"c:\Users\Harsh\Downloads\Datasets\Jsb16thSeparated.json"
    dst_path = "data/raw/Jsb16thSeparated.json"
    
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        os.makedirs("data/raw", exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print("✅ Dataset copied to project")
    elif os.path.exists(dst_path):
        print("✅ Dataset already available")
    else:
        print("⚠️ Dataset not found - will create sample data")
        create_sample_data()

def create_sample_data():
    """Create minimal sample data for testing."""
    sample_data = {
        "test": [
            [[60, 64, 67, 72], [62, 65, 69, 74], [64, 67, 71, 76], [60, 64, 67, 72]],
            [[65, 69, 72, 77], [67, 71, 74, 79], [69, 72, 76, 81], [65, 69, 72, 77]],
            [[57, 60, 64, 69], [59, 62, 66, 71], [60, 64, 67, 72], [57, 60, 64, 69]]
        ]
    }
    
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/Jsb16thSeparated.json", "w") as f:
        json.dump(sample_data, f)
    print("✅ Sample data created")

def main():
    print("🎵 Bach Transformer Setup & Launch")
    print("=" * 35)
    
    # Install dependencies
    install_dependencies()
    
    # Copy dataset
    copy_dataset()
    
    # Import after installation
    try:
        import torch
        import gradio as gr
        print("✅ All dependencies loaded")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import project modules
    from src.models import MiniMusicTransformer
    from src.utils import get_device, save_midi, set_seed, PAD, BOS, EOS, SEP, REST
    from src.data import load_jsb, split_train_valid
    
    print("✅ Project modules imported")
    
    # Quick training function
    def quick_train():
        print("🚀 Starting quick training...")
        set_seed(42)
        device = get_device()
        
        # Load data
        try:
            sequences, vocab, ivocab = load_jsb("data/raw/Jsb16thSeparated.json")
            print(f"Loaded {len(sequences)} sequences")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
        
        if not sequences:
            print("❌ No sequences loaded")
            return False
        
        # Use first 20 sequences for quick training
        from torch.utils.data import Dataset, DataLoader
        
        class SimpleDataset(Dataset):
            def __init__(self, sequences, seq_len=64):
                self.chunks = []
                for seq in sequences[:20]:  # Use first 20 for speed
                    for i in range(0, len(seq) - seq_len, seq_len // 2):
                        chunk = seq[i:i + seq_len + 1]
                        if len(chunk) == seq_len + 1:
                            self.chunks.append(torch.tensor(chunk, dtype=torch.long))
            
            def __len__(self):
                return len(self.chunks)
            
            def __getitem__(self, idx):
                chunk = self.chunks[idx]
                return chunk[:-1], chunk[1:]
        
        dataset = SimpleDataset(sequences)
        if len(dataset) == 0:
            print("❌ No training chunks created")
            return False
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model
        model = MiniMusicTransformer(
            vocab_size=vocab['vocab_size'],
            d_model=64,  # Small for speed
            n_heads=4,
            n_layers=2,
            max_len=64
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD)
        
        # Train for 2 epochs
        model.train()
        for epoch in range(2):
            total_loss = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/2, Loss: {total_loss/len(loader):.4f}")
        
        # Save model
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab['vocab_size'],
        }, "checkpoints/model.pt")
        
        with open("outputs/vocab.json", 'w') as f:
            json.dump({'vocab': vocab, 'ivocab': ivocab}, f)
        
        print("✅ Quick training completed!")
        return True
    
    # Generation function
    def generate_music(prime_tokens_str, max_new_tokens, temperature, top_k):
        try:
            device = get_device()
            
            # Load or train model
            if not os.path.exists("checkpoints/model.pt"):
                success = quick_train()
                if not success:
                    return None, "❌ Training failed"
            
            # Load model
            checkpoint = torch.load("checkpoints/model.pt", map_location=device)
            model = MiniMusicTransformer(vocab_size=checkpoint['vocab_size']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with open("outputs/vocab.json", 'r') as f:
                vocab_data = json.load(f)
            ivocab = vocab_data['ivocab']
            
            # Parse prime tokens
            prime_tokens = [BOS]
            if prime_tokens_str.strip():
                for token_str in prime_tokens_str.split(','):
                    try:
                        token = int(token_str.strip())
                        if 0 <= token <= 127:
                            prime_tokens.append(token + 5)
                        elif token in [0, 1, 2, 3, 4]:
                            prime_tokens.append(token)
                    except ValueError:
                        continue
            
            # Generate
            idx = torch.tensor(prime_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                generated = model.generate(
                    idx, 
                    max_new_tokens=int(max_new_tokens), 
                    temperature=float(temperature), 
                    top_k=int(top_k) if top_k > 0 else 0
                )
            
            tokens = generated.squeeze(0).cpu().tolist()
            
            # Save MIDI
            output_path = "outputs/generated.mid"
            save_midi(tokens, ivocab, output_path)
            
            return output_path, f"✅ Generated {len(tokens)} tokens! MIDI saved."
            
        except Exception as e:
            import traceback
            return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    # Create Gradio interface
    with gr.Blocks(title="Bach Transformer") as demo:
        gr.Markdown("# 🎵 Bach Transformer Music Generator")
        gr.Markdown("Generate Bach-style chorales using a transformer model.")
        
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
    
    print("🌐 Starting web interface...")
    demo.launch(server_name="localhost", server_port=7860, share=False)

if __name__ == "__main__":
    main()
