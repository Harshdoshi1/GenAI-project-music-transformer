#!/usr/bin/env python3

import os
import sys
import json
import subprocess

# Check and install dependencies
def check_dependencies():
    try:
        import torch
        import gradio as gr
        return True
    except ImportError:
        print("Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "gradio", "pretty_midi", "tqdm", "numpy"])
            import torch
            import gradio as gr
            return True
        except:
            return False

if not check_dependencies():
    print("❌ Failed to install dependencies")
    sys.exit(1)

import torch
import gradio as gr

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import MiniMusicTransformer
from src.utils import get_device, save_midi


def load_model_and_vocab():
    """Load trained model and vocabulary."""
    checkpoint_path = "checkpoints/model.pt"
    vocab_path = "outputs/vocab.json"
    
    if not os.path.exists(checkpoint_path):
        return None, None, "No trained model found. Please run training first."
    
    if not os.path.exists(vocab_path):
        return None, None, "No vocabulary file found. Please run training first."
    
    try:
        device = get_device()
        
        # Load vocab
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        vocab = vocab_data['vocab']
        ivocab = vocab_data['ivocab']
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = MiniMusicTransformer(vocab_size=checkpoint['vocab_size']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, ivocab, "Model loaded successfully!"
        
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def quick_train_if_needed():
    """Run quick training if no model exists."""
    if not os.path.exists("checkpoints/model.pt"):
        try:
            print("No model found, running quick training...")
            from quick_train import quick_train
            return quick_train()
        except Exception as e:
            print(f"Quick training failed: {e}")
            return False
    return True

def generate_music(prime_tokens_str, max_new_tokens, temperature, top_k):
    """Generate music from prime tokens."""
    try:
        # Auto-train if needed
        if not quick_train_if_needed():
            return None, "❌ Training failed. Please check data file and try again."
        
        model, ivocab, status = load_model_and_vocab()
        if model is None:
            return None, status
        
        device = get_device()
        
        # Parse prime tokens
        if prime_tokens_str.strip():
            prime_list = [int(x.strip()) for x in prime_tokens_str.split(',') if x.strip()]
        else:
            prime_list = []
        
        # Convert MIDI notes to pitch tokens (add 5 offset)
        prime_tokens = []
        for token in prime_list:
            if 0 <= token <= 127:  # MIDI pitch
                prime_tokens.append(token + 5)  # Convert to pitch token
            elif token in [0, 1, 2, 3, 4]:  # Special tokens
                prime_tokens.append(token)
        
        # Add BOS if not present
        if not prime_tokens or prime_tokens[0] != 1:  # BOS = 1
            prime_tokens = [1] + prime_tokens
        
        # Generate
        idx = torch.tensor(prime_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                idx, 
                max_new_tokens=int(max_new_tokens), 
                temperature=float(temperature), 
                top_k=int(top_k) if top_k > 0 else 0
            )
        
        # Convert to list
        tokens = generated.squeeze(0).cpu().tolist()
        
        # Save MIDI
        output_path = "outputs/ui.mid"
        os.makedirs("outputs", exist_ok=True)
        save_midi(tokens, ivocab, output_path)
        
        return output_path, f"✅ Generated {len(tokens)} tokens successfully! MIDI saved to {output_path}"
        
    except Exception as e:
        import traceback
        error_msg = f"Generation failed: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Bach Transformer Music Generator") as demo:
        gr.Markdown("# 🎵 Bach Transformer Music Generator")
        gr.Markdown("Generate Bach-style chorales using a trained transformer model.")
        
        # Check data file status
        data_files = ["data/raw/jsb-chorales-16th.json", "data/raw/Jsb16thSeparated.json"]
        data_exists = any(os.path.exists(f) for f in data_files)
        
        if not data_exists:
            gr.Markdown("## ⚠️ Setup Required")
            gr.Markdown("Please place your dataset file in `data/raw/` with one of these names:")
            for f in data_files:
                gr.Markdown(f"- `{f}`")
            gr.Markdown("Then refresh this page.")
        else:
            gr.Markdown("## 🎼 Generate Music")
            
            with gr.Row():
                with gr.Column():
                    prime_tokens = gr.Textbox(
                        label="Prime Tokens (CSV)", 
                        value="60,64,67,3",
                        placeholder="Enter MIDI notes separated by commas (e.g., 60,64,67,3)",
                        info="MIDI notes (0-127) or special tokens (0=PAD, 1=BOS, 2=EOS, 3=SEP, 4=REST)"
                    )
                    
                    max_new = gr.Slider(
                        minimum=16, 
                        maximum=512, 
                        value=128, 
                        step=16,
                        label="Max New Tokens",
                        info="Number of new tokens to generate"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1, 
                        maximum=2.0, 
                        value=1.0, 
                        step=0.1,
                        label="Temperature",
                        info="Higher = more random, Lower = more predictable"
                    )
                    
                    top_k = gr.Slider(
                        minimum=0, 
                        maximum=50, 
                        value=0, 
                        step=1,
                        label="Top-K Sampling",
                        info="0 = disabled, higher = more constrained sampling"
                    )
                    
                    generate_btn = gr.Button("🎵 Generate Music", variant="primary", interactive=data_exists)
                
                with gr.Column():
                    status_text = gr.Textbox(
                        label="Status", 
                        interactive=False,
                        lines=5
                    )
                    
                    download_file = gr.File(
                        label="Generated MIDI",
                        interactive=False
                    )
            
            # Event handlers
            if data_exists:
                generate_btn.click(
                    fn=generate_music,
                    inputs=[prime_tokens, max_new, temperature, top_k],
                    outputs=[download_file, status_text]
                )
        
        # Initial status check
        def check_initial_status():
            if not data_exists:
                return "⚠️ Please add dataset file to data/raw/ directory"
            elif not os.path.exists("checkpoints/model.pt"):
                return "🔄 No model found. Will train automatically on first generation."
            else:
                return "✅ Model ready! You can generate music."
        
        demo.load(
            fn=check_initial_status,
            outputs=[status_text]
        )
    
    return demo


def main():
    """Start the Gradio app."""
    demo = create_interface()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
