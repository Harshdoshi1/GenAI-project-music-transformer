#!/bin/bash

# Bach Transformer - One-shot training and generation script

echo "🎵 Bach Transformer - Training and Generation"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "src/train.py" ]; then
    echo "❌ Error: Please run this script from the bach-transformer directory"
    exit 1
fi

# Check for data file
if [ ! -f "data/raw/jsb-chorales-16th.json" ] && [ ! -f "data/raw/Jsb16thSeparated.json" ]; then
    echo "❌ Error: No dataset found in data/raw/"
    echo "   Please place jsb-chorales-16th.json or Jsb16thSeparated.json in data/raw/"
    exit 1
fi

echo "📚 Starting training..."
python -m src.train --epochs 3 --seq_len 256

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ Training completed!"
echo ""
echo "🎼 Generating demo music..."
python -m src.generate --prime_tokens "60,64,67,3" --out_midi outputs/demo.mid

if [ $? -ne 0 ]; then
    echo "❌ Generation failed!"
    exit 1
fi

echo "✅ Generation completed!"
echo "🎵 Demo MIDI saved to: outputs/demo.mid"
echo ""
echo "🚀 You can now:"
echo "   - Listen to outputs/demo.mid"
echo "   - Run 'python app.py' for the web UI"
echo "   - Generate more music with different prime tokens"
