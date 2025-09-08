# 🎵 Bach Transformer - Quick Start Guide

## Step 1: Setup
```powershell
# Navigate to project directory
cd "c:\Users\Harsh\Music\GenAI Project\bach-transformer"

# Install dependencies (automatic)
python setup.py
```

## Step 2: Add Dataset
Place your JSB dataset file in `data/raw/` with one of these names:
- `jsb-chorales-16th.json`
- `Jsb16thSeparated.json`

## Step 3: Run Web Interface
```powershell
python app.py
```

Then open: http://localhost:7860

## Alternative: Command Line

### Quick Training
```powershell
python quick_train.py
```

### Generate Music
```powershell
python -m src.generate --prime_tokens "60,64,67,3" --out_midi outputs/demo.mid
```

### Run Tests
```powershell
python tests/test_smoke.py
```

## Batch Script (Windows)
```powershell
.\run.bat
```

## Troubleshooting

1. **Module not found**: Make sure you're in the `bach-transformer` directory
2. **No data file**: Place dataset in `data/raw/` directory
3. **Dependencies missing**: Run `python setup.py` first

## What This Does

- Trains a small transformer model on Bach chorales
- Generates new music in SATB (4-voice) format
- Outputs MIDI files you can play
- Web interface for easy interaction
