# bach-transformer

Tiny Music Transformer (relative bias) trained on JSB Chorales 16th grid.

## Goal

Tiny Music Transformer (relative bias) trained on JSB Chorales 16th grid.

## Quickstart

1. Create and activate a virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset file in `data/raw/`:
   - Either `data/raw/jsb-chorales-16th.json`
   - Or `data/raw/Jsb16thSeparated.json`

4. Train the model:
   ```bash
   python -m src.train --data_path data/raw/jsb-chorales-16th.json --epochs 5 --batch_size 32
   ```

5. Generate a sample:
   ```bash
   python -m src.generate --checkpoint checkpoints/best.pt --out_midi outputs/sample.mid --max_len 256 --temperature 1.0 --top_k 0
   ```

## Runbook

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
Put dataset file at: `data/raw/jsb-chorales-16th.json` (or `Jsb16thSeparated.json`)

### Training
```bash
# Train with default settings
python -m src.train --epochs 3 --seq_len 256

# Train with custom parameters
python -m src.train --data data/raw/Jsb16thSeparated.json --epochs 5 --batch_size 16 --seq_len 512
```

### Generation
```bash
# Generate with prime tokens
python -m src.generate --prime_tokens "60,64,67,3" --out_midi outputs/demo.mid

# Generate with different settings
python -m src.generate --checkpoint checkpoints/model.pt --vocab outputs/vocab.json --max_new 512 --temperature 0.8
```

### One-Shot Command
```bash
# Windows
run.bat

# Linux/Mac
./run.sh
```

### Web UI
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

### Testing
```bash
python tests/test_smoke.py
```

## Notes

- Expects the dataset at `data/raw/jsb-chorales-16th.json` or `data/raw/Jsb16thSeparated.json`.
- The loader converts SATB chorales into sequences with tokens: PAD=0, BOS=1, EOS=2, SEP=3, REST=4, pitches=5-132.
- Each timestep is encoded as S,A,T,B,SEP pattern.
- Outputs are saved to `outputs/`. Checkpoints are saved under `checkpoints/`.
