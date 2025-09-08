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

## Notes

- Expects the dataset at `data/raw/jsb-chorales-16th.json` or `data/raw/Jsb16thSeparated.json`.
- The loader accepts simple integer-encoded notes (0-127) per sequence. If your JSON is structured as JSB chorales with multiple voices, it will flatten to a single stream by interleaving voices at each time step.
- Outputs are saved to `outputs/`. Checkpoints are saved under `checkpoints/`.
