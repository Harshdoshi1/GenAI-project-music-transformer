@echo off
REM Bach Transformer - One-shot training and generation script for Windows

echo 🎵 Bach Transformer - Training and Generation
echo =============================================

REM Check if we're in the right directory
if not exist "src\train.py" (
    echo ❌ Error: Please run this script from the bach-transformer directory
    exit /b 1
)

REM Check for data file
if not exist "data\raw\jsb-chorales-16th.json" if not exist "data\raw\Jsb16thSeparated.json" (
    echo ❌ Error: No dataset found in data/raw/
    echo    Please place jsb-chorales-16th.json or Jsb16thSeparated.json in data/raw/
    exit /b 1
)

echo 📚 Starting training...
python -m src.train --epochs 3 --seq_len 256

if errorlevel 1 (
    echo ❌ Training failed!
    exit /b 1
)

echo ✅ Training completed!
echo.
echo 🎼 Generating demo music...
python -m src.generate --prime_tokens "60,64,67,3" --out_midi outputs/demo.mid

if errorlevel 1 (
    echo ❌ Generation failed!
    exit /b 1
)

echo ✅ Generation completed!
echo 🎵 Demo MIDI saved to: outputs/demo.mid
echo.
echo 🚀 You can now:
echo    - Listen to outputs/demo.mid
echo    - Run 'python app.py' for the web UI
echo    - Generate more music with different prime tokens
