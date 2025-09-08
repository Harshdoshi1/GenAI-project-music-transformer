#!/usr/bin/env python3

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_data_file():
    """Check if data file exists."""
    data_files = [
        "data/raw/jsb-chorales-16th.json",
        "data/raw/Jsb16thSeparated.json"
    ]
    
    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"✅ Found data file: {filepath}")
            return True
    
    print("❌ No data file found!")
    print("Please place one of these files in data/raw/:")
    for f in data_files:
        print(f"  - {f}")
    return False

def main():
    print("🎵 Bach Transformer Setup")
    print("========================")
    
    # Check if we're in the right directory
    if not os.path.exists("src/train.py"):
        print("❌ Please run this from the bach-transformer directory")
        return False
    
    # Install dependencies
    if not install_requirements():
        return False
    
    # Check data file
    if not check_data_file():
        return False
    
    print("\n🚀 Setup complete! You can now run:")
    print("   python app.py          # Web interface")
    print("   python quick_train.py  # Quick training")
    print("   .\\run.bat             # Full pipeline (Windows)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
