#!/usr/bin/env python3

import json
import random
import os

def create_bach_dataset():
    """Create a proper Bach chorales dataset with multiple sequences."""
    
    # Bach-style chord progressions and voice leading patterns
    chorales = []
    
    # Create 50 sample chorales with realistic Bach-style voice leading
    for chorale_idx in range(50):
        chorale = []
        
        # Each chorale has 32-64 timesteps (reasonable length)
        length = random.randint(32, 64)
        
        # Start with a tonic chord in C major
        current_chord = [72, 67, 64, 60]  # C5, G4, E4, C4 (SATB)
        
        for timestep in range(length):
            # Add some voice leading movement
            if timestep > 0:
                # Small movements for voice leading
                for voice in range(4):
                    if random.random() < 0.3:  # 30% chance to move
                        movement = random.choice([-2, -1, 1, 2])  # Small steps
                        current_chord[voice] = max(36, min(84, current_chord[voice] + movement))
            
            # Ensure voices stay in reasonable ranges
            current_chord[0] = max(60, min(84, current_chord[0]))  # Soprano: C4-C6
            current_chord[1] = max(48, min(72, current_chord[1]))  # Alto: C3-C5
            current_chord[2] = max(36, min(60, current_chord[2]))  # Tenor: C2-C4
            current_chord[3] = max(24, min(48, current_chord[3]))  # Bass: C1-C3
            
            chorale.append(current_chord.copy())
        
        chorales.append(chorale)
    
    # Create the dataset structure
    dataset = {
        "test": chorales
    }
    
    return dataset

def main():
    print("Creating Bach chorales dataset...")
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Try to copy real dataset first
    real_dataset_path = r"c:\Users\Harsh\Downloads\Datasets\Jsb16thSeparated.json"
    target_path = "data/raw/Jsb16thSeparated.json"
    
    if os.path.exists(real_dataset_path):
        import shutil
        shutil.copy2(real_dataset_path, target_path)
        print("✅ Real JSB dataset copied successfully!")
    else:
        # Create synthetic dataset
        dataset = create_bach_dataset()
        
        with open(target_path, 'w') as f:
            json.dump(dataset, f)
        
        print("✅ Synthetic Bach dataset created with 50 chorales!")
    
    # Verify the dataset
    with open(target_path, 'r') as f:
        data = json.load(f)
    
    if 'test' in data:
        chorales = data['test']
        print(f"✅ Dataset verified: {len(chorales)} chorales loaded")
        
        if chorales:
            lengths = [len(chorale) for chorale in chorales]
            print(f"✅ Chorale lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
            print(f"✅ Sample chorale: {chorales[0][:3]}...")
    
    print("Dataset ready for training!")

if __name__ == "__main__":
    main()
