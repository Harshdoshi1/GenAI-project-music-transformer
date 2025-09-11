import json
import random
import os

print("Creating proper Bach dataset...")

# Create directories
os.makedirs('data/raw', exist_ok=True)

# Create 50 Bach-style chorales with proper length
chorales = []
for i in range(50):
    chorale = []
    length = random.randint(32, 64)  # Good length for training chunks
    
    # Start with C major chord (SATB format)
    chord = [72, 67, 64, 60]  # C5, G4, E4, C4
    
    for t in range(length):
        if t > 0 and random.random() < 0.3:  # 30% chance to move voices
            for voice in range(4):
                if random.random() < 0.5:  # 50% chance each voice moves
                    chord[voice] += random.choice([-2, -1, 1, 2])  # Small steps
        
        # Keep voices in proper ranges
        chord[0] = max(60, min(84, chord[0]))  # Soprano: C4-C6
        chord[1] = max(48, min(72, chord[1]))  # Alto: C3-C5  
        chord[2] = max(36, min(60, chord[2]))  # Tenor: C2-C4
        chord[3] = max(24, min(48, chord[3]))  # Bass: C1-C3
        
        chorale.append(chord.copy())
    
    chorales.append(chorale)

# Save dataset in JSB format
dataset = {'test': chorales}
with open('data/raw/Jsb16thSeparated.json', 'w') as f:
    json.dump(dataset, f)

print(f'✅ Dataset created: {len(chorales)} chorales')
lengths = [len(c) for c in chorales]
print(f'✅ Chorale lengths: {min(lengths)}-{max(lengths)} timesteps')
print(f'✅ Sample: {chorales[0][:3]}...')
print('Ready for training!')
