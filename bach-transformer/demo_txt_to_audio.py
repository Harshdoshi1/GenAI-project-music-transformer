#!/usr/bin/env python3

import os
import sys

def demo_txt_to_audio():
    """Demo the txt_to_audio converter with a simple test."""
    
    print("🎵 Bach Transformer - Text to Audio Demo")
    print("=" * 50)
    
    # Create a simple test file
    test_content = """# Generated Bach-style Chorale
# Test Output for Audio Conversion
========================================

Time | Note/Event
-----|------------
0.0  | BOS (Start of sequence)
0.2  | C4 (MIDI 60)
0.4  | E4 (MIDI 64) 
0.6  | G4 (MIDI 67)
0.8  | SEP (Voice separator)
1.0  | F4 (MIDI 65)
1.2  | A4 (MIDI 69)
1.4  | C5 (MIDI 72)
1.6  | SEP (Voice separator)
2.0  | G4 (MIDI 67)
2.2  | B4 (MIDI 71)
2.4  | D5 (MIDI 74)
2.6  | EOS (End of sequence)

Generated using Bach Transformer model
"""
    
    # Write test file
    test_file = "test_music.txt"
    with open(test_file, 'w') as f:
        f.write(test_content)
    print(f"📝 Created test file: {test_file}")
    
    # Try to import and use the converter
    try:
        # Add src to Python path
        src_path = os.path.join(os.getcwd(), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import the converter
        try:
            import pretty_midi
            print("✅ pretty_midi is available")
        except ImportError:
            print("❌ pretty_midi not installed")
            print("   Installing pretty_midi...")
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'pretty_midi'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ pretty_midi installed successfully")
                import pretty_midi
            else:
                print(f"❌ Failed to install pretty_midi: {result.stderr}")
                return False
        
        from txt_to_audio import parse_generated_music_txt, create_midi_from_notes
        print("✅ txt_to_audio module imported successfully")
        
        # Parse the test file
        print(f"\n📖 Parsing {test_file}...")
        time_pitch_pairs = parse_generated_music_txt(test_file)
        print(f"✅ Extracted {len(time_pitch_pairs)} musical notes")
        
        if not time_pitch_pairs:
            print("❌ No musical notes found in test file")
            return False
        
        # Show extracted notes
        print(f"\n🎼 Extracted Notes:")
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i, (time, pitch) in enumerate(time_pitch_pairs):
            octave = pitch // 12
            note_name = note_names[pitch % 12] + str(octave)
            print(f"   {time:5.1f}s | MIDI {pitch:3d} | {note_name}")
        
        # Create MIDI
        print(f"\n🎹 Creating MIDI file...")
        midi = create_midi_from_notes(time_pitch_pairs, note_duration=0.5, velocity=100)
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Save MIDI
        output_file = "outputs/demo_generated.mid"
        midi.write(output_file)
        print(f"✅ MIDI saved to: {output_file}")
        
        # Verify file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 MIDI file size: {file_size} bytes")
            
            if file_size > 0:
                print(f"\n🎉 Success! MIDI file created successfully!")
                print(f"🎵 You can play {output_file} in:")
                print(f"   • Windows Media Player")
                print(f"   • VLC Media Player") 
                print(f"   • Any DAW (FL Studio, Ableton, etc.)")
                print(f"   • Online MIDI players")
                
                # Show MIDI info
                if midi.instruments and midi.instruments[0].notes:
                    notes = midi.instruments[0].notes
                    duration = max(note.end for note in notes) - min(note.start for note in notes)
                    pitch_range = (min(n.pitch for n in notes), max(n.pitch for n in notes))
                    print(f"\n🎼 Musical Info:")
                    print(f"   • Duration: {duration:.1f} seconds")
                    print(f"   • Note count: {len(notes)}")
                    print(f"   • Pitch range: MIDI {pitch_range[0]}-{pitch_range[1]}")
                
                return True
            else:
                print("❌ MIDI file is empty")
                return False
        else:
            print("❌ MIDI file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"🧹 Cleaned up test file: {test_file}")

if __name__ == "__main__":
    print("🚀 Starting Bach Transformer Text-to-Audio Demo...")
    success = demo_txt_to_audio()
    
    if success:
        print(f"\n✅ Demo completed successfully!")
        print(f"🎵 The txt_to_audio converter is working correctly.")
        print(f"\n💡 Usage:")
        print(f"   python -m src.txt_to_audio --input your_music.txt --midi_out output.mid")
    else:
        print(f"\n❌ Demo failed!")
        print(f"🔧 Try installing dependencies: pip install pretty_midi")
