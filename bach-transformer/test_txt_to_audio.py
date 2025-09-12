#!/usr/bin/env python3

import os
import sys
import subprocess

def test_txt_to_audio():
    """Test the txt_to_audio converter directly."""
    
    print("🧪 Testing txt_to_audio converter...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    try:
        from txt_to_audio import parse_generated_music_txt, create_midi_from_notes
        print("✅ Successfully imported txt_to_audio module")
        
        # Test parsing
        input_file = "generated_music.txt"
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return False
            
        print(f"📖 Parsing {input_file}...")
        time_pitch_pairs = parse_generated_music_txt(input_file)
        print(f"✅ Extracted {len(time_pitch_pairs)} musical notes")
        
        if not time_pitch_pairs:
            print("❌ No musical notes found")
            return False
            
        # Show extracted notes
        print("\n🎼 Extracted Notes:")
        for i, (time, pitch) in enumerate(time_pitch_pairs):
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = pitch // 12
            note_name = note_names[pitch % 12] + str(octave)
            print(f"   {time:5.1f}s | MIDI {pitch:3d} | {note_name}")
        
        # Create MIDI
        print(f"\n🎹 Creating MIDI...")
        midi = create_midi_from_notes(time_pitch_pairs, note_duration=0.4, velocity=100)
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Save MIDI
        output_file = "outputs/test_generated.mid"
        midi.write(output_file)
        print(f"✅ MIDI saved to: {output_file}")
        
        # Check file
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 MIDI file size: {file_size} bytes")
            
            if file_size > 0:
                print("🎉 Test successful! MIDI file created.")
                return True
            else:
                print("❌ MIDI file is empty")
                return False
        else:
            print("❌ MIDI file was not created")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure pretty_midi is installed: pip install pretty_midi")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_txt_to_audio()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
