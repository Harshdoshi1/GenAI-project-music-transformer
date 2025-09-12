#!/usr/bin/env python3

import os
import sys

def verify_txt_to_audio():
    """Verify the txt_to_audio conversion works by direct execution."""
    
    print("🔍 Bach Transformer - Text-to-Audio Verification")
    print("=" * 55)
    
    # Check if pretty_midi can be imported
    try:
        import pretty_midi
        print("✅ pretty_midi library is available")
    except ImportError:
        print("❌ pretty_midi not available - attempting to install...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pretty_midi'])
            import pretty_midi
            print("✅ pretty_midi installed and imported successfully")
        except Exception as e:
            print(f"❌ Failed to install pretty_midi: {e}")
            return False
    
    # Add src to Python path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        # Import our converter
        from txt_to_audio import parse_generated_music_txt, create_midi_from_notes, extract_midi_pitch
        print("✅ txt_to_audio module imported successfully")
        
        # Test the extract_midi_pitch function
        print("\n🧪 Testing MIDI pitch extraction...")
        test_cases = [
            "F4 (MIDI 65)",
            "C5 (MIDI 72)",
            "A4 (MIDI 69)",
            "G4 (MIDI 67)"
        ]
        
        for test_case in test_cases:
            pitch = extract_midi_pitch(test_case)
            print(f"   '{test_case}' -> MIDI {pitch}")
        
        # Check if generated_music.txt exists
        input_file = "generated_music.txt"
        if not os.path.exists(input_file):
            print(f"\n⚠️  {input_file} not found, creating test file...")
            test_content = """# Test Music File
Time | Note/Event
-----|------------
0.0  | BOS (Start of sequence)
0.2  | C4 (MIDI 60)
0.4  | E4 (MIDI 64)
0.6  | G4 (MIDI 67)
0.8  | EOS (End of sequence)
"""
            with open(input_file, 'w') as f:
                f.write(test_content)
            print(f"✅ Created test file: {input_file}")
        
        # Parse the file
        print(f"\n📖 Parsing {input_file}...")
        time_pitch_pairs = parse_generated_music_txt(input_file)
        print(f"✅ Extracted {len(time_pitch_pairs)} note(s)")
        
        if not time_pitch_pairs:
            print("❌ No notes extracted - check file format")
            return False
        
        # Show extracted notes
        print("\n🎼 Extracted notes:")
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for time, pitch in time_pitch_pairs:
            octave = pitch // 12
            note_name = note_names[pitch % 12] + str(octave)
            print(f"   {time:4.1f}s | MIDI {pitch:3d} | {note_name}")
        
        # Create MIDI
        print(f"\n🎹 Creating MIDI file...")
        midi = create_midi_from_notes(time_pitch_pairs, note_duration=0.5, velocity=100)
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Save MIDI
        output_file = "outputs/verification_test.mid"
        midi.write(output_file)
        print(f"✅ MIDI file saved: {output_file}")
        
        # Verify file was created and has content
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 File size: {file_size} bytes")
            
            if file_size > 0:
                print(f"\n🎉 SUCCESS! Text-to-Audio conversion is working!")
                
                # Show MIDI details
                if midi.instruments and midi.instruments[0].notes:
                    notes = midi.instruments[0].notes
                    print(f"🎼 MIDI contains {len(notes)} note(s)")
                    if notes:
                        duration = max(note.end for note in notes)
                        print(f"🕐 Total duration: {duration:.1f} seconds")
                
                print(f"\n💡 How to use:")
                print(f"   python -m src.txt_to_audio --input generated_music.txt --midi_out outputs/my_music.mid")
                
                print(f"\n🎵 Play the MIDI file with:")
                print(f"   • Windows Media Player")
                print(f"   • VLC Media Player")
                print(f"   • Any music software")
                
                return True
            else:
                print("❌ MIDI file is empty")
                return False
        else:
            print("❌ MIDI file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_txt_to_audio()
    
    if success:
        print(f"\n✅ Verification completed successfully!")
        print(f"🎵 The txt_to_audio converter is ready to use.")
    else:
        print(f"\n❌ Verification failed!")
        print(f"🔧 Check dependencies and file formats.")
        
    input("\nPress Enter to continue...")
