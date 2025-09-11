#!/usr/bin/env python3

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.convert_to_midi import parse_generated_music_file, convert_to_midi, extract_midi_from_note_info

def demo_midi_conversion():
    """Demonstrate the MIDI conversion functionality."""
    
    print("🎵 Bach Transformer - MIDI Conversion Demo")
    print("=" * 45)
    
    # Check if input file exists
    input_file = "generated_music.txt"
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please generate music first using the web interface.")
        return False
    
    try:
        # Parse the text file
        print(f"📖 Parsing {input_file}...")
        time_note_pairs = parse_generated_music_file(input_file)
        print(f"✅ Parsed {len(time_note_pairs)} time/note events")
        
        if not time_note_pairs:
            print("❌ No musical events found in input file")
            return False
        
        # Show sample events
        print("\n🎼 Sample Events:")
        for i, (time, note_info) in enumerate(time_note_pairs[:8]):
            midi_pitch = extract_midi_from_note_info(note_info)
            midi_str = f"MIDI {midi_pitch}" if midi_pitch else "No MIDI"
            print(f"  {time:5.1f}s | {note_info[:30]:<30} | {midi_str}")
        
        if len(time_note_pairs) > 8:
            print(f"  ... and {len(time_note_pairs) - 8} more events")
        
        # Convert to MIDI
        print(f"\n🎹 Converting to MIDI...")
        midi = convert_to_midi(time_note_pairs, note_duration=0.4, velocity=100)
        
        # Count notes
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        print(f"✅ Created MIDI with {total_notes} notes")
        
        if total_notes == 0:
            print("⚠️  Warning: No notes were generated. Check input file format.")
            return False
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)
        
        # Save MIDI file
        output_file = "outputs/generated.mid"
        midi.write(output_file)
        print(f"✅ MIDI saved to: {output_file}")
        
        # Show file info
        file_size = os.path.getsize(output_file)
        print(f"📁 File size: {file_size} bytes")
        
        # Show musical info
        if midi.instruments and midi.instruments[0].notes:
            notes = midi.instruments[0].notes
            first_note = notes[0]
            last_note = notes[-1]
            duration = last_note.end - first_note.start
            pitch_range = (min(n.pitch for n in notes), max(n.pitch for n in notes))
            
            print(f"\n🎼 Musical Information:")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Pitch range: {pitch_range[0]}-{pitch_range[1]} (MIDI)")
            print(f"   Total notes: {len(notes)}")
            print(f"   First note: MIDI {first_note.pitch} at {first_note.start:.1f}s")
            print(f"   Last note: MIDI {last_note.pitch} at {last_note.start:.1f}s")
        
        print(f"\n🎉 MIDI conversion completed successfully!")
        print(f"🎵 You can now play {output_file} in any MIDI player or DAW")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_interface():
    """Test the CLI interface."""
    print(f"\n🖥️  Testing CLI Interface:")
    print(f"Command: python -m src.convert_to_midi --input generated_music.txt --out outputs/generated.mid")
    
    # Import and test CLI
    try:
        from src.convert_to_midi import main
        import sys
        
        # Simulate CLI arguments
        original_argv = sys.argv
        sys.argv = [
            'convert_to_midi',
            '--input', 'generated_music.txt',
            '--out', 'outputs/generated_cli.mid',
            '--duration', '0.5',
            '--velocity', '90'
        ]
        
        print("Running CLI conversion...")
        main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("✅ CLI interface working correctly")
        
    except SystemExit:
        # Expected for successful CLI completion
        sys.argv = original_argv
        print("✅ CLI completed successfully")
    except Exception as e:
        sys.argv = original_argv
        print(f"❌ CLI test failed: {e}")

def main():
    """Run the complete demo."""
    success = demo_midi_conversion()
    
    if success:
        test_cli_interface()
        
        print(f"\n🚀 Usage Examples:")
        print(f"   # Basic conversion:")
        print(f"   python -m src.convert_to_midi --input generated_music.txt --out outputs/my_music.mid")
        print(f"   ")
        print(f"   # With custom settings:")
        print(f"   python -m src.convert_to_midi --input generated_music.txt --out outputs/my_music.mid --duration 0.6 --velocity 120")
        print(f"   ")
        print(f"   # With MP3 export attempt:")
        print(f"   python -m src.convert_to_midi --input generated_music.txt --out outputs/my_music.mid --mp3")
        
        print(f"\n✨ MIDI Converter is ready to use!")
    else:
        print(f"\n💡 To use the MIDI converter:")
        print(f"   1. Generate music using the web interface first")
        print(f"   2. Run this demo again: python demo_midi_conversion.py")

if __name__ == "__main__":
    main()
