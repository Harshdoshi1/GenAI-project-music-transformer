#!/usr/bin/env python3

import os
import sys
import subprocess
import json

def install_dependencies():
    """Install required dependencies if not available."""
    dependencies = ['pretty_midi', 'pydub']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"📦 Installing {dep}...")
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"✅ {dep} installed successfully")
                else:
                    print(f"⚠️  {dep} installation had issues: {result.stderr}")
            except Exception as e:
                print(f"⚠️  Could not install {dep}: {e}")

def create_sample_music_files():
    """Create sample music text files for testing."""
    
    # Sample 1: Simple C major chord progression
    sample1 = """# Bach Transformer - Sample Chorale 1
# C Major Chord Progression
========================================

Time | Note/Event
-----|------------
0.0  | BOS (Start of sequence)
0.0  | C4 (MIDI 60)
0.0  | E4 (MIDI 64)
0.0  | G4 (MIDI 67)
0.0  | C5 (MIDI 72)
0.5  | SEP (Voice separator)
0.5  | F4 (MIDI 65)
0.5  | A4 (MIDI 69)
0.5  | C5 (MIDI 72)
0.5  | F5 (MIDI 77)
1.0  | SEP (Voice separator)
1.0  | G4 (MIDI 67)
1.0  | B4 (MIDI 71)
1.0  | D5 (MIDI 74)
1.0  | G5 (MIDI 79)
1.5  | SEP (Voice separator)
1.5  | C4 (MIDI 60)
1.5  | E4 (MIDI 64)
1.5  | G4 (MIDI 67)
1.5  | C5 (MIDI 72)
2.0  | EOS (End of sequence)

Generated using Bach Transformer model
Sequence length: 16 notes
Format: SATB chorale style
"""

    # Sample 2: Bach-style melody
    sample2 = """# Bach Transformer - Sample Chorale 2  
# Bach-style Melody
========================================

Time | Note/Event
-----|------------
0.0  | BOS (Start of sequence)
0.0  | G4 (MIDI 67)
0.25 | A4 (MIDI 69)
0.5  | B4 (MIDI 71)
0.75 | C5 (MIDI 72)
1.0  | SEP (Voice separator)
1.0  | E4 (MIDI 64)
1.25 | F4 (MIDI 65)
1.5  | G4 (MIDI 67)
1.75 | A4 (MIDI 69)
2.0  | SEP (Voice separator)
2.0  | C4 (MIDI 60)
2.25 | D4 (MIDI 62)
2.5  | E4 (MIDI 64)
2.75 | F4 (MIDI 65)
3.0  | SEP (Voice separator)
3.0  | G3 (MIDI 55)
3.25 | A3 (MIDI 57)
3.5  | B3 (MIDI 59)
3.75 | C4 (MIDI 60)
4.0  | EOS (End of sequence)

Generated using Bach Transformer model
Sequence length: 16 notes
Format: SATB chorale style
"""

    samples = {
        'sample1_cmajor.txt': sample1,
        'sample2_melody.txt': sample2
    }
    
    for filename, content in samples.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📝 Created sample file: {filename}")
    
    return list(samples.keys())

def test_conversion(input_file):
    """Test the conversion process."""
    
    print(f"\n🎵 Testing conversion for: {input_file}")
    print("-" * 50)
    
    try:
        # Add src to path
        src_path = os.path.join(os.getcwd(), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import converter functions
        from txt_to_audio import parse_generated_music_txt, create_midi_from_notes
        
        # Parse the file
        print(f"📖 Parsing {input_file}...")
        time_pitch_pairs = parse_generated_music_txt(input_file)
        print(f"✅ Extracted {len(time_pitch_pairs)} musical notes")
        
        if not time_pitch_pairs:
            print("❌ No musical notes found")
            return False
        
        # Show first few notes
        print(f"\n🎼 First few notes:")
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i, (time, pitch) in enumerate(time_pitch_pairs[:8]):
            octave = pitch // 12
            note_name = note_names[pitch % 12] + str(octave)
            print(f"   {time:5.1f}s | MIDI {pitch:3d} | {note_name}")
        
        if len(time_pitch_pairs) > 8:
            print(f"   ... and {len(time_pitch_pairs) - 8} more notes")
        
        # Create MIDI
        print(f"\n🎹 Creating MIDI...")
        midi = create_midi_from_notes(time_pitch_pairs, note_duration=0.4, velocity=100)
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"outputs/{base_name}.mid"
        
        # Save MIDI
        midi.write(output_file)
        print(f"✅ MIDI saved to: {output_file}")
        
        # Verify and show info
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 File size: {file_size} bytes")
            
            if midi.instruments and midi.instruments[0].notes:
                notes = midi.instruments[0].notes
                duration = max(note.end for note in notes) - min(note.start for note in notes)
                pitch_range = (min(n.pitch for n in notes), max(n.pitch for n in notes))
                
                print(f"🎼 Musical info:")
                print(f"   • Duration: {duration:.1f} seconds")
                print(f"   • Note count: {len(notes)}")
                print(f"   • Pitch range: MIDI {pitch_range[0]}-{pitch_range[1]}")
            
            return True
        else:
            print("❌ MIDI file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    
    print("🎵 Bach Transformer - Complete Text-to-Audio Demo")
    print("=" * 60)
    
    # Install dependencies
    print("\n📦 Checking dependencies...")
    install_dependencies()
    
    # Create sample files
    print("\n📝 Creating sample music files...")
    sample_files = create_sample_music_files()
    
    # Test existing generated_music.txt if it exists
    if os.path.exists('generated_music.txt'):
        sample_files.insert(0, 'generated_music.txt')
    
    # Test conversions
    print(f"\n🧪 Testing conversions...")
    results = {}
    
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            success = test_conversion(sample_file)
            results[sample_file] = success
        else:
            print(f"⚠️  File not found: {sample_file}")
            results[sample_file] = False
    
    # Summary
    print(f"\n📊 Conversion Results Summary:")
    print("=" * 40)
    
    successful = 0
    for filename, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} | {filename}")
        if success:
            successful += 1
    
    print(f"\n🎯 Overall: {successful}/{len(results)} conversions successful")
    
    if successful > 0:
        print(f"\n🎉 Text-to-Audio conversion is working!")
        print(f"📁 Check the 'outputs/' directory for MIDI files")
        print(f"\n💡 Usage:")
        print(f"   python -m src.txt_to_audio --input your_music.txt --midi_out output.mid")
        print(f"\n🎵 Play MIDI files with:")
        print(f"   • Windows Media Player")
        print(f"   • VLC Media Player")
        print(f"   • Any DAW software")
        print(f"   • Online MIDI players")
        
        # Show available output files
        if os.path.exists('outputs'):
            midi_files = [f for f in os.listdir('outputs') if f.endswith('.mid')]
            if midi_files:
                print(f"\n📁 Generated MIDI files:")
                for midi_file in midi_files:
                    full_path = os.path.join('outputs', midi_file)
                    size = os.path.getsize(full_path)
                    print(f"   • {midi_file} ({size} bytes)")
    else:
        print(f"\n❌ All conversions failed!")
        print(f"🔧 Try installing dependencies manually:")
        print(f"   pip install pretty_midi pydub")
    
    # Clean up sample files
    print(f"\n🧹 Cleaning up sample files...")
    for sample_file in sample_files:
        if sample_file != 'generated_music.txt' and os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"   Removed: {sample_file}")

if __name__ == "__main__":
    main()
