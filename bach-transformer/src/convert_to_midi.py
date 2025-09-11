#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

import pretty_midi


def parse_generated_music_file(file_path: str) -> List[Tuple[float, str]]:
    """
    Parse the generated_music.txt file and extract (time, note) pairs.
    
    Args:
        file_path: Path to the generated music text file
        
    Returns:
        List of (time, note_info) tuples
    """
    time_note_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the start of the time/note table
    table_started = False
    for line in lines:
        line = line.strip()
        
        # Skip comments and headers
        if line.startswith('#') or line.startswith('=') or not line:
            continue
            
        # Skip header lines
        if 'Time' in line and 'Note' in line:
            table_started = True
            continue
            
        if line.startswith('---'):
            continue
            
        # Skip legend and footer
        if line.startswith('Total tokens') or line.startswith('Duration') or line.startswith('Format'):
            break
            
        if line.startswith('Legend'):
            break
            
        if not table_started:
            continue
            
        # Parse table rows: "  0.0 |     1 | BOS (Start)"
        # Pattern: time | token | note_info
        match = re.match(r'\s*(\d+\.?\d*)\s*\|\s*\d+\s*\|\s*(.+)', line)
        if match:
            time_str = match.group(1)
            note_info = match.group(2).strip()
            
            try:
                time = float(time_str)
                time_note_pairs.append((time, note_info))
            except ValueError:
                continue
    
    return time_note_pairs


def extract_midi_from_note_info(note_info: str) -> Optional[int]:
    """
    Extract MIDI pitch number from note info string.
    
    Args:
        note_info: String like "C4 (MIDI 60)" or "F4 (MIDI 65)"
        
    Returns:
        MIDI pitch number or None if not found
    """
    # Look for pattern like "F4 (MIDI 65)"
    midi_match = re.search(r'MIDI (\d+)', note_info)
    if midi_match:
        return int(midi_match.group(1))
    
    # Alternative: try to parse note name directly
    note_match = re.match(r'([A-G]#?)(\d+)', note_info)
    if note_match:
        note_name = note_match.group(1)
        octave = int(note_match.group(2))
        
        # Convert note name to MIDI
        note_to_number = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        if note_name in note_to_number:
            midi_pitch = octave * 12 + note_to_number[note_name]
            return midi_pitch
    
    return None


def convert_to_midi(time_note_pairs: List[Tuple[float, str]], 
                   note_duration: float = 0.4,
                   velocity: int = 100) -> pretty_midi.PrettyMIDI:
    """
    Convert time/note pairs to MIDI using pretty_midi.
    
    Args:
        time_note_pairs: List of (time, note_info) tuples
        note_duration: Duration for each note in seconds
        velocity: MIDI velocity (0-127)
        
    Returns:
        PrettyMIDI object
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an instrument (Acoustic Grand Piano)
    piano = pretty_midi.Instrument(program=0)
    
    for time, note_info in time_note_pairs:
        # Check for special events
        if 'EOS' in note_info:
            break
            
        if 'SEP' in note_info or 'BOS' in note_info:
            continue
            
        if 'REST' in note_info:
            continue
            
        # Extract MIDI pitch
        midi_pitch = extract_midi_from_note_info(note_info)
        if midi_pitch is not None:
            # Ensure pitch is in valid MIDI range
            midi_pitch = max(0, min(127, midi_pitch))
            
            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=time,
                end=time + note_duration
            )
            piano.notes.append(note)
    
    # Add the instrument to the MIDI object
    midi.instruments.append(piano)
    
    return midi


def export_to_mp3(midi_file_path: str, mp3_file_path: str) -> bool:
    """
    Convert MIDI file to MP3 using pydub.
    
    Args:
        midi_file_path: Path to input MIDI file
        mp3_file_path: Path to output MP3 file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from pydub import AudioSegment
        from pydub.utils import which
        
        # Check if ffmpeg is available
        if not which("ffmpeg"):
            print("❌ ffmpeg not found. Install ffmpeg to enable MP3 export.")
            print("   Download from: https://ffmpeg.org/download.html")
            return False
        
        # Note: Direct MIDI to MP3 conversion is complex
        # This is a placeholder - in practice, you'd need a MIDI synthesizer
        print("⚠️  Direct MIDI to MP3 conversion requires additional setup.")
        print("   Consider using external tools like FluidSynth or TiMidity++")
        print("   Or use online MIDI to MP3 converters.")
        
        return False
        
    except ImportError:
        print("❌ pydub not installed. Install with: pip install pydub")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Convert Bach Transformer generated music text to MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.convert_to_midi --input generated_music.txt --out outputs/generated.mid
  python -m src.convert_to_midi --input generated_music.txt --out outputs/generated.mid --mp3
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to generated music text file'
    )
    
    parser.add_argument(
        '--out', '-o',
        required=True,
        help='Output MIDI file path'
    )
    
    parser.add_argument(
        '--mp3',
        action='store_true',
        help='Also export as MP3 (requires ffmpeg)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=0.4,
        help='Note duration in seconds (default: 0.4)'
    )
    
    parser.add_argument(
        '--velocity', '-v',
        type=int,
        default=100,
        help='MIDI velocity 0-127 (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    try:
        print(f"🎵 Converting {args.input} to MIDI...")
        
        # Parse the text file
        time_note_pairs = parse_generated_music_file(args.input)
        print(f"✅ Parsed {len(time_note_pairs)} time/note events")
        
        if not time_note_pairs:
            print("❌ No musical events found in input file")
            sys.exit(1)
        
        # Convert to MIDI
        midi = convert_to_midi(time_note_pairs, args.duration, args.velocity)
        
        # Count actual notes
        note_count = sum(len(instrument.notes) for instrument in midi.instruments)
        print(f"✅ Created MIDI with {note_count} notes")
        
        if note_count == 0:
            print("⚠️  Warning: No notes were generated. Check input file format.")
        
        # Save MIDI file
        midi.write(args.out)
        print(f"✅ MIDI saved to: {args.out}")
        
        # Optional MP3 export
        if args.mp3:
            mp3_path = args.out.replace('.mid', '.mp3')
            success = export_to_mp3(args.out, mp3_path)
            if success:
                print(f"✅ MP3 saved to: {mp3_path}")
        
        # Show file info
        file_size = os.path.getsize(args.out)
        print(f"📁 File size: {file_size} bytes")
        
        # Show summary
        if midi.instruments and midi.instruments[0].notes:
            first_note = midi.instruments[0].notes[0]
            last_note = midi.instruments[0].notes[-1]
            duration = last_note.end - first_note.start
            print(f"🎼 Duration: {duration:.1f} seconds")
            print(f"🎹 Pitch range: {min(n.pitch for n in midi.instruments[0].notes)}-{max(n.pitch for n in midi.instruments[0].notes)}")
        
        print("🎉 Conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
