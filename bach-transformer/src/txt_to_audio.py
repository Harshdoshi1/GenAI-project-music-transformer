#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

import pretty_midi


def parse_generated_music_txt(file_path: str) -> List[Tuple[float, int]]:
    """
    Parse the generated_music.txt file and extract (time, midi_pitch) pairs.
    
    Args:
        file_path: Path to the generated music text file
        
    Returns:
        List of (time, midi_pitch) tuples
    """
    time_pitch_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip comments and headers
        if line.startswith('#') or line.startswith('=') or not line:
            continue
            
        # Skip header lines
        if 'Time' in line and ('Note' in line or 'Token' in line):
            continue
            
        if line.startswith('---') or line.startswith('|'):
            continue
            
        # Skip footer lines
        if any(keyword in line for keyword in ['Generated using', 'Sequence length', 'Format:', 'Legend:', 'Total tokens']):
            break
            
        # Parse table rows with 3 columns: time | token | note/event
        # Example: "0.2  | F4 (MIDI 65)" or "0.2  |    70 | F4 (MIDI 65)"
        parts = line.split('|')
        if len(parts) >= 2:
            try:
                time_str = parts[0].strip()
                note_event = parts[-1].strip()  # Use last column for note/event
                
                time = float(time_str)
                
                # Check for special events
                if any(keyword in note_event for keyword in ['BOS', 'SEP', 'REST']):
                    continue
                    
                if 'EOS' in note_event:
                    break
                    
                # Extract MIDI pitch from note event
                midi_pitch = extract_midi_pitch(note_event)
                if midi_pitch is not None:
                    time_pitch_pairs.append((time, midi_pitch))
                    
            except (ValueError, IndexError):
                continue
    
    return time_pitch_pairs


def extract_midi_pitch(note_event: str) -> Optional[int]:
    """
    Extract MIDI pitch number from note event string.
    
    Args:
        note_event: String like "F4 (MIDI 65)" or "C5 (MIDI 60)"
        
    Returns:
        MIDI pitch number or None if not found
    """
    # Look for pattern like "MIDI 65" in parentheses
    midi_match = re.search(r'MIDI (\d+)', note_event)
    if midi_match:
        pitch = int(midi_match.group(1))
        # Ensure valid MIDI range
        if 0 <= pitch <= 127:
            return pitch
    
    # Alternative: try to parse note name directly (e.g., "C4", "F#5")
    note_match = re.match(r'([A-G]#?)(\d+)', note_event.split()[0] if note_event.split() else '')
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
            if 0 <= midi_pitch <= 127:
                return midi_pitch
    
    return None


def create_midi_from_notes(time_pitch_pairs: List[Tuple[float, int]], 
                          note_duration: float = 0.4,
                          velocity: int = 100) -> pretty_midi.PrettyMIDI:
    """
    Create MIDI file from time/pitch pairs.
    
    Args:
        time_pitch_pairs: List of (time, midi_pitch) tuples
        note_duration: Duration for each note in seconds
        velocity: MIDI velocity (0-127)
        
    Returns:
        PrettyMIDI object
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create piano instrument (program 0 = Acoustic Grand Piano)
    piano = pretty_midi.Instrument(program=0)
    
    for time, midi_pitch in time_pitch_pairs:
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


def convert_midi_to_mp3(midi_file_path: str, mp3_file_path: str) -> bool:
    """
    Convert MIDI file to MP3 using pydub and fluidsynth.
    
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
            print("❌ Install ffmpeg to enable MP3 export")
            print("   Windows: Download from https://ffmpeg.org/download.html and add to PATH")
            print("   Ubuntu: sudo apt install ffmpeg")
            print("   macOS: brew install ffmpeg")
            return False
        
        # Try to use fluidsynth for MIDI synthesis
        try:
            # This requires fluidsynth and a soundfont
            # For now, we'll provide instructions for manual conversion
            print("⚠️  Direct MIDI to MP3 conversion requires additional setup:")
            print("   1. Install FluidSynth: https://www.fluidsynth.org/")
            print("   2. Download a soundfont (e.g., GeneralUser GS)")
            print("   3. Use: fluidsynth -ni soundfont.sf2 input.mid -F output.wav")
            print("   4. Convert WAV to MP3 with ffmpeg")
            print(f"   Alternative: Use online MIDI to MP3 converters with {midi_file_path}")
            return False
            
        except Exception as e:
            print(f"❌ MIDI synthesis failed: {e}")
            return False
        
    except ImportError:
        print("❌ pydub not installed. Install with: pip install pydub")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Convert Bach Transformer generated music text to MIDI and MP3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.txt_to_audio --input outputs/generated_music.txt
  python -m src.txt_to_audio --input generated_music.txt --midi_out my_music.mid --mp3_out my_music.mp3
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to generated music text file'
    )
    
    parser.add_argument(
        '--midi_out',
        default='outputs/generated.mid',
        help='Path to save MIDI file (default: outputs/generated.mid)'
    )
    
    parser.add_argument(
        '--mp3_out',
        default='outputs/generated.mp3',
        help='Path to save MP3 file (default: outputs/generated.mp3)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=0.4,
        help='Note duration in seconds (default: 0.4)'
    )
    
    parser.add_argument(
        '--velocity',
        type=int,
        default=100,
        help='MIDI velocity 0-127 (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(os.path.dirname(args.midi_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.mp3_out), exist_ok=True)
    
    try:
        print(f"🎵 Converting {args.input} to audio...")
        
        # Parse the text file
        print(f"📖 Parsing music text file...")
        time_pitch_pairs = parse_generated_music_txt(args.input)
        print(f"✅ Extracted {len(time_pitch_pairs)} musical notes")
        
        if not time_pitch_pairs:
            print("❌ No musical notes found in input file")
            print("   Make sure the file contains lines like: '0.2 | F4 (MIDI 65)'")
            sys.exit(1)
        
        # Show sample notes
        print(f"\n🎼 Sample Notes:")
        for i, (time, pitch) in enumerate(time_pitch_pairs[:5]):
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = pitch // 12
            note_name = note_names[pitch % 12] + str(octave)
            print(f"   {time:5.1f}s | MIDI {pitch:3d} | {note_name}")
        
        if len(time_pitch_pairs) > 5:
            print(f"   ... and {len(time_pitch_pairs) - 5} more notes")
        
        # Create MIDI
        print(f"\n🎹 Creating MIDI file...")
        midi = create_midi_from_notes(time_pitch_pairs, args.duration, args.velocity)
        
        # Save MIDI file
        midi.write(args.midi_out)
        print(f"✅ MIDI saved to: {args.midi_out}")
        
        # Show MIDI info
        file_size = os.path.getsize(args.midi_out)
        print(f"📁 MIDI file size: {file_size} bytes")
        
        if midi.instruments and midi.instruments[0].notes:
            notes = midi.instruments[0].notes
            duration = max(note.end for note in notes) - min(note.start for note in notes)
            pitch_range = (min(n.pitch for n in notes), max(n.pitch for n in notes))
            
            print(f"🎼 Musical info: {duration:.1f}s duration, pitch range {pitch_range[0]}-{pitch_range[1]}")
        
        # Attempt MP3 conversion
        print(f"\n🎧 Attempting MP3 conversion...")
        mp3_success = convert_midi_to_mp3(args.midi_out, args.mp3_out)
        
        if mp3_success:
            print(f"✅ MP3 saved to: {args.mp3_out}")
        else:
            print(f"⚠️  MP3 conversion not available - MIDI file created successfully")
        
        print(f"\n🎉 Audio conversion completed!")
        print(f"🎵 You can play {args.midi_out} in any MIDI player, DAW, or music software")
        
        if not mp3_success:
            print(f"\n💡 To get MP3 audio:")
            print(f"   1. Open {args.midi_out} in MuseScore, GarageBand, or similar")
            print(f"   2. Export as MP3/WAV from the software")
            print(f"   3. Or use online MIDI to MP3 converters")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
