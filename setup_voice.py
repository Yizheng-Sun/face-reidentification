#!/usr/bin/env python3
"""
Voice setup and testing utility for face re-identification system.
"""

import os
import argparse
import logging
from utils.audio_utils import create_voice_samples_directory
from models import VoiceProcessor
from database import VoiceDatabase
import librosa
import numpy as np


def setup_voice_directory():
    """Create voice samples directory with instructions."""
    voice_dir = create_voice_samples_directory()
    print(f"✓ Voice samples directory created: {voice_dir}")
    print("\nNext steps:")
    print("1. Add audio files (.wav, .mp3, .m4a) for each person to the voices directory")
    print("2. Name files as: person_name.extension (e.g., john_doe.wav)")
    print("3. Ensure audio files are 3-5 seconds long with clear speech")
    print("4. Run: python main.py --enable-voice --update-voice-db")
    return voice_dir


def test_voice_sample(audio_path: str):
    """Test a single voice sample."""
    try:
        voice_processor = VoiceProcessor()
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=voice_processor.sample_rate)
        
        print(f"Testing audio file: {audio_path}")
        print(f"Duration: {len(audio) / voice_processor.sample_rate:.2f} seconds")
        
        # Test voice activity detection
        has_speech = voice_processor.is_speech(audio)
        print(f"Speech detected: {has_speech}")
        
        # Test speaker embedding extraction
        embedding = voice_processor.get_speaker_embedding(audio)
        if embedding is not None:
            print(f"✓ Speaker embedding extracted successfully (size: {embedding.shape})")
        else:
            print("✗ Failed to extract speaker embedding")
            
        return has_speech and embedding is not None
        
    except Exception as e:
        print(f"✗ Error testing audio file: {e}")
        return False


def build_voice_database_test(voices_dir: str, db_path: str = "./database/voice_database_test"):
    """Build and test voice database."""
    try:
        voice_processor = VoiceProcessor()
        voice_db = VoiceDatabase(db_path=db_path)
        
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        processed_count = 0
        
        print(f"Building voice database from: {voices_dir}")
        
        for filename in os.listdir(voices_dir):
            if not any(filename.lower().endswith(ext) for ext in audio_extensions):
                continue
                
            name = filename.rsplit('.', 1)[0]
            audio_path = os.path.join(voices_dir, filename)
            
            print(f"\nProcessing: {filename} -> {name}")
            
            if test_voice_sample(audio_path):
                # Load and process audio
                audio, sr = librosa.load(audio_path, sr=voice_processor.sample_rate)
                embedding = voice_processor.get_speaker_embedding(audio)
                
                if embedding is not None:
                    voice_db.add_voice(embedding, name)
                    processed_count += 1
                    print(f"✓ Added to database")
        
        if processed_count > 0:
            voice_db.save()
            print(f"\n✓ Voice database built successfully with {processed_count} voices")
            print(f"Database saved to: {db_path}")
            
            # Test similarity search
            print("\n--- Testing voice similarity ---")
            if voice_db.index.ntotal >= 2:
                # Test with first voice
                test_name = voice_db.metadata[0]
                test_embedding = np.random.randn(voice_db.embedding_size)  # Random embedding for demo
                result_name, similarity = voice_db.search(test_embedding, threshold=0.5)
                print(f"Test search result: {result_name} (similarity: {similarity:.3f})")
            
        else:
            print("\n✗ No valid voice samples found")
            
    except Exception as e:
        print(f"✗ Error building voice database: {e}")


def main():
    parser = argparse.ArgumentParser(description="Voice setup and testing utility")
    parser.add_argument("--setup", action="store_true", help="Set up voice samples directory")
    parser.add_argument("--test-file", type=str, help="Test a single audio file")
    parser.add_argument("--test-db", action="store_true", help="Build and test voice database")
    parser.add_argument("--voices-dir", type=str, default="./assets/voices", help="Voice samples directory")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_voice_directory()
    
    elif args.test_file:
        if os.path.exists(args.test_file):
            test_voice_sample(args.test_file)
        else:
            print(f"File not found: {args.test_file}")
    
    elif args.test_db:
        if os.path.exists(args.voices_dir):
            build_voice_database_test(args.voices_dir)
        else:
            print(f"Voice directory not found: {args.voices_dir}")
            print("Run with --setup first")
    
    else:
        print("Voice Setup and Testing Utility")
        print("Usage:")
        print("  python setup_voice.py --setup                    # Create voice directory")
        print("  python setup_voice.py --test-file audio.wav      # Test single audio file")
        print("  python setup_voice.py --test-db                  # Build and test voice database")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 