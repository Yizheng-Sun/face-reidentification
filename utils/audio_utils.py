import cv2
import numpy as np
import librosa
import tempfile
import os
import logging
from typing import Optional, Tuple
import subprocess


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None, sample_rate: int = 16000) -> Optional[str]:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Path to output audio file (optional)
        sample_rate: Target sample rate
        
    Returns:
        Path to extracted audio file or None if failed
    """
    try:
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ac', '1',  # mono
            '-ar', str(sample_rate),  # sample rate
            '-y',  # overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Audio extracted to: {output_path}")
            return output_path
        else:
            logging.error(f"FFmpeg error: {result.stderr}")
            return None
            
    except Exception as e:
        logging.error(f"Failed to extract audio: {e}")
        return None


def load_audio_segment(audio_path: str, start_time: float, duration: float, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Load audio segment from file.
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        duration: Duration in seconds
        sample_rate: Target sample rate
        
    Returns:
        Audio data as numpy array or None if failed
    """
    try:
        audio, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            offset=start_time,
            duration=duration
        )
        return audio
    except Exception as e:
        logging.error(f"Failed to load audio segment: {e}")
        return None


def get_video_audio_info(video_path: str) -> Tuple[float, float, int, int]:
    """
    Get video and audio information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (duration, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return duration, fps, width, height


def frame_to_time(frame_number: int, fps: float) -> float:
    """
    Convert frame number to time in seconds.
    
    Args:
        frame_number: Frame number
        fps: Frames per second
        
    Returns:
        Time in seconds
    """
    return frame_number / fps if fps > 0 else 0


def time_to_frame(time_seconds: float, fps: float) -> int:
    """
    Convert time to frame number.
    
    Args:
        time_seconds: Time in seconds
        fps: Frames per second
        
    Returns:
        Frame number
    """
    return int(time_seconds * fps)


class AudioVideoSynchronizer:
    """
    Synchronize audio and video processing.
    """
    
    def __init__(self, video_path: str, audio_window: float = 1.0, sample_rate: int = 16000):
        """
        Initialize synchronizer.
        
        Args:
            video_path: Path to video file
            audio_window: Audio window duration in seconds
            sample_rate: Audio sample rate
        """
        self.video_path = video_path
        self.audio_window = audio_window
        self.sample_rate = sample_rate
        
        # Extract audio from video
        self.audio_path = extract_audio_from_video(video_path, sample_rate=sample_rate)
        if self.audio_path is None:
            raise ValueError("Failed to extract audio from video")
        
        # Get video info
        self.duration, self.fps, self.width, self.height = get_video_audio_info(video_path)
        
        logging.info(f"Video info - Duration: {self.duration:.2f}s, FPS: {self.fps}, Resolution: {self.width}x{self.height}")
    
    def get_audio_for_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get audio segment corresponding to a video frame.
        
        Args:
            frame_number: Video frame number
            
        Returns:
            Audio data for the time window around the frame
        """
        # Convert frame to time
        frame_time = frame_to_time(frame_number, self.fps)
        
        # Get audio window centered on frame time
        start_time = max(0, frame_time - self.audio_window / 2)
        
        # Load audio segment
        return load_audio_segment(
            self.audio_path,
            start_time,
            self.audio_window,
            self.sample_rate
        )
    
    def cleanup(self):
        """
        Clean up temporary files.
        """
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.unlink(self.audio_path)
                logging.info("Cleaned up temporary audio file")
            except Exception as e:
                logging.warning(f"Failed to cleanup audio file: {e}")


def create_voice_samples_directory():
    """
    Create directory structure for voice samples.
    """
    voice_dir = "./assets/voices"
    os.makedirs(voice_dir, exist_ok=True)
    
    # Create README for voice samples
    readme_content = """# Voice Samples Directory

Place audio files (.wav, .mp3, .m4a) for each person here.
The filename (without extension) will be used as the person's name for voice identification.

Example:
- john_doe.wav - Voice sample for John Doe
- jane_smith.mp3 - Voice sample for Jane Smith

Audio files should be:
- At least 3-5 seconds long
- Clear speech without background noise
- Single speaker only

Supported formats: .wav, .mp3, .m4a, .flac
"""
    
    readme_path = os.path.join(voice_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    return voice_dir 