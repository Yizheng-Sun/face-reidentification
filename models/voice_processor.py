import os
import logging
import numpy as np
import torch
import torchaudio
import librosa
import webrtcvad
from speechbrain.pretrained import SpeakerRecognition
from typing import Optional, Tuple, List
import tempfile


class VoiceProcessor:
    def __init__(self, sample_rate: int = 16000, vad_mode: int = 3):
        """
        Initialize voice processor with VAD and speaker recognition.
        
        Args:
            sample_rate: Audio sample rate (16kHz recommended for VAD)
            vad_mode: WebRTC VAD aggressiveness (0-3, 3 is most aggressive)
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        
        # Initialize speaker recognition model (ECAPA-TDNN)
        try:
            self.speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./weights/spkrec-ecapa-voxceleb"
            )
            logging.info("Loaded speaker recognition model successfully")
        except Exception as e:
            logging.error(f"Failed to load speaker recognition model: {e}")
            self.speaker_model = None
        
        # Audio buffer for processing
        self.frame_duration_ms = 30  # 30ms frames for VAD
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """
        Detect if audio contains speech using WebRTC VAD.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            bool: True if speech is detected
        """
        if len(audio_data) == 0:
            return False
            
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Process in 30ms frames
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            if len(frame) == self.frame_size:
                try:
                    if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                        speech_frames += 1
                    total_frames += 1
                except Exception:
                    continue
        
        # Consider it speech if more than 30% of frames contain speech
        return total_frames > 0 and (speech_frames / total_frames) > 0.3
    
    def get_speaker_embedding(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Speaker embedding vector or None if failed
        """
        if self.speaker_model is None or len(audio_data) == 0:
            return None
            
        try:
            # Convert to torch tensor
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            
            # Get embedding using speechbrain model
            embedding = self.speaker_model.encode_batch(audio_tensor)
            
            # Convert to numpy and normalize
            embedding_np = embedding.squeeze().detach().cpu().numpy()
            return embedding_np
            
        except Exception as e:
            logging.error(f"Failed to extract speaker embedding: {e}")
            return None
    
    def extract_audio_from_video_segment(self, video_path: str, start_time: float, duration: float) -> Optional[np.ndarray]:
        """
        Extract audio segment from video file.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio data as numpy array or None if failed
        """
        try:
            # Use librosa to extract audio
            audio, sr = librosa.load(
                video_path, 
                sr=self.sample_rate, 
                offset=start_time, 
                duration=duration
            )
            return audio
        except Exception as e:
            logging.error(f"Failed to extract audio from video: {e}")
            return None
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process audio chunk to detect speech and extract speaker embedding.
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            Tuple of (is_speech, speaker_embedding)
        """
        # Check if audio contains speech
        has_speech = self.is_speech(audio_chunk)
        
        # Extract speaker embedding if speech is detected
        speaker_embedding = None
        if has_speech:
            speaker_embedding = self.get_speaker_embedding(audio_chunk)
        
        return has_speech, speaker_embedding


class AudioBuffer:
    """
    Circular buffer to store audio data synchronized with video frames.
    """
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 2.0):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Audio sample rate
            buffer_duration: Buffer duration in seconds
        """
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
    
    def add_audio(self, audio_data: np.ndarray) -> None:
        """
        Add audio data to buffer.
        
        Args:
            audio_data: Audio data to add
        """
        data_len = len(audio_data)
        
        if data_len >= self.buffer_size:
            # If data is larger than buffer, take the last buffer_size samples
            self.buffer = audio_data[-self.buffer_size:].copy()
            self.write_pos = 0
            self.is_full = True
        else:
            # Add data to circular buffer
            if self.write_pos + data_len <= self.buffer_size:
                self.buffer[self.write_pos:self.write_pos + data_len] = audio_data
                self.write_pos += data_len
            else:
                # Wrap around
                part1_len = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = audio_data[:part1_len]
                self.buffer[:data_len - part1_len] = audio_data[part1_len:]
                self.write_pos = data_len - part1_len
            
            if self.write_pos >= self.buffer_size:
                self.write_pos = 0
                self.is_full = True
    
    def get_recent_audio(self, duration: float) -> np.ndarray:
        """
        Get recent audio data from buffer.
        
        Args:
            duration: Duration in seconds to retrieve
            
        Returns:
            Recent audio data
        """
        samples_needed = int(self.sample_rate * duration)
        samples_needed = min(samples_needed, self.buffer_size)
        
        if not self.is_full and self.write_pos < samples_needed:
            # Not enough data available
            return self.buffer[:self.write_pos].copy()
        
        if self.is_full or self.write_pos >= samples_needed:
            # Get recent samples
            if self.write_pos >= samples_needed:
                return self.buffer[self.write_pos - samples_needed:self.write_pos].copy()
            else:
                # Wrap around
                part1 = self.buffer[self.buffer_size - (samples_needed - self.write_pos):]
                part2 = self.buffer[:self.write_pos]
                return np.concatenate([part1, part2])
        
        return np.array([], dtype=np.float32) 