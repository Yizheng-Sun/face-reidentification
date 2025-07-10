# Real-Time Face Re-Identification with FAISS, ArcFace, SCRFD & Voice Recognition

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-reidentification/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-reidentification)](https://github.com/yakhyo/face-reidentification/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-reidentification)

<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/Yizheng-Sun/face-reidentification/blob/main/output_video.mp4" muted="false" width="100%"></video>

This repository implements **multimodal person re-identification** using SCRFD for face detection, ArcFace for face recognition, and advanced voice identification. It supports inference from webcam or video sources with optional voice-based identity verification.

## Features

### 🎯 Core Features
- [x] **FAISS Vector Database Integration**: Enables fast and scalable face re-identification using a FAISS index built from facial embeddings. Faces must be placed in the `assets/faces/` directory.
- [x] **Face Detection**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) (SCRFD) for efficient and accurate face detection.
  - Added models: SCRFD 500M (2.41 MB), SCRFD 2.5G (3.14 MB), SCRFD 10G (16.1 MB)
- [x] **Face Recognition**: Employs [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) for robust face recognition.
  - Added models: ArcFace MobileFace (12.99 MB), ArcFace ResNet-50 (166 MB)
- [x] **Real-Time Inference**: Supports both webcam and video file input for real-time processing.

### 🎤 Voice Features (NEW!)
- [x] **Voice Activity Detection (VAD)**: Automatically detects when someone is speaking using WebRTC VAD
- [x] **Speaker Recognition**: Uses SpeechBrain's ECAPA-TDNN for robust speaker identification
- [x] **Audio-Visual Synchronization**: Synchronizes audio analysis with video frames for accurate identification
- [x] **Voice Database**: FAISS-based voice embedding database for fast speaker lookup
- [x] **Voice-Only Mode**: Optional mode that only identifies faces when the speaker's voice matches
- [x] **Multimodal Verification**: Cross-validates face and voice identity for enhanced accuracy

Project folder structure:

```
├── assets/
│   ├── demo.mp4
│   ├── in_video.mp4
│   ├── faces/              # Face images for identification
│   │   ├── person1.jpg
│   │   ├── person2.jpg
│   │   └── ...
│   └── voices/             # Voice samples for identification (NEW!)
│       ├── person1.wav
│       ├── person2.mp3
│       └── ...
├── database/
│   ├── __init__.py
│   ├── face_db.py
│   ├── voice_db.py         # Voice database management (NEW!)
│   ├── face_database/      # Face FAISS index and metadata
│   └── voice_database/     # Voice FAISS index and metadata (NEW!)
├── models/
│   ├── __init__.py
│   ├── scrfd.py
│   ├── arcface.py
│   └── voice_processor.py  # Voice processing and VAD (NEW!)
├── weights/
│   ├── det_10g.onnx
│   ├── det_2.5g.onnx
│   ├── det_500m.onnx
│   ├── w600k_r50.onnx
│   ├── w600k_mbf.onnx
│   └── spkrec-ecapa-voxceleb/  # Speaker recognition model (auto-downloaded)
├── utils/
│   ├── logging.py
│   ├── helpers.py
│   └── audio_utils.py      # Audio processing utilities (NEW!)
├── main.py
├── setup_voice.py          # Voice setup and testing utility (NEW!)
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yakyo/face-reidentification.git
cd face-reidentification
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note for Voice Features**: The voice recognition functionality requires additional audio processing libraries. If you encounter issues with audio dependencies, you may need to install `ffmpeg`:
> 
> - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
> - **macOS**: `brew install ffmpeg`
> - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

3. Download weight files:

   a) Download weights from following links:

   | Model              | Weights                                                                                                   | Size     | Type             |
   | ------------------ | --------------------------------------------------------------------------------------------------------- | -------- | ---------------- |
   | SCRFD 500M         | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx)   | 2.41 MB  | Face Detection   |
   | SCRFD 2.5G         | [det_2.5g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx)   | 3.14 MB  | Face Detection   |
   | SCRFD 10G          | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx)     | 16.1 MB  | Face Detection   |
   | ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 12.99 MB | Face Recognition |
   | ArcFace ResNet-50  | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB   | Face Recognition |

   b) Run below command to download weights to `weights` directory (linux):

   ```bash
   sh download.sh
   ```

4. Set up face and voice samples:

   **Face samples** (required): Put target faces into `assets/faces` folder
   ```
   faces/
       ├── person1.jpg
       ├── person2.jpg
       └── ...
   ```

   **Voice samples** (optional for voice features): Set up voice samples
   ```bash
   python setup_voice.py --setup  # Creates assets/voices/ directory
   ```
   Then add voice samples:
   ```
   voices/
       ├── person1.wav
       ├── person2.mp3
       └── ...
   ```

   > **Important**: Use the same person names for both face and voice files (e.g., `person1.jpg` and `person1.wav`)
   > 
   > **Voice file requirements**:
   > - Duration: 3-5 seconds minimum
   > - Format: .wav, .mp3, .m4a, .flac, .ogg
   > - Content: Clear speech, single speaker, minimal background noise

## Usage

### Basic Face Recognition

```bash
# Face recognition only (original functionality)
python main.py --source assets/in_video.mp4
```

### Voice Features Setup and Testing

```bash
# Set up voice samples directory
python setup_voice.py --setup

# Test a single voice sample
python setup_voice.py --test-file assets/voices/person1.wav

# Build and test voice database
python setup_voice.py --test-db
```

### Face + Voice Recognition

```bash
# Enable voice features with face recognition
python main.py --source assets/in_video.mp4 --enable-voice

# Build voice database for the first time
python main.py --source assets/in_video.mp4 --enable-voice --update-voice-db

# Voice-only mode: only identify faces when voice matches
python main.py --source assets/in_video.mp4 --enable-voice --voice-only-mode

# Custom thresholds for better accuracy
python main.py --source assets/in_video.mp4 --enable-voice \
    --similarity-thresh 0.5 \
    --voice-similarity-thresh 0.7 \
    --audio-window 2.0
```

### Advanced Usage Examples

```bash
# Headless mode for server deployment
python main.py --source input.mp4 --output output.mp4 --headless --enable-voice

# High accuracy mode
python main.py --source input.mp4 --enable-voice --voice-only-mode \
    --similarity-thresh 0.6 --voice-similarity-thresh 0.8

# Webcam with face recognition only (voice not supported for webcam)
python main.py --source 0
```

### Command Line Arguments

#### Core Arguments
```
--source SOURCE               Video file path or webcam ID (0, 1, etc.)
--output OUTPUT               Output video path (default: output_video.mp4)
--headless                    Run without GUI display
--max-num MAX_NUM             Maximum face detections per frame (0 = unlimited)
```

#### Face Recognition Arguments
```
--det-weight DET_WEIGHT       Path to face detection model
--rec-weight REC_WEIGHT       Path to face recognition model
--faces-dir FACES_DIR         Directory containing face images
--similarity-thresh FLOAT     Face similarity threshold (default: 0.4)
--confidence-thresh FLOAT     Face detection confidence threshold (default: 0.5)
--db-path DB_PATH             Face database path
--update-db                   Force rebuild face database
```

#### Voice Recognition Arguments
```
--enable-voice                Enable voice identification features
--voice-only-mode             Only identify faces when voice matches
--voices-dir VOICES_DIR       Directory containing voice samples
--voice-db-path VOICE_DB_PATH Voice database path
--voice-similarity-thresh FLOAT  Voice similarity threshold (default: 0.6)
--update-voice-db             Force rebuild voice database
--audio-window FLOAT          Audio analysis window in seconds (default: 1.5)
--vad-aggressiveness INT      Voice activity detection sensitivity (0-3, default: 3)
```

### Output Features

When voice features are enabled, the system displays:
- **Green "SPEECH DETECTED"** when someone is speaking
- **Red "NO SPEECH"** when no speech is detected
- **Voice identification** with confidence score
- **Face identification** with ✓ mark when voice confirms identity
- **Synchronized audio-visual analysis** for video files

> **Note**: Voice features are only available for video files, not webcam input due to audio synchronization requirements.

## Voice Features Details

### How Voice Identification Works

1. **Audio Extraction**: Audio is extracted from video files using FFmpeg
2. **Voice Activity Detection**: WebRTC VAD detects speech segments in real-time
3. **Speaker Embedding**: SpeechBrain's ECAPA-TDNN extracts speaker embeddings
4. **Voice Database**: FAISS indexes voice embeddings for fast similarity search
5. **Synchronization**: Audio analysis is synchronized with video frames
6. **Multimodal Fusion**: Face and voice identities are cross-validated

### Performance Considerations

- **Voice features add ~2-3x processing time** compared to face-only recognition
- **Audio window size** affects accuracy vs. latency (1.5s recommended)
- **Voice similarity threshold** should be tuned based on your voice samples quality
- **Memory usage** increases with voice database size (192D embeddings per person)

### Troubleshooting

**Common Issues:**
- **"Failed to extract audio"**: Install FFmpeg and ensure video has audio track
- **"No speech detected"**: Adjust VAD aggressiveness or check audio quality
- **"Failed to load speaker model"**: Check internet connection (model downloads automatically)
- **Low voice similarity scores**: Ensure voice samples are high quality and 3+ seconds long

**Tips for Better Accuracy:**
- Use high-quality voice samples (clear speech, minimal background noise)
- Match voice sample conditions to target video (similar audio quality)
- Adjust thresholds based on your specific use case
- Use voice-only mode for strict identity verification

## References

### Face Recognition
1. [SCRFD: Sample and Computation Redistribution for Efficient Face Detection](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
2. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

### Voice Recognition
3. [SpeechBrain: A PyTorch-based Speech Toolkit](https://github.com/speechbrain/speechbrain)
4. [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN](https://arxiv.org/abs/2005.07143)
5. [WebRTC Voice Activity Detector](https://github.com/wiseman/py-webrtcvad)

### Vector Databases
6. [FAISS: A library for efficient similarity search](https://github.com/facebookresearch/faiss)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the InsightFace team for face detection and recognition models
- Thanks to SpeechBrain team for speaker recognition capabilities
- Thanks to Facebook Research for FAISS vector database
- Thanks to all contributors and the open-source community
