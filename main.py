import os
import cv2
import random
import time
import warnings
import argparse
import logging
import numpy as np

from database import FaceDatabase, VoiceDatabase
from models import SCRFD, ArcFace, VoiceProcessor, AudioBuffer
from utils.logging import setup_logging
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
from utils.audio_utils import AudioVideoSynchronizer, create_voice_samples_directory


warnings.filterwarnings("ignore")
setup_logging(log_to_file=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition with FAISS")

    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_r50.onnx", help="Path to recognition model")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")
    parser.add_argument("--faces-dir", type=str, default="./assets/faces", help="Path to faces stored dir")
    parser.add_argument("--source", type=str, default="./assets/in_video.mp4", help="Video file or webcam source")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum number of face detections from a frame")
    parser.add_argument("--db-path", type=str, default="./database/face_database", help="path to vector db and metadata")
    parser.add_argument("--update-db", action="store_true", help="Force update of the face database")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Output path for annotated video")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display (for headless environments)")
    
    # Voice-related arguments
    parser.add_argument("--enable-voice", action="store_true", help="Enable voice identification features")
    parser.add_argument("--voices-dir", type=str, default="./assets/voices", help="Path to voice samples directory")
    parser.add_argument("--voice-db-path", type=str, default="./database/voice_database", help="Path to voice vector db and metadata")
    parser.add_argument("--voice-similarity-thresh", type=float, default=0.2, help="Similarity threshold for voice identification")
    parser.add_argument("--update-voice-db", action="store_true", help="Force update of the voice database")
    parser.add_argument("--audio-window", type=float, default=1.5, help="Audio window duration in seconds for voice analysis")
    parser.add_argument("--vad-aggressiveness", type=int, default=3, help="Voice Activity Detection aggressiveness (0-3)")
    parser.add_argument("--voice-only-mode", action="store_true", help="Only identify faces when voice matches (requires --enable-voice)")

    return parser.parse_args()


def build_face_database(detector: SCRFD, recognizer: ArcFace, params: argparse.Namespace, force_update: bool = False) -> FaceDatabase:
    face_db = FaceDatabase(db_path=params.db_path)

    if not force_update and face_db.load():
        logging.info("Loaded face database from disk.")
        return face_db

    logging.info("Building face database from images...")
    for filename in os.listdir(params.faces_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        name = filename.rsplit('.', 1)[0]
        image_path = os.path.join(params.faces_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"Could not read image: {image_path}")
            continue

        bboxes, kpss = detector.detect(image, max_num=1)

        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer.get_embedding(image, kpss[0])
        face_db.add_face(embedding, name)
        logging.info(f"Added face for: {name}")

    face_db.save()
    return face_db


def build_voice_database(voice_processor: VoiceProcessor, params: argparse.Namespace, force_update: bool = False) -> VoiceDatabase:
    """
    Build voice database from audio samples.
    """
    voice_db = VoiceDatabase(db_path=params.voice_db_path)

    if not force_update and voice_db.load():
        logging.info("Loaded voice database from disk.")
        return voice_db

    # Create voices directory if it doesn't exist
    if not os.path.exists(params.voices_dir):
        create_voice_samples_directory()
        logging.warning(f"Created voice samples directory: {params.voices_dir}")
        logging.warning("Please add voice samples and run again with --update-voice-db")
        return voice_db

    logging.info("Building voice database from audio samples...")
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    
    for filename in os.listdir(params.voices_dir):
        if not any(filename.lower().endswith(ext) for ext in audio_extensions):
            continue

        name = filename.rsplit('.', 1)[0]
        audio_path = os.path.join(params.voices_dir, filename)

        try:
            # Load audio file
            import librosa
            audio, sr = librosa.load(audio_path, sr=voice_processor.sample_rate)
            
            if len(audio) < voice_processor.sample_rate:  # Less than 1 second
                logging.warning(f"Audio file too short: {audio_path}. Skipping...")
                continue

            # Get speaker embedding
            embedding = voice_processor.get_speaker_embedding(audio)
            
            if embedding is not None:
                voice_db.add_voice(embedding, name)
                logging.info(f"Added voice for: {name}")
            else:
                logging.warning(f"Could not extract voice embedding from: {audio_path}")

        except Exception as e:
            logging.warning(f"Could not process audio file {audio_path}: {e}")
            continue

    voice_db.save()
    return voice_db


def frame_processor(frame: np.ndarray, detector: SCRFD, recognizer: ArcFace, face_db: FaceDatabase, 
                   colors: dict, params: argparse.Namespace, voice_processor: VoiceProcessor = None, 
                   voice_db: VoiceDatabase = None, audio_data: np.ndarray = None, 
                   frame_number: int = 0) -> np.ndarray:
    """
    Process frame with optional voice identification.
    """
    # Detect faces
    bboxes, kpss = detector.detect(frame, params.max_num)
    
    # Initialize voice analysis results
    has_speech = False
    speaker_name = "Unknown"
    voice_similarity = 0.0
    
    # Analyze audio if voice features are enabled
    if params.enable_voice and voice_processor is not None and voice_db is not None and audio_data is not None:
        try:
            has_speech, speaker_embedding = voice_processor.process_audio_chunk(audio_data)
            
            if has_speech and speaker_embedding is not None:
                speaker_name, voice_similarity = voice_db.search(speaker_embedding, params.voice_similarity_thresh)
                
                # Add voice activity indicator to frame
                cv2.putText(frame, "SPEECH DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if speaker_name != "Unknown":
                    cv2.putText(frame, f"Voice: {speaker_name} ({voice_similarity:.2f})", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "NO SPEECH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            logging.error(f"Voice processing error at frame {frame_number}: {e}")

    # Process detected faces
    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer.get_embedding(frame, kps)
        
        face_name, face_similarity = face_db.search(embedding, params.similarity_thresh)
        
        # Determine if we should show face identification
        show_face_id = True
        
        if params.voice_only_mode and params.enable_voice:
            # Only show face ID if voice is detected and matches
            if not has_speech:
                show_face_id = False
            elif speaker_name == "Unknown":
                show_face_id = False
            elif face_name != speaker_name:
                # Face and voice don't match - only show if both are the same person
                show_face_id = False
        
        # Draw bounding box and information
        if show_face_id and face_name != "Unknown":
            if face_name not in colors:
                colors[face_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Enhanced label with voice confirmation and speaking status
            label = face_name
            if params.enable_voice:
                if speaker_name == face_name:
                    label += "  (speaking)"  # Voice confirmed and speaking
                else:
                    label += "  (silent)"  # Someone is speaking (might be this person)
            
            draw_bbox_info(frame, bbox, similarity=face_similarity, name=label, color=colors[face_name])
        else:
            # Draw basic bounding box for unknown or non-matching faces
            color = (128, 128, 128) if params.voice_only_mode else (255, 0, 0)
            
            # Add speaking/silent status even for unknown faces if voice is enabled
            if params.enable_voice:
                status_label = "speaking" if has_speech else "silent"
                draw_bbox_info(frame, bbox, similarity=0.0, name=f"Unknown ({status_label})", color=color)
            else:
                draw_bbox(frame, bbox, color)

    return frame


def main(params):
    try:
        detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
        recognizer = ArcFace(params.rec_weight)
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    face_db = build_face_database(detector, recognizer, params, force_update=params.update_db)
    colors = {}

    # Initialize voice processing if enabled
    voice_processor = None
    voice_db = None
    audio_synchronizer = None
    
    if params.enable_voice:
        try:
            logging.info("Initializing voice processing...")
            voice_processor = VoiceProcessor(vad_mode=params.vad_aggressiveness)
            voice_db = build_voice_database(voice_processor, params, force_update=params.update_voice_db)
            
            # Validate voice-only mode requirements
            if params.voice_only_mode and voice_db.index.ntotal == 0:
                logging.warning("Voice-only mode enabled but no voice samples found. Disabling voice-only mode.")
                params.voice_only_mode = False
                
        except Exception as e:
            logging.error(f"Failed to initialize voice processing: {e}")
            logging.warning("Continuing without voice features")
            params.enable_voice = False

    cap = None
    out = None

    try:
        # Handle webcam vs video file
        is_webcam = isinstance(params.source, int)
        
        if is_webcam:
            cap = cv2.VideoCapture(params.source)
            logging.info(f"Opening webcam source: {params.source}")
            if params.enable_voice:
                logging.warning("Voice features not supported with webcam input")
                params.enable_voice = False
        else:
            if not os.path.exists(params.source):
                raise IOError(f"Video file does not exist: {params.source}")
            cap = cv2.VideoCapture(params.source)
            logging.info(f"Opening video file: {params.source}")
            
            # Initialize audio synchronizer for video files
            if params.enable_voice:
                try:
                    audio_synchronizer = AudioVideoSynchronizer(
                        params.source, 
                        audio_window=params.audio_window
                    )
                    logging.info("Audio-video synchronization initialized")
                except Exception as e:
                    logging.error(f"Failed to initialize audio synchronizer: {e}")
                    params.enable_voice = False
        
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {params.source}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
        
        # Initialize video writer
        out = cv2.VideoWriter(params.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count = 0
        total_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            
            # Get audio data for current frame if voice is enabled
            audio_data = None
            if params.enable_voice and audio_synchronizer is not None:
                try:
                    audio_data = audio_synchronizer.get_audio_for_frame(frame_count)
                except Exception as e:
                    logging.error(f"Failed to get audio for frame {frame_count}: {e}")
            
            # Process frame with optional voice data
            frame = frame_processor(
                frame, detector, recognizer, face_db, colors, params,
                voice_processor=voice_processor,
                voice_db=voice_db,
                audio_data=audio_data,
                frame_number=frame_count
            )
            
            end = time.time()

            out.write(frame)

            if not params.headless:
                cv2.imshow("Face Recognition with Voice", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            # Log progress periodically
            if frame_count % 100 == 0 or params.headless:
                elapsed_time = time.time() - total_start_time
                fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
                logging.info(f"Frame {frame_count}, Processing FPS: {1 / (end - start):.2f}, Overall FPS: {fps_current:.2f}")

            frame_count += 1

        total_time = time.time() - total_start_time
        logging.info(f"Processed {frame_count} frames in {total_time:.2f} seconds. Average FPS: {frame_count / total_time:.2f}")

    except Exception as e:
        logging.error(f"Error during video processing: {e}")
        raise

    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if not params.headless:
            cv2.destroyAllWindows()
        if audio_synchronizer is not None:
            audio_synchronizer.cleanup()


if __name__ == "__main__":
    args = parse_args()
    
    # Convert source to int if it's a webcam ID
    try:
        args.source = int(args.source)
    except ValueError:
        pass
    
    # Validate voice-related arguments
    if args.voice_only_mode and not args.enable_voice:
        logging.warning("--voice-only-mode requires --enable-voice. Enabling voice features.")
        args.enable_voice = True
    
    if args.enable_voice:
        logging.info("Voice identification features enabled")
        if args.voice_only_mode:
            logging.info("Voice-only mode: faces will only be identified when voice matches")
    
    main(args)
