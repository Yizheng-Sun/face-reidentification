#!/usr/bin/env python3

import cv2
import os

def test_video_file(video_path):
    print(f"Testing video file: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ File does not exist: {video_path}")
        return False
    
    # Get file size
    file_size = os.path.getsize(video_path)
    print(f"📁 File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ OpenCV failed to open the video file")
        cap.release()
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened successfully!")
    print(f"📺 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    print(f"🎞️ Frame count: {frame_count}")
    print(f"⏱️ Duration: {frame_count/fps:.2f} seconds")
    
    # Try to read first frame
    ret, frame = cap.read()
    if ret:
        print("✅ Successfully read first frame")
    else:
        print("❌ Failed to read first frame")
    
    cap.release()
    return True

if __name__ == "__main__":
    # Test the problematic video file
    test_video_file("assets/in_video.mp4")
    
    # Also test demo video if it exists
    if os.path.exists("assets/demo.mp4"):
        print("\n" + "="*50)
        test_video_file("assets/demo.mp4") 