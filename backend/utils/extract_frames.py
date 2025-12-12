import cv2
import os

def extract_frames(video_path, frames_dir, fps=25):
    """
    Extracts frames from video and saves in frames_dir.
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{count}.jpg"), frame)
        count += 1
    cap.release()
    return frames_dir
