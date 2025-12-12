import cv2
import mediapipe as mp
import torch
import os

mp_face = mp.solutions.face_mesh

def preprocess_video(frames_dir):
    """
    Extracts face landmarks from frames and returns tensor [num_frames, 468*3].
    """
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    all_landmarks = []

    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        for frame_path in frames:
            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = []
                for lm in landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                all_landmarks.append(coords)
            else:
                # no face detected: append zeros
                all_landmarks.append([0.0]*468*3)
    
    tensor = torch.tensor(all_landmarks, dtype=torch.float32)  # [num_frames, 1404]
    return tensor
