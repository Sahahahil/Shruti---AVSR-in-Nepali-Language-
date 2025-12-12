import os
import torch
from utils.extract_audio import extract_audio
from utils.extract_frames import extract_frames
from utils.preprocess_audio import preprocess_audio
from utils.preprocess_video import preprocess_video
from utils.decode_ctc import ctc_decode

from models.audio_model import AudioModel
from models.video_model import VideoModel
from models.fusion_model import FusionModel

# Dummy idx2char map
idx2char = {1: "अ", 2: "आ", 3: "इ", 4: "उ", 5: "ए", 6: "ओ"}

# Placeholder models
audio_model = AudioModel().eval()
video_model = VideoModel().eval()
fusion_model = FusionModel(vocab_size=len(idx2char)+1).eval()

def transcribe_video(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join("processed", f"{base_name}.wav")
    frames_dir = os.path.join("processed", base_name)

    extract_audio(video_path, audio_path)
    extract_frames(video_path, frames_dir)

    audio_tensor = preprocess_audio(audio_path)
    video_tensor = preprocess_video(frames_dir)

    with torch.no_grad():
        audio_feat = audio_model(audio_tensor.unsqueeze(0))
        video_feat = video_model(video_tensor.unsqueeze(0))
        logits = fusion_model(audio_feat, video_feat).squeeze(0)

    transcript = ctc_decode(logits, idx2char)
    return transcript
