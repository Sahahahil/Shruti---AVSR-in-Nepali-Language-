import os
import time
import torch
from utils.extract_audio import extract_audio
from utils.extract_frames import extract_frames
from utils.preprocess_audio import preprocess_audio
from utils.preprocess_video import preprocess_video
from utils.decode_ctc import ctc_decode

from models.audio_model import AudioModel
from models.video_model import VideoModel
from models.fusion_model import FusionModel

# Dummy idx2char map for Nepali characters
idx2char = {1: "अ", 2: "आ", 3: "इ", 4: "उ", 5: "ए", 6: "ओ"}

# Load models (lazy loaded for performance)
_models = {}

def load_models():
    """Load all models once"""
    global _models
    if not _models:
        try:
            _models["audio"] = AudioModel().eval()
            _models["video"] = VideoModel().eval()
            _models["fusion"] = FusionModel(vocab_size=len(idx2char)+1).eval()
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    return _models

def transcribe_video(video_path: str, mode: str = "avsr"):
    """
    Main transcription function supporting three modes:
    - "avsr": Audio-Visual Speech Recognition (fusion of audio + video)
    - "vsr_asr": Visual + Audio separately (VSR + ASR)
    - "asr_only": Audio only (ASR with video input)
    
    Args:
        video_path: Path to video file
        mode: Processing mode
        
    Returns:
        Dictionary with transcription results
    """
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join("processed", f"{base_name}.wav")
    frames_dir = os.path.join("processed", base_name)
    
    try:
        # Load models
        models = load_models()
        
        # Extract audio and frames
        extract_audio(video_path, audio_path)
        extract_frames(video_path, frames_dir)
        
        # Preprocess
        audio_tensor = preprocess_audio(audio_path)
        video_tensor = preprocess_video(frames_dir)
        
        processing_time = time.time() - start_time
        
        if mode == "avsr":
            # AVSR: Fusion of audio + video
            return _process_avsr(models, audio_tensor, video_tensor, processing_time)
        
        elif mode == "vsr_asr":
            # VSR + ASR: Process separately
            return _process_vsr_asr(models, audio_tensor, video_tensor, processing_time)
        
        elif mode == "asr_only":
            # ASR Only: Use only audio
            return _process_asr_only(models, audio_tensor, processing_time)
        
        else:
            # Default to AVSR
            return _process_avsr(models, audio_tensor, video_tensor, processing_time)
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise
    finally:
        # Cleanup
        cleanup_files(audio_path, frames_dir)

def _process_avsr(models, audio_tensor, video_tensor, processing_time):
    """Process using Audio-Visual fusion"""
    with torch.no_grad():
        audio_feat = models["audio"](audio_tensor.unsqueeze(0))
        video_feat = models["video"](video_tensor.unsqueeze(0))
        logits = models["fusion"](audio_feat, video_feat).squeeze(0)
    
    transcript = ctc_decode(logits, idx2char)
    
    return {
        "transcription": transcript,
        "confidence": 0.85,  # Placeholder
        "processing_time": processing_time,
        "mode": "AVSR"
    }

def _process_vsr_asr(models, audio_tensor, video_tensor, processing_time):
    """Process VSR and ASR separately"""
    with torch.no_grad():
        # ASR from audio
        asr_logits = models["audio"](audio_tensor.unsqueeze(0)).squeeze(0)
        asr_transcript = ctc_decode(asr_logits, idx2char)
        
        # VSR from video
        vsr_logits = models["video"](video_tensor.unsqueeze(0)).squeeze(0)
        vsr_transcript = ctc_decode(vsr_logits, idx2char)
    
    return {
        "asr_transcription": asr_transcript,
        "vsr_transcription": vsr_transcript,
        "asr_confidence": 0.82,  # Placeholder
        "vsr_confidence": 0.78,  # Placeholder
        "processing_time": processing_time,
        "mode": "VSR+ASR"
    }

def _process_asr_only(models, audio_tensor, processing_time):
    """Process using only audio (ASR)"""
    with torch.no_grad():
        logits = models["audio"](audio_tensor.unsqueeze(0)).squeeze(0)
    
    transcript = ctc_decode(logits, idx2char)
    
    return {
        "transcription": transcript,
        "confidence": 0.88,  # Placeholder
        "processing_time": processing_time,
        "mode": "ASR_Only"
    }

def extract_audio_only(video_path: str):
    """Extract only audio from video"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join("processed", f"{base_name}.wav")
    extract_audio(video_path, audio_path)
    return audio_path

def extract_video_features(video_path: str):
    """Extract video features only"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join("processed", base_name)
    extract_frames(video_path, frames_dir)
    video_tensor = preprocess_video(frames_dir)
    return video_tensor

def cleanup_files(audio_path: str, frames_dir: str):
    """Clean up temporary files"""
    import shutil
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Cleanup warning: {e}")
