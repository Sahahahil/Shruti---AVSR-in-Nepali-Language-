from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from video_model.infer_march import ASREngine, CNN_LSTM_VSR, Config, LipDetector, fuse_predictions


PROJECT_ROOT = Path(__file__).resolve().parent
VSR_CHECKPOINT_PATH = PROJECT_ROOT / "video_model" / "best_model.pth"
WAV2VEC_PATH = PROJECT_ROOT / "audio_model" / "asr_weights" / "wav2vec2-weights" / "checkpoint-9200"
PROCESSOR_PATH = (
    PROJECT_ROOT
    / "audio_model"
    / "asr_weights"
    / "wav2vec2-weights"
    / "wav2vec2-nepali-processor-20260315T054231Z-1-001"
    / "wav2vec2-nepali-processor"
)


class RealtimeModelRuntime:
    """Singleton runtime that loads trained ASR and VSR models once."""

    _instance: RealtimeModelRuntime | None = None

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        Config.DEVICE = self.device
        Config.VSR_CHECKPOINT = str(VSR_CHECKPOINT_PATH)
        Config.WAV2VEC_PATH = str(WAV2VEC_PATH)
        Config.WAV2VEC_PROCESSOR_PATH = str(PROCESSOR_PATH)

        # Bias fusion towards ASR because ASR model quality is stronger.
        Config.FUSION_MODE = "weighted"
        Config.VSR_WEIGHT = 0.3
        Config.ASR_WEIGHT = 0.7

        Config.NUM_FRAMES = 15
        Config.INFERENCE_EVERY_N_FRAMES = 1
        Config.AUDIO_WINDOW_S = 1.6

        self.vsr_model, self.classes = self._load_vsr_model()
        Config.CLASSES = self.classes
        Config.NUM_CLASSES = len(self.classes)

        self.asr_engine = ASREngine(
            wav2vec_path=Config.WAV2VEC_PATH,
            device=self.device,
            processor_path=Config.WAV2VEC_PROCESSOR_PATH,
        )

    @classmethod
    def get_instance(cls) -> RealtimeModelRuntime:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_vsr_model(self) -> tuple[CNN_LSTM_VSR, list[str]]:
        checkpoint = torch.load(Config.VSR_CHECKPOINT, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)

        checkpoint_classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
        classes = checkpoint_classes if checkpoint_classes else list(Config.CLASSES)

        model = CNN_LSTM_VSR(
            num_classes=len(classes),
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=Config.LSTM_LAYERS,
            dropout=0.0,
        ).to(self.device)

        model.load_state_dict(state, strict=False)
        model.eval()
        return model, classes


class RealtimeSession:
    def __init__(self) -> None:
        self.runtime = RealtimeModelRuntime.get_instance()
        self.mode = "avsr"
        self.classes = list(self.runtime.classes)

        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=Config.NUM_FRAMES)
        self.audio_buffer = np.array([], dtype=np.float32)

        n_classes = len(self.classes)
        self.vsr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        self.asr_probs = np.ones(n_classes, dtype=np.float32) / n_classes

        self.last_vsr_t = 0.0
        self.last_asr_t = 0.0

        self.lip_detector = LipDetector()

    def close(self) -> None:
        if self.lip_detector is not None:
            self.lip_detector.close()

    def set_mode(self, mode: str) -> None:
        if mode in {"avsr", "vsr_only", "asr_only"}:
            self.mode = mode

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.audio_buffer = np.array([], dtype=np.float32)
        n_classes = len(self.classes)
        self.vsr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        self.asr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        self.last_vsr_t = 0.0
        self.last_asr_t = 0.0

    def add_frame(self, frame_bgr: np.ndarray) -> dict | None:
        if self.mode == "asr_only":
            return None

        crop = self.lip_detector.crop_lips(frame_bgr)
        if crop is None:
            return None

        lip_tensor, _, _, _ = crop
        self.frame_buffer.append(lip_tensor)

        if len(self.frame_buffer) < Config.NUM_FRAMES:
            return None

        now = time.time()
        if now - self.last_vsr_t < 0.12:
            return None

        self.last_vsr_t = now
        clip = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.runtime.device)
        with torch.no_grad():
            logits = self.runtime.vsr_model(clip)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        self.vsr_probs = probs.astype(np.float32)
        if self.mode in {"avsr", "vsr_only"}:
            return self.predict()
        return None

    def add_audio(self, samples: list[float], sample_rate: int | None = None) -> dict | None:
        if not samples:
            return None

        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0:
            return None

        src_rate = int(sample_rate) if sample_rate else 48000
        audio_16k = self._resample_to_16k(arr, src_rate)

        if audio_16k.size == 0:
            return None

        self.audio_buffer = np.concatenate([self.audio_buffer, audio_16k])

        max_keep = int(Config.AUDIO_SR * 4.0)
        if self.audio_buffer.size > max_keep:
            self.audio_buffer = self.audio_buffer[-max_keep:]

        needed = int(Config.AUDIO_SR * Config.AUDIO_WINDOW_S)
        if self.audio_buffer.size < needed:
            return None

        now = time.time()
        if now - self.last_asr_t < 0.2:
            return None

        self.last_asr_t = now
        window = self.audio_buffer[-needed:]
        _, probs = self.runtime.asr_engine.transcribe(window)
        self.asr_probs = probs.astype(np.float32)

        if self.mode in {"avsr", "asr_only"}:
            return self.predict()
        return None

    def predict(self) -> dict:
        v_idx = int(np.argmax(self.vsr_probs))
        a_idx = int(np.argmax(self.asr_probs))

        v_conf = float(np.max(self.vsr_probs))
        a_conf = float(np.max(self.asr_probs))

        if self.mode == "vsr_only":
            return {
                "type": "prediction",
                "mode": "vsr_only",
                "prediction": self.classes[v_idx],
                "confidence": v_conf,
                "latency": 150,
                "vsr_prediction": self.classes[v_idx],
                "vsr_confidence": v_conf,
            }

        if self.mode == "asr_only":
            return {
                "type": "prediction",
                "mode": "asr_only",
                "prediction": self.classes[a_idx],
                "confidence": a_conf,
                "latency": 140,
                "asr_prediction": self.classes[a_idx],
                "asr_confidence": a_conf,
            }

        fused_probs, w_vsr, w_asr = fuse_predictions(self.vsr_probs, self.asr_probs, mode=Config.FUSION_MODE)
        f_idx = int(np.argmax(fused_probs))
        f_conf = float(np.max(fused_probs))

        return {
            "type": "prediction",
            "mode": "avsr",
            "prediction": self.classes[f_idx],
            "confidence": f_conf,
            "latency": 180,
            "vsr_prediction": self.classes[v_idx],
            "vsr_confidence": v_conf,
            "asr_prediction": self.classes[a_idx],
            "asr_confidence": a_conf,
            "vsr_weight": float(w_vsr),
            "asr_weight": float(w_asr),
        }

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
        if src_rate <= 0:
            src_rate = 48000

        if src_rate == Config.AUDIO_SR:
            return audio

        in_len = audio.shape[0]
        out_len = int(round(in_len * Config.AUDIO_SR / src_rate))
        if in_len < 2 or out_len < 2:
            return np.array([], dtype=np.float32)

        x_old = np.linspace(0.0, 1.0, num=in_len, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)
