from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

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
        self.model_path = self._resolve_hf_dir(WAV2VEC_PATH, ["config.json"])
        self.processor_path = self._resolve_hf_dir(PROCESSOR_PATH, ["preprocessor_config.json"])

        Config.DEVICE = self.device
        Config.VSR_CHECKPOINT = str(VSR_CHECKPOINT_PATH)
        Config.WAV2VEC_PATH = str(self.model_path)
        Config.WAV2VEC_PROCESSOR_PATH = str(self.processor_path)

        # Bias AVSR towards ASR for stronger audio-driven predictions.
        Config.FUSION_MODE = "weighted"
        Config.VSR_WEIGHT = 0.3
        Config.ASR_WEIGHT = 0.7

        Config.NUM_FRAMES = 15
        Config.INFERENCE_EVERY_N_FRAMES = 1
        Config.AUDIO_WINDOW_S = 2.0

        self.vsr_model: CNN_LSTM_VSR | None = None
        self.classes = list(Config.CLASSES)

        self.asr_engine = ASREngine(
            wav2vec_path=Config.WAV2VEC_PATH,
            device=self.device,
            processor_path=Config.WAV2VEC_PROCESSOR_PATH,
        )
        self.char_asr_engine = CharacterASREngine(
            wav2vec_path=Config.WAV2VEC_PATH,
            processor_path=str(self.processor_path),
            device=self.device,
        )

    @staticmethod
    def _resolve_hf_dir(path: Path, required_files: list[str]) -> Path:
        if not path.exists() or not path.is_dir():
            raise RuntimeError(f"Required directory does not exist: {path}")

        if all((path / name).exists() for name in required_files):
            return path

        subdirs = [p for p in path.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and all((subdirs[0] / name).exists() for name in required_files):
            return subdirs[0]

        raise RuntimeError(
            f"Could not find required files {required_files} in {path}"
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

    def ensure_vsr_loaded(self) -> tuple[CNN_LSTM_VSR, list[str]]:
        if self.vsr_model is None:
            self.vsr_model, self.classes = self._load_vsr_model()
            Config.CLASSES = self.classes
            Config.NUM_CLASSES = len(self.classes)
        return self.vsr_model, self.classes


class CharacterASREngine:
    """Character-level greedy CTC decoding from wav2vec2 checkpoint."""

    def __init__(self, wav2vec_path: str, processor_path: str, device: str) -> None:
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(wav2vec_path).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio_np: np.ndarray) -> tuple[str, float]:
        if audio_np.size == 0:
            return "", 0.0

        max_abs = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if max_abs > 0:
            audio_np = audio_np / max_abs

        inputs = self.processor(
            audio_np,
            sampling_rate=Config.AUDIO_SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to(self.device)

        logits = self.model(input_values, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        text = self.processor.batch_decode(pred_ids)[0].replace("|", " ").strip()

        token_conf = torch.max(probs, dim=-1).values
        confidence = float(token_conf.mean().item()) if token_conf.numel() else 0.0
        return text, confidence


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
        self.last_asr_char_t = 0.0

        self.avsr_smoothed_probs = np.ones(n_classes, dtype=np.float32) / n_classes

        # ASR-only utterance segmentation (similar to realtime.py: record until silence).
        self.asr_only_capturing = False
        self.asr_only_silence_samples = 0
        self.asr_only_energy_threshold = 0.015
        self.asr_only_silence_limit = int(Config.AUDIO_SR * 0.8)
        self.asr_only_max_samples = int(Config.AUDIO_SR * 5.0)
        self.asr_only_min_samples = int(Config.AUDIO_SR * 0.4)

        self.lip_detector: LipDetector | None = None

    def close(self) -> None:
        if self.lip_detector is not None:
            self.lip_detector.close()

    def set_mode(self, mode: str) -> None:
        if mode in {"avsr", "vsr_only", "asr_only"}:
            self.mode = mode
            if mode in {"avsr", "vsr_only"}:
                self._ensure_vsr_components()

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.audio_buffer = np.array([], dtype=np.float32)
        n_classes = len(self.classes)
        self.vsr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        self.asr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        self.last_vsr_t = 0.0
        self.last_asr_t = 0.0
        self.last_asr_char_t = 0.0
        self.avsr_smoothed_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        self.asr_only_capturing = False
        self.asr_only_silence_samples = 0

    def _ensure_vsr_components(self) -> None:
        _, classes = self.runtime.ensure_vsr_loaded()
        if self.classes != classes:
            self.classes = list(classes)
            n_classes = len(self.classes)
            self.vsr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
            self.asr_probs = np.ones(n_classes, dtype=np.float32) / n_classes
        if self.lip_detector is None:
            self.lip_detector = LipDetector()

    def add_frame(self, frame_bgr: np.ndarray) -> dict | None:
        if self.mode == "asr_only":
            return None

        self._ensure_vsr_components()
        if self.lip_detector is None:
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
            vsr_model, _ = self.runtime.ensure_vsr_loaded()
            logits = vsr_model(clip)
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

        if self.mode == "asr_only":
            chunk_rms = float(np.sqrt(np.mean(np.square(audio_16k)))) if audio_16k.size else 0.0
            is_speech = chunk_rms > self.asr_only_energy_threshold

            if is_speech:
                self.asr_only_capturing = True
                self.asr_only_silence_samples = 0
            elif self.asr_only_capturing:
                self.asr_only_silence_samples += int(audio_16k.size)

            if not self.asr_only_capturing:
                self.audio_buffer = np.array([], dtype=np.float32)
                return None

            if self.audio_buffer.size > self.asr_only_max_samples:
                should_finalize = True
            else:
                should_finalize = self.asr_only_silence_samples >= self.asr_only_silence_limit

            if not should_finalize:
                return None

            utterance = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
            self.asr_only_capturing = False
            self.asr_only_silence_samples = 0

            if utterance.size < self.asr_only_min_samples:
                return None

            _, probs = self.runtime.asr_engine.transcribe(utterance)
            probs = probs.astype(np.float32)
            pred_idx = int(np.argmax(probs))
            text = self.classes[pred_idx]
            confidence = float(np.max(probs))
            return {
                "type": "prediction",
                "mode": "asr_only",
                "prediction": text,
                "confidence": confidence,
                "latency": 200,
                "asr_prediction": text,
                "asr_confidence": confidence,
            }

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
        alpha = float(getattr(Config, "SMOOTHING_ALPHA", 0.3))
        self.avsr_smoothed_probs = alpha * fused_probs + (1.0 - alpha) * self.avsr_smoothed_probs
        self.avsr_smoothed_probs = self.avsr_smoothed_probs / max(float(np.sum(self.avsr_smoothed_probs)), 1e-8)

        f_idx = int(np.argmax(self.avsr_smoothed_probs))
        f_conf = float(np.max(self.avsr_smoothed_probs))

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
