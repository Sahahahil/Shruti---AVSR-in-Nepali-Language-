import os
import shutil
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from utils.extract_audio import extract_audio


# 9-word closed vocabulary used in the current project.
CLASSES = ['अगाडि', 'जाऊ', 'तल', 'तिमी', 'दायाँ', 'पछाडि', 'बायाँ', 'माथि', 'रोक']
NUM_CLASSES = len(CLASSES)

BASE_DIR = Path(__file__).resolve().parent
VSR_CHECKPOINT = BASE_DIR / 'video_model' / 'best_model.pth'
WAV2VEC_PATH = BASE_DIR / 'audio_model' / 'asr_weights' / 'wav2vec2-weights' / 'checkpoint-9200'
WAV2VEC_PROCESSOR_PATH = (
    BASE_DIR
    / 'audio_model'
    / 'asr_weights'
    / 'wav2vec2-nepali-processor-20260315T054231Z-1-001'
    / 'wav2vec2-nepali-processor'
)

NUM_FRAMES = 50
IMG_SIZE = (64, 64)
LIP_PAD_X = 30
LIP_PAD_Y = 15
LIP_ASPECT_RATIO = 2.0

# Fusion calibration and balancing knobs.
# Defaulted to the ASR-prior settings that performed best in manual runs.
FUSION_MODE = 'weighted'  # options: confidence, weighted, hybrid
VSR_WEIGHT = 0.10
ASR_WEIGHT = 0.90
CONFIDENCE_BLEND = 0.25
VSR_TEMPERATURE = 3.0
ASR_TEMPERATURE = 0.3
MIN_MODALITY_WEIGHT = 0.10
MAX_VSR_WEIGHT = 0.35
VSR_FUSION_FLOOR = 0.20
ASR_INFORMATIVE_FLOOR = 0.14
ASR_DISAGREE_WEIGHT = 0.97

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_models: dict[str, object] = {}

OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
INNER_LIPS = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]


class LipCNN(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(8, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.GroupNorm(8, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.dropout(x)


class CNN_LSTM_VSR(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.cnn = LipCNN(dropout=0.0)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.attention = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.0), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, channels, height, width = x.shape
        cnn_out = self.cnn(x.view(bsz * timesteps, channels, height, width)).view(bsz, timesteps, -1)
        lstm_out, _ = self.lstm(cnn_out)
        attn_w = F.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)
        return self.fc(context)


def _ensure_paths_exist() -> None:
    missing = []
    if not VSR_CHECKPOINT.exists():
        missing.append(str(VSR_CHECKPOINT))
    if not WAV2VEC_PATH.exists():
        missing.append(str(WAV2VEC_PATH))
    if not WAV2VEC_PROCESSOR_PATH.exists():
        missing.append(str(WAV2VEC_PROCESSOR_PATH))

    if missing:
        raise FileNotFoundError('Missing model path(s): ' + ', '.join(missing))


def _load_vsr_model() -> CNN_LSTM_VSR:
    model = CNN_LSTM_VSR(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(VSR_CHECKPOINT, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_asr_models() -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC, int]:
    processor = Wav2Vec2Processor.from_pretrained(str(WAV2VEC_PROCESSOR_PATH))
    asr_model = Wav2Vec2ForCTC.from_pretrained(str(WAV2VEC_PATH)).to(DEVICE)
    asr_model.eval()
    blank_id = processor.tokenizer.pad_token_id or 0
    return processor, asr_model, blank_id


def load_models() -> dict[str, object]:
    global _models
    if _models:
        return _models

    _ensure_paths_exist()
    vsr_model = _load_vsr_model()
    processor, asr_model, blank_id = _load_asr_models()

    _models = {
        'vsr': vsr_model,
        'asr_processor': processor,
        'asr_model': asr_model,
        'asr_blank_id': blank_id,
    }
    return _models


def _extract_lip_tensor_sequence(video_path: str, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6,
    )

    outer_lips = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ]
    inner_lips = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ]

    lip_tensors: list[torch.Tensor] = []
    try:
        while len(lip_tensors) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            lip_map = np.full((IMG_SIZE[1], IMG_SIZE[0]), 255, dtype=np.uint8)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                outer_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in outer_lips], dtype=np.int32)
                inner_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in inner_lips], dtype=np.int32)

                x, y, bw, bh = cv2.boundingRect(outer_pts)
                x = max(0, x - LIP_PAD_X)
                y = max(0, y - LIP_PAD_Y)
                bw = min(w - x, bw + 2 * LIP_PAD_X)
                bh = min(h - y, bh + 2 * LIP_PAD_Y)

                desired_bw = int(LIP_ASPECT_RATIO * bh)
                dw = desired_bw - bw
                x = max(0, x - dw // 2)
                bw = min(w - x, desired_bw)

                if bw > 0 and bh > 0:
                    lip_roi = np.full((bh, bw), 255, dtype=np.uint8)
                    outer_shifted = outer_pts - [x, y]
                    inner_shifted = inner_pts - [x, y]
                    cv2.fillPoly(lip_roi, [outer_shifted], 180)
                    cv2.fillPoly(lip_roi, [inner_shifted], 0)
                    lip_map = cv2.resize(lip_roi, IMG_SIZE)

            lip_tensor = torch.from_numpy(lip_map).float().unsqueeze(0) / 255.0
            lip_tensor = (lip_tensor - 0.5) / 0.5
            lip_tensors.append(lip_tensor)
    finally:
        cap.release()
        face_mesh.close()

    if not lip_tensors:
        # No face at all in the video; keep shape stable with blank sequence.
        blank = torch.zeros((num_frames, 1, IMG_SIZE[1], IMG_SIZE[0]), dtype=torch.float32)
        return blank

    while len(lip_tensors) < num_frames:
        lip_tensors.append(lip_tensors[-1].clone())

    return torch.stack(lip_tensors[:num_frames], dim=0)


def _extract_lip_tensor_from_frame(
    frame_bgr: np.ndarray,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> torch.Tensor | None:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    outer_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in OUTER_LIPS], dtype=np.int32)
    inner_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in INNER_LIPS], dtype=np.int32)

    x, y, bw, bh = cv2.boundingRect(outer_pts)
    x = max(0, x - LIP_PAD_X)
    y = max(0, y - LIP_PAD_Y)
    bw = min(w - x, bw + 2 * LIP_PAD_X)
    bh = min(h - y, bh + 2 * LIP_PAD_Y)

    desired_bw = int(LIP_ASPECT_RATIO * bh)
    dw = desired_bw - bw
    x = max(0, x - dw // 2)
    bw = min(w - x, desired_bw)

    if bw <= 0 or bh <= 0:
        return None

    lip_roi = np.full((bh, bw), 255, dtype=np.uint8)
    outer_shifted = outer_pts - [x, y]
    inner_shifted = inner_pts - [x, y]
    cv2.fillPoly(lip_roi, [outer_shifted], 180)
    cv2.fillPoly(lip_roi, [inner_shifted], 0)

    lip_map = cv2.resize(lip_roi, IMG_SIZE)
    lip_tensor = torch.from_numpy(lip_map).float().unsqueeze(0) / 255.0
    lip_tensor = (lip_tensor - 0.5) / 0.5
    return lip_tensor


class RealtimeInferencer:
    def __init__(self, audio_window_s: float = 2.0):
        models = load_models()
        self.vsr_model: CNN_LSTM_VSR = models['vsr']
        self.asr_processor: Wav2Vec2Processor = models['asr_processor']
        self.asr_model: Wav2Vec2ForCTC = models['asr_model']
        self.asr_blank_id: int = models['asr_blank_id']

        self.audio_sr = 16000
        self.min_audio_samples = int(0.5 * self.audio_sr)
        self.max_audio_samples = int(audio_window_s * self.audio_sr)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.6,
        )

        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=NUM_FRAMES)
        self.audio_buffer: deque[np.ndarray] = deque()
        self.audio_samples = 0

        self.fusion_mode = FUSION_MODE
        self.vsr_weight = VSR_WEIGHT
        self.asr_weight = ASR_WEIGHT
        self.vsr_temperature = VSR_TEMPERATURE
        self.asr_temperature = ASR_TEMPERATURE

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.audio_buffer.clear()
        self.audio_samples = 0

    def close(self) -> None:
        self.face_mesh.close()

    def set_fusion(
        self,
        mode: str | None = None,
        vsr_weight: float | None = None,
        asr_weight: float | None = None,
        vsr_temperature: float | None = None,
        asr_temperature: float | None = None,
    ) -> None:
        if mode in {'confidence', 'weighted', 'hybrid'}:
            self.fusion_mode = mode

        if vsr_weight is not None and np.isfinite(vsr_weight):
            self.vsr_weight = float(np.clip(vsr_weight, 0.0, 1.0))

        if asr_weight is not None and np.isfinite(asr_weight):
            self.asr_weight = float(np.clip(asr_weight, 0.0, 1.0))

        if vsr_temperature is not None and np.isfinite(vsr_temperature):
            self.vsr_temperature = float(max(vsr_temperature, 1e-3))

        if asr_temperature is not None and np.isfinite(asr_temperature):
            self.asr_temperature = float(max(asr_temperature, 1e-3))

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        lip_tensor = _extract_lip_tensor_from_frame(frame_bgr, self.face_mesh)
        if lip_tensor is not None:
            self.frame_buffer.append(lip_tensor)
        elif self.frame_buffer:
            # Keep temporal length stable if lip landmarks briefly drop.
            self.frame_buffer.append(self.frame_buffer[-1].clone())

    def add_audio(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return

        chunk = samples.astype(np.float32)
        self.audio_buffer.append(chunk)
        self.audio_samples += chunk.size

        while self.audio_samples > self.max_audio_samples and self.audio_buffer:
            removed = self.audio_buffer.popleft()
            self.audio_samples -= removed.size

    def has_vsr_context(self) -> bool:
        return len(self.frame_buffer) >= NUM_FRAMES

    def has_asr_context(self) -> bool:
        return self.audio_samples >= self.min_audio_samples

    @torch.no_grad()
    def _predict_vsr(self) -> tuple[np.ndarray, str, float]:
        if not self.has_vsr_context():
            probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
            idx = int(np.argmax(probs))
            return probs, CLASSES[idx], float(np.max(probs))

        clip = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0).to(DEVICE)
        logits = self.vsr_model(clip).squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_conf = float(np.max(probs))
        return probs, CLASSES[pred_idx], pred_conf

    def _predict_asr(self) -> tuple[np.ndarray, str, float, str]:
        if not self.audio_buffer:
            probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
            idx = int(np.argmax(probs))
            return probs, CLASSES[idx], float(np.max(probs)), ''

        audio_np = np.concatenate(list(self.audio_buffer), axis=0).astype(np.float32)
        return _predict_asr(self.asr_processor, self.asr_model, self.asr_blank_id, audio_np)

    def predict(self, mode: str) -> dict:
        asr_probs, asr_word, asr_conf, greedy_text = self._predict_asr()

        if mode == 'asr_only':
            return {
                'prediction': greedy_text or asr_word,
                'confidence': asr_conf,
                'asr_prediction': greedy_text or asr_word,
                'asr_confidence': asr_conf,
            }

        vsr_probs, vsr_word, vsr_conf = self._predict_vsr()

        if mode == 'vsr_only':
            return {
                'prediction': vsr_word,
                'confidence': vsr_conf,
                'vsr_prediction': vsr_word,
                'vsr_confidence': vsr_conf,
            }

        if mode == 'vsr_asr':
            primary_word = asr_word if asr_conf >= vsr_conf else vsr_word
            primary_conf = max(asr_conf, vsr_conf)
            return {
                'prediction': primary_word,
                'confidence': primary_conf,
                'vsr_prediction': vsr_word,
                'vsr_confidence': vsr_conf,
                'asr_prediction': greedy_text or asr_word,
                'asr_confidence': asr_conf,
            }

        fused_probs, _, _ = _fuse(
            vsr_probs,
            asr_probs,
            mode=self.fusion_mode,
            vsr_weight=self.vsr_weight,
            asr_weight=self.asr_weight,
            vsr_temperature=self.vsr_temperature,
            asr_temperature=self.asr_temperature,
        )
        fused_idx = int(np.argmax(fused_probs))
        fused_conf = float(np.max(fused_probs))
        return {
            'prediction': CLASSES[fused_idx],
            'confidence': fused_conf,
            'vsr_prediction': vsr_word,
            'vsr_confidence': vsr_conf,
            'asr_prediction': greedy_text or asr_word,
            'asr_confidence': asr_conf,
        }


def _load_audio_np(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    return audio_np


@torch.no_grad()
def _predict_vsr(vsr_model: CNN_LSTM_VSR, video_path: str) -> tuple[np.ndarray, str, float]:
    clip = _extract_lip_tensor_sequence(video_path).unsqueeze(0).to(DEVICE)  # [1, T, 1, 64, 64]
    logits = vsr_model(clip).squeeze(0)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_conf = float(np.max(probs))
    return probs, CLASSES[pred_idx], pred_conf


@torch.no_grad()
def _predict_asr(
    processor: Wav2Vec2Processor,
    asr_model: Wav2Vec2ForCTC,
    blank_id: int,
    audio_np: np.ndarray,
) -> tuple[np.ndarray, str, float, str]:
    if len(audio_np) < 16000 * 0.5 or float(np.max(np.abs(audio_np))) < 0.02:
        probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        return probs, CLASSES[int(np.argmax(probs))], float(np.max(probs)), ''

    max_abs = float(np.max(np.abs(audio_np)))
    if max_abs > 0:
        audio_np = audio_np / max_abs

    inputs = processor(audio_np, sampling_rate=16000, return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(DEVICE)
    logits = asr_model(input_values).logits  # [1, T, V]

    pred_ids = torch.argmax(logits, dim=-1)
    # Keep realtime ASR text behavior aligned with backend/audio_model/asr_weights/realtime.py.
    greedy_text = processor.batch_decode(pred_ids)[0]
    greedy_text = greedy_text.replace('|', ' ').strip()

    log_probs = F.log_softmax(logits, dim=-1)
    input_lengths = torch.full((1,), log_probs.size(1), dtype=torch.long, device=DEVICE)

    log_p = np.full(NUM_CLASSES, float('-inf'), dtype=np.float32)
    for i, word in enumerate(CLASSES):
        target_ids = processor.tokenizer.encode(word)
        if not target_ids:
            continue
        target = torch.tensor(target_ids, dtype=torch.long, device=DEVICE)
        target_len = len(target_ids)
        target_lengths = torch.tensor([target_len], dtype=torch.long, device=DEVICE)

        loss = F.ctc_loss(
            log_probs=log_probs.transpose(0, 1),
            targets=target,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            reduction='none',
            blank=blank_id,
        )
        log_p[i] = -float(loss.item()) / max(target_len, 1)

    if np.all(np.isinf(log_p)):
        log_p.fill(0.0)
    else:
        finite_vals = log_p[np.isfinite(log_p)]
        min_finite = float(np.min(finite_vals)) if finite_vals.size else 0.0
        log_p[np.isinf(log_p)] = min_finite - 30.0

    probs = F.softmax(torch.from_numpy(log_p), dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_conf = float(np.max(probs))
    return probs, CLASSES[pred_idx], pred_conf, greedy_text


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return probs

    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    scaled = log_probs / temperature
    return F.softmax(torch.from_numpy(scaled), dim=-1).cpu().numpy()


def _fuse(
    bias_asr = 2
    bias_vsr = 1
    vsr_probs: np.ndarray,
    asr_probs: np.ndarray,
    mode: str = FUSION_MODE,
    vsr_weight: float = VSR_WEIGHT,
    asr_weight: float = ASR_WEIGHT,
    vsr_temperature: float = VSR_TEMPERATURE,
    asr_temperature: float = ASR_TEMPERATURE,
) -> tuple[np.ndarray, float, float]:
    vsr_cal = _apply_temperature(vsr_probs, vsr_temperature)
    asr_cal = _apply_temperature(asr_probs, asr_temperature)

    vsr_conf = float(np.max(vsr_cal))
    asr_conf = float(np.max(asr_cal))
    total = vsr_conf + asr_conf + 1e-8
    conf_w_vsr = vsr_conf / total
    conf_w_asr = asr_conf / total

    if mode == 'weighted':
        prior_total = max(vsr_weight + asr_weight, 1e-8)
        w_vsr = vsr_weight / prior_total
        w_asr = asr_weight / prior_total
    elif mode == 'confidence':
        w_vsr = conf_w_vsr
        w_asr = conf_w_asr
    else:
        # Hybrid mode: blend confidence-driven weights with prior modality weights.
        prior_total = max(vsr_weight + asr_weight, 1e-8)
        prior_w_vsr = vsr_weight / prior_total
        prior_w_asr = asr_weight / prior_total
        w_vsr = CONFIDENCE_BLEND * conf_w_vsr + (1.0 - CONFIDENCE_BLEND) * prior_w_vsr
        w_asr = CONFIDENCE_BLEND * conf_w_asr + (1.0 - CONFIDENCE_BLEND) * prior_w_asr

        # apply bias
        w_vsr *= bias_vsr
        w_asr *= bias_asr

        # renormalize
        norm = w_vsr + w_asr + 1e-8
        w_vsr /= norm
        w_asr /= norm

    # Bound per-modality contribution and cap VSR to prioritize ASR.
    w_vsr = float(np.clip(w_vsr, MIN_MODALITY_WEIGHT, 1.0 - MIN_MODALITY_WEIGHT))
    w_vsr = min(w_vsr, MAX_VSR_WEIGHT)
    w_asr = 1.0 - w_vsr

    fused = w_vsr * vsr_cal + w_asr * asr_cal

    # Guardrail: if modalities disagree and ASR is at least mildly informative,
    # strongly favor ASR to avoid pathological overconfident VSR dominance.
    vsr_top = int(np.argmax(vsr_cal))
    asr_top = int(np.argmax(asr_cal))
    if vsr_top != asr_top and asr_conf >= ASR_INFORMATIVE_FLOOR:
        w_asr = ASR_DISAGREE_WEIGHT
        w_vsr = 1.0 - w_asr
        fused = w_vsr * vsr_cal + w_asr * asr_cal

    # If VSR is too weak, let ASR take over fully.
    if vsr_conf < VSR_FUSION_FLOOR:
        fused = asr_cal
        w_vsr, w_asr = 0.0, 1.0

    fused = fused / np.sum(fused)
    return fused, w_vsr, w_asr


def transcribe_video(
    video_path: str,
    mode: str = 'avsr',
    fusion_mode: str = FUSION_MODE,
    vsr_weight: float = VSR_WEIGHT,
    asr_weight: float = ASR_WEIGHT,
    vsr_temperature: float = VSR_TEMPERATURE,
    asr_temperature: float = ASR_TEMPERATURE,
) -> dict:
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join('processed', f'{base_name}.wav')

    try:
        models = load_models()

        if mode == 'vsr_only':
            vsr_probs, vsr_word, vsr_conf = _predict_vsr(models['vsr'], video_path)
            processing_time = time.time() - start_time
            return {
                'transcription': vsr_word,
                'classification': vsr_word,
                'confidence': vsr_conf,
                'processing_time': processing_time,
                'mode': 'VSR_Only',
            }

        extract_audio(video_path, audio_path)
        audio_np = _load_audio_np(audio_path)

        asr_probs, asr_word, asr_conf, greedy_text = _predict_asr(
            models['asr_processor'],
            models['asr_model'],
            models['asr_blank_id'],
            audio_np,
        )

        processing_time = time.time() - start_time

        if mode == 'asr_only':
            return {
                'transcription': greedy_text or asr_word,
                'classification': asr_word,
                'confidence': asr_conf,
                'processing_time': processing_time,
                'mode': 'ASR_Only',
            }

        vsr_probs, vsr_word, vsr_conf = _predict_vsr(models['vsr'], video_path)

        if mode == 'vsr_asr':
            return {
                'asr_transcription': greedy_text or asr_word,
                'vsr_transcription': vsr_word,
                'asr_confidence': asr_conf,
                'vsr_confidence': vsr_conf,
                'processing_time': processing_time,
                'mode': 'VSR+ASR',
            }

        fused_probs, _, _ = _fuse(
            vsr_probs,
            asr_probs,
            mode=fusion_mode,
            vsr_weight=vsr_weight,
            asr_weight=asr_weight,
            vsr_temperature=vsr_temperature,
            asr_temperature=asr_temperature,
        )
        fused_idx = int(np.argmax(fused_probs))
        fused_conf = float(np.max(fused_probs))

        return {
            'transcription': CLASSES[fused_idx],
            'confidence': fused_conf,
            'processing_time': processing_time,
            'mode': 'AVSR',
        }
    except Exception as e:
        print(f'Error during transcription: {e}')
        raise
    finally:
        cleanup_files(audio_path)


def extract_audio_only(video_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join('processed', f'{base_name}.wav')
    extract_audio(video_path, audio_path)
    return audio_path


def cleanup_files(audio_path: str) -> None:
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)

        processed_root = Path('processed')
        if processed_root.exists():
            # Remove stale extracted frame dirs from old pipeline runs.
            for item in processed_root.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
    except Exception as e:
        print(f'Cleanup warning: {e}')
