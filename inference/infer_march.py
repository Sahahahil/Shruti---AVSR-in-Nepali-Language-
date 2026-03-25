"""
Real-Time AVSR Inference — Late Fusion of VSR + ASR (Updated March 2026)
====================================================
VSR  : CNN + Uni-LSTM
ASR  : Wav2Vec2 → CTC forced probabilities over 9 exact words + length normalization

Fixes applied:
  1. LipCNN normalisation: BatchNorm2d → GroupNorm to match training checkpoint
     (critical — mismatched norm layers caused silent weight loading failure)
  2. VSR/ASR temperature re-balanced: VSR 2.0→1.2 / ASR 0.5→0.8
     (VSR was being over-flattened in confidence fusion, ASR always won)
  3. Inference cadence: INFERENCE_EVERY_N_FRAMES 25→15 (more frequent VSR reads)
  4. Decision buffer: window 0.5s→1.0s, min_samples 3→2, stability_frames 3→2
     (old settings required ~3 s of consistent output before committing)

Usage same as before.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyaudio
import threading
import queue
import argparse
import time
import warnings
from collections import deque, Counter
import mediapipe as mp
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    VSR_CHECKPOINT = "/home/sahild/MajorProject/LM-VSR/CEEL/checkpoints-transfer-march/best_model.pth"
    WAV2VEC_PATH   = "/home/sahild/MajorProject/LM-VSR/wav2vec2-gold/wav2vec2-nepali-finetuned-v2-10-Dec/checkpoint-9200/"
    WAV2VEC_PROCESSOR_PATH = "/home/sahild/MajorProject/LM-VSR/wav2vec2-gold/wav2vec2-nepali-processor/"

    CLASSES     = ['अगाडि', 'जाऊ', 'तल', 'तिमी', 'दायाँ', 'पछाडि', 'बायाँ', 'माथि', 'रोक']
    NUM_CLASSES = 9

    NUM_FRAMES   = 50
    IMG_SIZE     = (64, 64)
    LIP_PAD_X    = 30
    LIP_PAD_Y    = 15
    LIP_ASPECT_RATIO = 2.0
    FACE_DETECTION_CONFIDENCE = 0.4
    FACE_TRACKING_CONFIDENCE = 0.6
    LANDMARK_SMOOTH_ALPHA = 0.65
    BBOX_SMOOTH_ALPHA = 0.7
    MAX_MISSED_LIP_FRAMES = 8
    MAX_WEAK_LIP_FRAMES = 6
    CONTOUR_MEDIAN_BLUR = 3
    CONTOUR_MORPH_KERNEL = 3
    ADAPTIVE_PAD_ENABLED = True
    MOUTH_OPEN_REF_RATIO = 0.25
    MOUTH_CLOSED_THRESHOLD = 0.10
    SPEECH_ON_FRAMES = 1
    SPEECH_OFF_FRAMES = 1
    MIN_PAD_SCALE = 0.85
    MAX_PAD_SCALE = 1.45
    MAX_LANDMARK_JUMP_PX = 22.0
    MIN_OUTER_AREA = 120.0
    MIN_FILL_RATIO = 0.05
    MAX_FILL_RATIO = 0.9
    DECISION_WINDOW_SEC = 0.0
    DECISION_MIN_SAMPLES = 1
    DECISION_MIN_CONFIDENCE = 0.45
    CNN_OUT_DIM  = 256
    LSTM_HIDDEN  = 256
    LSTM_LAYERS  = 2
    DROPOUT      = 0.0

    AUDIO_SR        = 16000
    AUDIO_CHUNK     = 1024
    AUDIO_WINDOW_S  = 2.0           # try 1.5 if still problematic
    AUDIO_CHANNELS  = 1

    FUSION_MODE = 'confidence'
    VSR_WEIGHT  = 0.6
    ASR_WEIGHT  = 0.4
    
    # Temperature scaling for calibration (lower = sharper, higher = flatter)
    # FIX: VSR was at 2.0 which over-flattened its probs → ASR always won fusion.
    # After fixing GroupNorm, VSR confidence is real — match temperatures more fairly.
    VSR_TEMPERATURE = 0.7
    ASR_TEMPERATURE = 0.8   # was 0.5 — slight softening; CTC length-norm already sharpens it

    INFERENCE_EVERY_N_FRAMES = 1
    SMOOTHING_ALPHA = 0.3
    STABILITY_FRAMES = 1
    MIN_CONFIDENCE = 0.4

    WINDOW_W = 900
    WINDOW_H = 600

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# VSR MODEL  (unchanged)
# ============================================================================

class LipCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # FIX: GroupNorm instead of BatchNorm2d — must match the training checkpoint.
        # The training script used GroupNorm(8, C). Loading a GroupNorm checkpoint into
        # BatchNorm2d with strict=False is silent: affine weights load fine but
        # running_mean/running_var stay at defaults (0/1), corrupting all activations
        # in eval() mode. This was the primary cause of VSR failure at inference time.
        self.bn1   = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.GroupNorm(8, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.GroupNorm(8, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.dropout         = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.dropout(x)


class CNN_LSTM_VSR(nn.Module):
    def __init__(self, num_classes=9, hidden_size=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.cnn  = LipCNN(dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        cnn_out = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_out)
        attn_w  = F.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)
        return self.fc(context)


# ============================================================================
# LIP DETECTOR  (notebook contour mapping)
# ============================================================================

class LipDetector:
    OUTER_LIPS = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ]
    INNER_LIPS = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ]
    INNER_UPPER_IDX = 5
    INNER_LOWER_IDX = 15

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.FACE_TRACKING_CONFIDENCE,
        )

        self.prev_outer_pts = None
        self.prev_inner_pts = None
        self.prev_bbox = None
        self.last_output = None
        self.missed_frames = 0
        self.weak_frames = 0

    @staticmethod
    def _ema_points(current_pts: np.ndarray, prev_pts: np.ndarray | None, alpha: float):
        if prev_pts is None:
            return current_pts
        return alpha * current_pts + (1.0 - alpha) * prev_pts

    @staticmethod
    def _ema_bbox(current_bbox: tuple[int, int, int, int], prev_bbox: tuple[int, int, int, int] | None, alpha: float):
        if prev_bbox is None:
            return current_bbox
        cx, cy, cw, ch = current_bbox
        px, py, pw, ph = prev_bbox
        x = int(alpha * cx + (1.0 - alpha) * px)
        y = int(alpha * cy + (1.0 - alpha) * py)
        w = int(alpha * cw + (1.0 - alpha) * pw)
        h = int(alpha * ch + (1.0 - alpha) * ph)
        return x, y, w, h

    @staticmethod
    def _mouth_open_ratio(inner_pts: np.ndarray, lip_h: int):
        if lip_h <= 0:
            return 0.0

        upper_y = float(inner_pts[LipDetector.INNER_UPPER_IDX][1])
        lower_y = float(inner_pts[LipDetector.INNER_LOWER_IDX][1])
        mouth_open_px = abs(lower_y - upper_y)
        return mouth_open_px / max(float(lip_h), 1.0)

    @staticmethod
    def _compute_pad_scale(inner_pts: np.ndarray, lip_h: int):
        mouth_open_ratio = LipDetector._mouth_open_ratio(inner_pts, lip_h)

        normalized = mouth_open_ratio / max(Config.MOUTH_OPEN_REF_RATIO, 1e-6)
        scale = 0.85 + 0.35 * normalized
        return float(np.clip(scale, Config.MIN_PAD_SCALE, Config.MAX_PAD_SCALE))

    def _is_geometry_reliable(self, outer_pts_i: np.ndarray, inner_pts_i: np.ndarray,
                              bw: int, bh: int, mean_jump_px: float):
        if bw <= 0 or bh <= 0:
            return False

        outer_area = float(abs(cv2.contourArea(outer_pts_i)))
        inner_area = float(abs(cv2.contourArea(inner_pts_i)))
        fill_ratio = outer_area / max(float(bw * bh), 1.0)

        if outer_area < Config.MIN_OUTER_AREA:
            return False
        if fill_ratio < Config.MIN_FILL_RATIO or fill_ratio > Config.MAX_FILL_RATIO:
            return False
        if inner_area >= 0.95 * outer_area:
            return False
        if mean_jump_px > Config.MAX_LANDMARK_JUMP_PX:
            return False

        return True

    def crop_lips(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            self.missed_frames += 1
            if self.last_output is not None and self.missed_frames <= Config.MAX_MISSED_LIP_FRAMES:
                return self.last_output

            self.prev_outer_pts = None
            self.prev_inner_pts = None
            self.prev_bbox = None
            self.last_output = None
            self.weak_frames = 0
            return None

        lm = res.multi_face_landmarks[0].landmark
        self.missed_frames = 0

        outer_pts = np.array([
            [int(lm[i].x * w), int(lm[i].y * h)] for i in self.OUTER_LIPS
        ], dtype=np.float32)
        inner_pts = np.array([
            [int(lm[i].x * w), int(lm[i].y * h)] for i in self.INNER_LIPS
        ], dtype=np.float32)

        prev_outer = self.prev_outer_pts
        outer_pts = self._ema_points(outer_pts, prev_outer, Config.LANDMARK_SMOOTH_ALPHA)
        inner_pts = self._ema_points(inner_pts, self.prev_inner_pts, Config.LANDMARK_SMOOTH_ALPHA)

        mean_jump_px = 0.0
        if prev_outer is not None:
            mean_jump_px = float(np.mean(np.linalg.norm(outer_pts - prev_outer, axis=1)))

        self.prev_outer_pts = outer_pts
        self.prev_inner_pts = inner_pts

        outer_pts_i = np.round(outer_pts).astype(np.int32)
        inner_pts_i = np.round(inner_pts).astype(np.int32)

        x, y, bw, bh = cv2.boundingRect(outer_pts_i)

        mouth_open_ratio = self._mouth_open_ratio(inner_pts, bh)
        pad_scale = 1.0
        if Config.ADAPTIVE_PAD_ENABLED:
            pad_scale = self._compute_pad_scale(inner_pts, bh)

        pad_x = max(1, int(round(Config.LIP_PAD_X * pad_scale)))
        pad_y = max(1, int(round(Config.LIP_PAD_Y * pad_scale)))

        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        bw = min(w - x, bw + 2 * pad_x)
        bh = min(h - y, bh + 2 * pad_y)

        desired_bw = int(Config.LIP_ASPECT_RATIO * bh)
        dw = desired_bw - bw
        x = max(0, x - dw // 2)
        bw = min(w - x, desired_bw)

        x, y, bw, bh = self._ema_bbox((x, y, bw, bh), self.prev_bbox, Config.BBOX_SMOOTH_ALPHA)
        self.prev_bbox = (x, y, bw, bh)

        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        bw = int(np.clip(bw, 1, w - x))
        bh = int(np.clip(bh, 1, h - y))

        if not self._is_geometry_reliable(outer_pts_i, inner_pts_i, bw, bh, mean_jump_px):
            self.weak_frames += 1
            if self.last_output is not None and self.weak_frames <= Config.MAX_WEAK_LIP_FRAMES:
                return self.last_output
            return None

        self.weak_frames = 0

        if bw <= 0 or bh <= 0:
            return None

        # SimpMaj-style contour map: white background, gray lips, black mouth opening.
        lip_roi = np.full((bh, bw), 255, dtype=np.uint8)
        outer_shifted = np.round(outer_pts - [x, y]).astype(np.int32)
        inner_shifted = np.round(inner_pts - [x, y]).astype(np.int32)

        cv2.fillPoly(lip_roi, [outer_shifted], 180)
        cv2.fillPoly(lip_roi, [inner_shifted], 0)

        if Config.CONTOUR_MEDIAN_BLUR >= 3 and Config.CONTOUR_MEDIAN_BLUR % 2 == 1:
            lip_roi = cv2.medianBlur(lip_roi, Config.CONTOUR_MEDIAN_BLUR)

        if Config.CONTOUR_MORPH_KERNEL >= 2:
            k = Config.CONTOUR_MORPH_KERNEL
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            lip_roi = cv2.morphologyEx(lip_roi, cv2.MORPH_OPEN, kernel)
            lip_roi = cv2.morphologyEx(lip_roi, cv2.MORPH_CLOSE, kernel)

        lip_model = cv2.resize(lip_roi, Config.IMG_SIZE)
        lip_tensor = torch.from_numpy(lip_model).float().unsqueeze(0) / 255.0
        lip_tensor = (lip_tensor - 0.5) / 0.5

        lip_bbox = (x, y, x + bw, y + bh)
        self.last_output = (lip_tensor, lip_bbox, lip_roi, float(mouth_open_ratio))
        return self.last_output

    def close(self):
        self.face_mesh.close()


# ============================================================================
# ASR — CTC forced path probabilities + LENGTH NORMALIZATION
# ============================================================================

class ASREngine:
    def __init__(self, wav2vec_path: str, device: str, processor_path: str = None):
        print(f"  Loading ASR from: {wav2vec_path}")
        proc_path = processor_path if processor_path else wav2vec_path
        print(f"  Processor from: {proc_path}")
        self.processor = Wav2Vec2Processor.from_pretrained(proc_path)
        self.model     = Wav2Vec2ForCTC.from_pretrained(wav2vec_path).to(device)
        self.model.eval()
        self.device    = device
        self.blank_id  = self.processor.tokenizer.pad_token_id or 0
        print("  ASR loaded ✓")

        # Optional: print token ids for first few words to verify tokenizer
        for w in Config.CLASSES[:3]:
            ids = self.processor.tokenizer.encode(w)
            print(f"  {w:8} → token ids: {ids} (len={len(ids)})")

    @torch.no_grad()
    def transcribe(self, audio_np: np.ndarray) -> tuple[str, np.ndarray]:
        """
        Returns (greedy_text_debug, class_probabilities [9])
        """
        # Skip very short / silent input
        if len(audio_np) < Config.AUDIO_SR * 0.5 or np.max(np.abs(audio_np)) < 0.02:
            return "", np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES

        # Normalize amplitude
        max_abs = np.max(np.abs(audio_np))
        if max_abs > 0:
            audio_np = audio_np / max_abs

        inputs = self.processor(
            audio_np, sampling_rate=Config.AUDIO_SR,
            return_tensors='pt', padding=True
        )
        input_values = inputs.input_values.to(self.device)
        logits = self.model(input_values).logits  # [1, T, V]

        # Greedy for debug only
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.decode(pred_ids[0]).strip().lower()

        log_probs = F.log_softmax(logits, dim=-1)  # [1, T, V]
        input_lengths = torch.full((1,), log_probs.size(1), dtype=torch.long, device=self.device)

        log_p = np.full(Config.NUM_CLASSES, float('-inf'), dtype=np.float32)

        for i, word in enumerate(Config.CLASSES):
            target_ids = self.processor.tokenizer.encode(word)
            if not target_ids:
                continue
            target = torch.tensor([target_ids], dtype=torch.long, device=self.device)
            target_len = len(target_ids)
            target_lengths = torch.tensor([target_len], dtype=torch.long, device=self.device)

            loss = F.ctc_loss(
                log_probs=log_probs.transpose(0, 1),  # [T, 1, V]
                targets=target.squeeze(0),            # [target_len] - 1D for ctc_loss
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='none',
                blank=self.blank_id
            )
            # ────────────────────────────────────────────────
            # KEY FIX: length normalization (per character)
            log_p[i] = -loss.item() / target_len
            # ────────────────────────────────────────────────

        # Debug print (comment out after confirming it works)
        print("\n[ASR debug] log_p (length-norm):")
        for w, lp in zip(Config.CLASSES, log_p):
            print(f"  {w:8} {lp:8.3f}")
        print("")

        # Protect against all -inf
        if np.all(np.isinf(log_p)):
            log_p.fill(0.0)
        else:
            finite_vals = log_p[np.isfinite(log_p)]
            if len(finite_vals) > 0:
                min_finite = finite_vals.min()
                log_p[np.isinf(log_p)] = min_finite - 30

        # Optional: mild temperature if too peaked/spiky after norm
        # log_p = log_p / 1.2

        probs = F.softmax(torch.from_numpy(log_p), dim=-1).numpy()

        # Debug probs
        print("[ASR debug] probs:", "  ".join(f"{p*100:5.1f}%" for p in probs))

        return text, probs


# ============================================================================
# AUDIO CAPTURE THREAD  (unchanged)
# ============================================================================

class AudioCapture:
    def __init__(self, audio_queue: queue.Queue):
        self.queue   = audio_queue
        self.running = False
        self.pa      = pyaudio.PyAudio()
        self.stream  = None
        self.buffer  = []

    def start(self):
        self.running = True
        self.stream  = self.pa.open(
            format=pyaudio.paFloat32,
            channels=Config.AUDIO_CHANNELS,
            rate=Config.AUDIO_SR,
            input=True,
            frames_per_buffer=Config.AUDIO_CHUNK,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        audio_np = np.frombuffer(in_data, dtype=np.float32)
        self.buffer.append(audio_np)

        total_samples = sum(len(b) for b in self.buffer)
        if total_samples >= int(Config.AUDIO_SR * Config.AUDIO_WINDOW_S):
            combined = np.concatenate(self.buffer)
            self.queue.put(combined.copy())
            self.buffer = []

        return (None, pyaudio.paContinue)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


# ============================================================================
# FUSION  (unchanged)
# ============================================================================

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to calibrate confidence.
    Lower temp = sharper (more confident), Higher temp = flatter (less confident)
    """
    if temperature == 1.0:
        return probs
    # Convert to log space, scale, then back to probs
    log_probs = np.log(probs + 1e-10)
    scaled = log_probs / temperature
    return F.softmax(torch.from_numpy(scaled), dim=-1).numpy()


def fuse_predictions(vsr_probs: np.ndarray, asr_probs: np.ndarray,
                     mode: str = 'confidence') -> tuple[np.ndarray, float, float]:
    # Apply temperature scaling to calibrate confidence levels
    vsr_calibrated = apply_temperature(vsr_probs, Config.VSR_TEMPERATURE)
    asr_calibrated = apply_temperature(asr_probs, Config.ASR_TEMPERATURE)
    
    if mode == 'confidence':
        vsr_conf = float(np.max(vsr_calibrated))
        asr_conf = float(np.max(asr_calibrated))
        total    = vsr_conf + asr_conf + 1e-8
        w_vsr    = vsr_conf / total
        w_asr    = asr_conf / total
    else:
        w_vsr = Config.VSR_WEIGHT
        w_asr = Config.ASR_WEIGHT

    fused = w_vsr * vsr_calibrated + w_asr * asr_calibrated

    # Prevent weak VSR from polluting fusion
    if np.max(vsr_calibrated) < 0.35:
        fused = asr_calibrated

    return fused, w_vsr, w_asr


# ============================================================================
# TEMPORAL SMOOTHING  (unchanged)
# ============================================================================

class PredictionSmoother:
    def __init__(self, num_classes: int, alpha: float = 0.3,
                 stability_frames: int = 3, min_confidence: float = 0.4):
        self.num_classes = num_classes
        self.alpha = alpha
        self.stability_frames = stability_frames
        self.min_confidence = min_confidence

        self.smoothed_probs = np.ones(num_classes) / num_classes
        self.candidate_pred = None
        self.candidate_count = 0
        self.stable_pred = None

    def update(self, raw_probs: np.ndarray) -> tuple[np.ndarray, int | None, bool]:
        self.smoothed_probs = (
            self.alpha * raw_probs +
            (1 - self.alpha) * self.smoothed_probs
        )

        current_pred = int(np.argmax(self.smoothed_probs))
        current_conf = float(np.max(self.smoothed_probs))

        is_stable = False
        if current_conf < self.min_confidence:
            self.candidate_pred = None
            self.candidate_count = 0
        elif current_pred == self.candidate_pred:
            self.candidate_count += 1
            if self.candidate_count >= self.stability_frames:
                self.stable_pred = current_pred
                is_stable = True
        else:
            self.candidate_pred = current_pred
            self.candidate_count = 1

        return self.smoothed_probs, self.stable_pred, is_stable


class TemporalDecisionBuffer:
    def __init__(self, window_sec: float, min_samples: int, min_confidence: float, num_classes: int):
        self.window_sec = window_sec
        self.min_samples = min_samples
        self.min_confidence = min_confidence
        self.num_classes = num_classes
        self.buffer = deque()
        self.last_committed_pred = None

    def clear(self):
        self.buffer.clear()
        self.last_committed_pred = None

    def add(self, probs: np.ndarray, now_t: float):
        self.buffer.append((now_t, probs.copy()))
        while self.buffer and (now_t - self.buffer[0][0]) > self.window_sec:
            self.buffer.popleft()

    def decide(self):
        if len(self.buffer) < self.min_samples:
            return None, np.zeros(self.num_classes, dtype=np.float32), False

        span = self.buffer[-1][0] - self.buffer[0][0]
        if span < self.window_sec * 0.8:
            return None, np.zeros(self.num_classes, dtype=np.float32), False

        avg_probs = np.mean([p for _, p in self.buffer], axis=0)
        conf = float(np.max(avg_probs))
        pred = int(np.argmax(avg_probs))

        if conf < self.min_confidence:
            return None, avg_probs, False

        self.last_committed_pred = pred
        return pred, avg_probs, True


# ============================================================================
# DISPLAY  (unchanged — latin labels due to OpenCV Devanagari issues)
# ============================================================================

COLOR_BG         = (18,  18,  18)
COLOR_ACCENT     = (0,  200, 120)
COLOR_VIDEO      = (66, 133, 244)
COLOR_AUDIO      = (234, 67,  53)
COLOR_FUSED      = (251, 188,  4)
COLOR_WHITE      = (240, 240, 240)
COLOR_GRAY       = (100, 100, 100)
COLOR_BOX_BG     = (35,  35,  35)
COLOR_STABLE     = (76, 175, 80)
COLOR_UNSTABLE   = (158, 158, 158)

LABEL_MAP_ROMAN = {
    'अगाडि': 'agadi',
    'जाऊ': 'jau',
    'तल': 'tala',
    'तिमी': 'timi',
    'दायाँ': 'daya',
    'पछाडि': 'pachhadi',
    'बायाँ': 'baya',
    'माथि': 'mathi',
    'रोक': 'rok',
}


def ui_label(name: str, upper: bool = False) -> str:
    val = LABEL_MAP_ROMAN.get(name, name)
    return val.upper() if upper else val


def draw_bar(canvas, x, y, w, h, value, color, label=None, show_pct=True):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BOX_BG, -1)
    filled = int(w * np.clip(value, 0, 1))
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_GRAY, 1)
    if show_pct:
        pct_text = f"{value*100:.1f}%"
        cv2.putText(canvas, pct_text, (x + w + 8, y + h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
    if label:
        cv2.putText(canvas, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)


def render_ui(frame_bgr, vsr_probs, asr_probs, fused_probs,
              vsr_pred, asr_pred, fused_pred,
              vsr_conf, asr_conf, w_vsr, w_asr,
              lip_bbox, lip_contour, fps, mode, is_stable=False):
    H, W = Config.WINDOW_H, Config.WINDOW_W
    canvas = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)

    cam_w, cam_h = 420, 360
    if frame_bgr is not None:
        resized = cv2.resize(frame_bgr, (cam_w, cam_h))
        canvas[80:80 + cam_h, 20:20 + cam_w] = resized
        if lip_bbox is not None:
            x1, y1, x2, y2 = lip_bbox
            sx = cam_w / frame_bgr.shape[1]
            sy = cam_h / frame_bgr.shape[0]
            lx1 = int(x1 * sx) + 20
            ly1 = int(y1 * sy) + 80
            lx2 = int(x2 * sx) + 20
            ly2 = int(y2 * sy) + 80
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), COLOR_ACCENT, 2)
            cv2.putText(canvas, "lips", (lx1, ly1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ACCENT, 1)

    if lip_contour is not None:
        contour_h, contour_w = 110, 220
        contour_preview = cv2.resize(lip_contour, (contour_w, contour_h), interpolation=cv2.INTER_NEAREST)
        contour_preview = cv2.cvtColor(contour_preview, cv2.COLOR_GRAY2BGR)
        y0, x0 = 470, 20
        canvas[y0:y0 + contour_h, x0:x0 + contour_w] = contour_preview
        cv2.rectangle(canvas, (x0, y0), (x0 + contour_w, y0 + contour_h), COLOR_ACCENT, 1)
        cv2.putText(canvas, "lip contour (bw)", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)

    cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.putText(canvas, f"Mode: {mode}", (20, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

    cv2.putText(canvas, "Real-Time AVSR", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
    cv2.putText(canvas, "Nepali Word Recognition", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

    px = 460
    bar_w = 360

    def draw_panel(title, color, probs, pred, conf, gate_w, panel_y):
        cv2.rectangle(canvas, (px, panel_y), (px + bar_w + 80, panel_y + 140),
                      COLOR_BOX_BG, -1)
        cv2.rectangle(canvas, (px, panel_y), (px + bar_w + 80, panel_y + 140),
                      color, 1)

        cv2.putText(canvas, title, (px + 8, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas, f"fusion weight: {gate_w*100:.0f}%",
                    (px + 8, panel_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1)

        for i in range(Config.NUM_CLASSES):
            by = panel_y + 52 + i * 14
            draw_bar(canvas, px + 8, by, bar_w - 20, 10, float(probs[i]), color)
            lbl = ui_label(Config.CLASSES[i] if i < len(Config.CLASSES) else f"c{i}")
            cv2.putText(canvas, lbl, (px + 8, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_GRAY, 1)

        pred_label = ui_label(Config.CLASSES[pred], upper=True) if pred is not None and pred < len(Config.CLASSES) else '---'
        cv2.rectangle(canvas, (px + bar_w - 10, panel_y + 8),
                      (px + bar_w + 78, panel_y + 48), color, -1)
        cv2.putText(canvas, pred_label,
                    (px + bar_w - 5, panel_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BG, 2)

        cv2.putText(canvas, f"{conf*100:.1f}%",
                    (px + bar_w - 2, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    draw_panel("VSR  (Video)", COLOR_VIDEO, vsr_probs, vsr_pred, vsr_conf, w_vsr, 30)
    draw_panel("ASR  (Audio)", COLOR_AUDIO, asr_probs, asr_pred, asr_conf, w_asr, 185)

    fy = 340
    cv2.rectangle(canvas, (px, fy), (px + bar_w + 80, fy + 230), COLOR_BOX_BG, -1)
    cv2.rectangle(canvas, (px, fy), (px + bar_w + 80, fy + 230), COLOR_FUSED, 2)
    cv2.putText(canvas, "AVSR (Fused)", (px + 8, fy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FUSED, 2)
    cv2.putText(canvas, f"VSR:{w_vsr*100:.0f}% + ASR:{w_asr*100:.0f}%",
                (px + 8, fy + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1)

    for i in range(Config.NUM_CLASSES):
        by = fy + 58 + i * 16
        draw_bar(canvas, px + 8, by, bar_w - 20, 12, float(fused_probs[i]), COLOR_FUSED)
        fused_lbl = ui_label(Config.CLASSES[i] if i < len(Config.CLASSES) else f"c{i}")
        cv2.putText(canvas, fused_lbl, (px + 8, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_GRAY, 1)

    fused_label = ui_label(Config.CLASSES[fused_pred], upper=True) if fused_pred is not None and fused_pred < len(Config.CLASSES) else '---'
    fused_devanagari_hint = f"({Config.CLASSES[fused_pred]})" if fused_pred is not None and fused_pred < len(Config.CLASSES) else ''

    stability_color = COLOR_STABLE if is_stable else COLOR_UNSTABLE
    stability_text = "STABLE" if is_stable else "..."
    cv2.circle(canvas, (px + bar_w + 60, fy + 205), 6, stability_color, -1)
    cv2.putText(canvas, stability_text, (px + bar_w + 20, fy + 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, stability_color, 1)

    cv2.putText(canvas, f">> {fused_label} {fused_devanagari_hint}",
                (px + 8, fy + 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_FUSED, 2)

    cv2.line(canvas, (450, 0), (450, H), COLOR_GRAY, 1)

    return canvas


# ============================================================================
# MAIN LOOP  (unchanged except debug print frequency reduced)
# ============================================================================

def run_inference(cfg: Config, mode: str = 'avsr'):
    print("=" * 70)
    print("  Real-Time AVSR — CTC length-normalized")
    print(f"  Mode   : {mode}")
    print(f"  Device : {cfg.DEVICE}")
    print(f"  Fusion : {cfg.FUSION_MODE}")
    print("=" * 70)

    vsr_model = None
    if mode in ('avsr', 'video_only'):
        print("\nLoading VSR...")
        ckpt = torch.load(cfg.VSR_CHECKPOINT, map_location=cfg.DEVICE)
        state = ckpt.get('model_state_dict', ckpt)

        checkpoint_classes = ckpt.get('classes', None) if isinstance(ckpt, dict) else None
        if checkpoint_classes:
            cfg.CLASSES = checkpoint_classes
            cfg.NUM_CLASSES = len(checkpoint_classes)

        vsr_model = CNN_LSTM_VSR(
            num_classes  = cfg.NUM_CLASSES,
            hidden_size  = cfg.LSTM_HIDDEN,
            num_layers   = cfg.LSTM_LAYERS,
            dropout      = 0.0,
        ).to(cfg.DEVICE)

        missing, unexpected = vsr_model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARN] Missing keys ({len(missing)}) — checkpoint may not fully match model:")
            for k in missing[:10]:
                print(f"    - {k}")
            # Abort if any CNN norm layers are missing — that means a BatchNorm/GroupNorm
            # mismatch is still present and VSR will silently produce garbage outputs.
            norm_missing = [k for k in missing if any(tag in k for tag in ('bn1', 'bn2', 'bn3', 'bn4'))]
            if norm_missing:
                raise RuntimeError(
                    f"\n[FATAL] CNN normalisation keys missing from checkpoint: {norm_missing}\n"
                    f"This means LipCNN in inference uses different norm layers than training.\n"
                    f"Check that both scripts use the same norm type (GroupNorm or BatchNorm2d)."
                )
        if unexpected:
            print(f"  [INFO] Unexpected keys (ignored): {len(unexpected)}")
        vsr_model.eval()
        print("  VSR loaded ✓")

    asr_engine = None
    if mode in ('avsr', 'audio_only'):
        print("\nLoading ASR...")
        asr_engine = ASREngine(cfg.WAV2VEC_PATH, cfg.DEVICE, cfg.WAV2VEC_PROCESSOR_PATH)

    lip_detector = None
    if mode in ('avsr', 'video_only'):
        print("\nInitializing MediaPipe...")
        lip_detector = LipDetector()
        print("  MediaPipe ready ✓")

    cap = None
    if mode in ('avsr', 'video_only'):
        print("\nOpening webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam!")
        print("  Webcam ready ✓")

    audio_queue   = queue.Queue(maxsize=5)
    audio_capture = None
    if mode in ('avsr', 'audio_only'):
        print("\nStarting audio capture...")
        audio_capture = AudioCapture(audio_queue)
        audio_capture.start()
        print("  Microphone ready ✓")

    frame_buffer  = deque(maxlen=cfg.NUM_FRAMES)
    vsr_history = deque(maxlen=5)
    frame_count   = 0

    vsr_probs  = np.ones(cfg.NUM_CLASSES) / cfg.NUM_CLASSES
    asr_probs  = np.ones(cfg.NUM_CLASSES) / cfg.NUM_CLASSES
    fused_probs = np.ones(cfg.NUM_CLASSES) / cfg.NUM_CLASSES

    vsr_pred = asr_pred = fused_pred = None
    vsr_conf = asr_conf = 0.0
    w_vsr_disp = w_asr_disp = 0.5
    lip_bbox = None
    lip_contour = None

    smoother = PredictionSmoother(
        num_classes=cfg.NUM_CLASSES,
        alpha=cfg.SMOOTHING_ALPHA,
        stability_frames=cfg.STABILITY_FRAMES,
        min_confidence=cfg.MIN_CONFIDENCE,
    )

    decision_buffer = TemporalDecisionBuffer(
        window_sec=cfg.DECISION_WINDOW_SEC,
        min_samples=cfg.DECISION_MIN_SAMPLES,
        min_confidence=cfg.DECISION_MIN_CONFIDENCE,
        num_classes=cfg.NUM_CLASSES,
    )

    mouth_open_ratio = 0.0
    speech_on_count = 0
    speech_off_count = 0
    is_speaking = False
    is_stable = False

    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()

    asr_lock = threading.Lock()

    def asr_worker():
        nonlocal asr_probs, asr_pred, asr_conf
        while True:
            try:
                audio_np = audio_queue.get(timeout=1.0)
                if audio_np is None:
                    break
                _, probs = asr_engine.transcribe(audio_np)
                with asr_lock:
                    asr_probs = probs
                    asr_pred  = int(np.argmax(probs))
                    asr_conf  = float(np.max(probs))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"  [ASR Error] {e}")

    asr_thread = None
    if asr_engine is not None:
        asr_thread = threading.Thread(target=asr_worker, daemon=True)
        asr_thread.start()

    print("\n" + "=" * 70)
    print("  Running! Press Q to quit.")
    print("=" * 70 + "\n")

    dummy_frame = np.full((cfg.WINDOW_H, cfg.WINDOW_W, 3), COLOR_BG, dtype=np.uint8)
    cv2.putText(dummy_frame, "Starting...", (300, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
    cv2.imshow("Real-Time AVSR", dummy_frame)

    debug_counter = 0

    try:
        while True:
            t0 = time.time()

            frame = None
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                if cv2.Laplacian(frame, cv2.CV_64F).var() < 20:
                    continue

                lip_bbox = None
                lip_contour = None
                result = lip_detector.crop_lips(frame)
                if result is not None:
                    lip_tensor, lip_bbox, lip_contour, mouth_open_ratio = result
                    frame_buffer.append(lip_tensor)

                    if mouth_open_ratio >= cfg.MOUTH_CLOSED_THRESHOLD:
                        speech_on_count += 1
                        speech_off_count = 0
                    else:
                        speech_off_count += 1
                        speech_on_count = 0
                else:
                    mouth_open_ratio = 0.0
                    speech_off_count += 1
                    speech_on_count = 0

                if (not is_speaking) and speech_on_count >= cfg.SPEECH_ON_FRAMES:
                    is_speaking = True
                    decision_buffer.clear()

                if is_speaking and speech_off_count >= cfg.SPEECH_OFF_FRAMES:
                    is_speaking = False
                    decision_buffer.clear()

                frame_count += 1

                if (frame_count % cfg.INFERENCE_EVERY_N_FRAMES == 0
                        and len(frame_buffer) >= cfg.NUM_FRAMES
                        and vsr_model is not None):

                    with torch.no_grad():
                        clip = torch.stack(list(frame_buffer)[-cfg.NUM_FRAMES:]).unsqueeze(0).to(cfg.DEVICE)
                        logits = vsr_model(clip)
                        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

                    vsr_probs = probs
                    vsr_pred  = int(np.argmax(probs))
                    vsr_conf  = float(np.max(probs))

                    vsr_history.append(vsr_pred)

                    if len(vsr_history) >= 3:
                        vsr_pred = Counter(vsr_history).most_common(1)[0][0]

            with asr_lock:
                _asr_probs = asr_probs.copy()

            if mode == 'video_only':
                fused_probs = vsr_probs
                fused_pred  = vsr_pred
                w_vsr_disp, w_asr_disp = 1.0, 0.0
            elif mode == 'audio_only':
                fused_probs = _asr_probs
                fused_pred  = asr_pred
                w_vsr_disp, w_asr_disp = 0.0, 1.0
            else:
                fused_probs, w_vsr_disp, w_asr_disp = fuse_predictions(
                    vsr_probs, _asr_probs, cfg.FUSION_MODE
                )
                fused_pred = int(np.argmax(fused_probs))

            smoothed_probs, stable_pred, is_stable = smoother.update(fused_probs)

            if is_speaking:
                display_probs = smoothed_probs
                display_pred = stable_pred if stable_pred is not None else fused_pred
                is_stable = display_pred is not None
            else:
                display_probs = np.zeros(cfg.NUM_CLASSES, dtype=np.float32)
                display_pred = None
                is_stable = False

            elapsed = time.time() - prev_time
            prev_time = time.time()
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            fps = float(np.mean(fps_buffer))

            ui = render_ui(
                frame_bgr   = frame,
                vsr_probs   = vsr_probs,
                asr_probs   = _asr_probs,
                fused_probs = display_probs,
                vsr_pred    = vsr_pred,
                asr_pred    = asr_pred,
                fused_pred  = display_pred,
                vsr_conf    = vsr_conf,
                asr_conf    = asr_conf,
                w_vsr       = w_vsr_disp,
                w_asr       = w_asr_disp,
                lip_bbox    = lip_bbox,
                lip_contour = lip_contour,
                fps         = fps,
                mode        = mode,
                is_stable   = is_stable,
            )

            cv2.imshow("Real-Time AVSR", ui)

            # Reduced debug frequency
            debug_counter += 1
            if debug_counter % 8 == 0:
                pred_names = Config.CLASSES
                fp = pred_names[display_pred] if display_pred is not None else '---'
                stability_marker = '✓' if is_stable else '~'
                speech_marker = 'talk' if is_speaking else 'idle'
                print(f"  Pred: {fp:<8} {stability_marker} [{speech_marker}] "
                      f"VSR:{vsr_conf*100:5.1f}%  ASR:{asr_conf*100:5.1f}%  FPS:{fps:.1f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\nShutting down...")
        if cap:
            cap.release()
        if audio_capture:
            audio_capture.stop()
        if audio_queue:
            audio_queue.put(None)
        if lip_detector:
            lip_detector.close()
        cv2.destroyAllWindows()
        print("Done.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Real-Time AVSR — Late Fusion VSR + ASR')
    parser.add_argument('--vsr_checkpoint', type=str, default=Config.VSR_CHECKPOINT)
    parser.add_argument('--wav2vec_path',   type=str, default=Config.WAV2VEC_PATH)
    parser.add_argument('--wav2vec_processor', type=str, default=Config.WAV2VEC_PROCESSOR_PATH)
    parser.add_argument('--mode', type=str, default='avsr',
                        choices=['avsr', 'video_only', 'audio_only'])
    parser.add_argument('--fusion', type=str, default='confidence',
                        choices=['confidence', 'weighted'])
    parser.add_argument('--vsr_weight', type=float, default=Config.VSR_WEIGHT)
    parser.add_argument('--asr_weight', type=float, default=Config.ASR_WEIGHT)
    parser.add_argument('--vsr_temp', type=float, default=Config.VSR_TEMPERATURE,
                        help='VSR temperature (higher=flatter, default=2.0)')
    parser.add_argument('--asr_temp', type=float, default=Config.ASR_TEMPERATURE,
                        help='ASR temperature (lower=sharper, default=0.5)')
    parser.add_argument('--audio_window', type=float, default=Config.AUDIO_WINDOW_S)

    args = parser.parse_args()

    Config.VSR_CHECKPOINT  = args.vsr_checkpoint
    Config.WAV2VEC_PATH    = args.wav2vec_path
    Config.WAV2VEC_PROCESSOR_PATH = args.wav2vec_processor
    Config.FUSION_MODE     = args.fusion
    Config.VSR_WEIGHT      = args.vsr_weight
    Config.ASR_WEIGHT      = args.asr_weight
    Config.VSR_TEMPERATURE = args.vsr_temp
    Config.ASR_TEMPERATURE = args.asr_temp
    Config.AUDIO_WINDOW_S  = args.audio_window

    run_inference(Config, mode=args.mode)


if __name__ == '__main__':
    main()