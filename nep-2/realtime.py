import torch
import sounddevice as sd
import webrtcvad
import numpy as np
from scipy.signal import resample
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- Constants ---
SAMPLING_RATE = 16000
FRAME_DURATION_MS = 30  # ms for VAD (10, 20 or 30 only)
FRAME_SIZE = int(SAMPLING_RATE * FRAME_DURATION_MS / 1000)

# --- Model Paths ---
MODEL_PATH = "/home/sahilduwal/MajorProject/Shruti---AVSR-in-Nepali-Language-/nep-2/wav2vec2-nepali-finetuned-v2"
PROCESSOR_PATH = "/home/sahilduwal/MajorProject/Shruti---AVSR-in-Nepali-Language-/nep-2/wav2vec2-nepali-processor"

# --- Load processor and model ---
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Setup VAD ---
vad = webrtcvad.Vad(2)  # Aggressiveness (0-3): 0=least, 3=most aggressive

def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, sample_rate=SAMPLING_RATE)

def record_until_silence(max_duration=5):
    print("🎙️ Listening...")
    audio_buffer = []
    silence_frames = 0
    max_silence = int(0.8 * 1000 / FRAME_DURATION_MS)  # 0.8 sec silence tolerance

    stream = sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='int16', blocksize=FRAME_SIZE)
    with stream:
        while True:
            frame, _ = stream.read(FRAME_SIZE)
            frame = frame.flatten()
            frame_bytes = frame.tobytes()

            if is_speech(frame_bytes):
                silence_frames = 0
                audio_buffer.extend(frame)
            else:
                silence_frames += 1
                if silence_frames > max_silence:
                    break
            if len(audio_buffer) > SAMPLING_RATE * max_duration:
                break
    return np.array(audio_buffer).astype(np.float32) / 32768.0  # Normalize

def transcribe(audio):
    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.replace("|", " ").strip()

if __name__ == "__main__":
    print("🟢 Nepali Real-Time Speech Recognition with Voice Detection (Ctrl+C to stop)")
    try:
        while True:
            audio = record_until_silence()
            if len(audio) > 0:
                text = transcribe(audio)
                print("🗣️ You said:", text)
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
