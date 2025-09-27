import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import webrtcvad
import numpy as np
import json
import os
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Model

# --- Constants ---
SAMPLING_RATE = 16000
FRAME_DURATION_MS = 30  # ms for VAD (10, 20 or 30 only)
FRAME_SIZE = int(SAMPLING_RATE * FRAME_DURATION_MS / 1000)

# --- Model Paths ---
MODEL_PATH = "/home/sahil_duwal/MajorProject/Shruti---AVSR-in-Nepali-Language-/nep-2/wav2vec2_custom_head"
TOKENIZER_DIR = "/home/sahil_duwal/MajorProject/Shruti---AVSR-in-Nepali-Language-/nep-2/tokenizer_nepali"
PRETRAINED_W2V = "/home/sahil_duwal/MajorProject/Shruti---AVSR-in-Nepali-Language-/nep-2/wav2vec2-nepali-finetuned-v2"

# --- Custom Model Classes (from your training script) ---
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.relu = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)
        return x + residual

class CustomTransformerCTCHead(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_layers: int, num_heads: int, dropout: float, vocab_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        
        if input_dim != model_dim:
            self.input_proj = nn.Linear(input_dim, model_dim)
        else:
            self.input_proj = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.output_proj(x)
        return logits

class W2V2WithCustomHead(nn.Module):
    def __init__(self, pretrained_name: str, processor: Wav2Vec2Processor, tokenizer: Wav2Vec2CTCTokenizer,
                 freeze_w2v: bool=True, use_adapters: bool=True, adapter_dim: int=128,
                 trans_layers: int=4, trans_dim: int=768, trans_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.get_vocab())

        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.wav2vec_feature_dim = self.wav2vec.config.hidden_size

        if freeze_w2v:
            for p in self.wav2vec.parameters():
                p.requires_grad = False

        self.use_adapters = use_adapters
        if use_adapters:
            self.adapter = Adapter(self.wav2vec_feature_dim, bottleneck=adapter_dim)
        else:
            self.adapter = None

        self.custom_head = CustomTransformerCTCHead(
            input_dim=self.wav2vec_feature_dim, 
            model_dim=trans_dim, 
            num_layers=trans_layers, 
            num_heads=trans_heads, 
            dropout=dropout, 
            vocab_size=self.vocab_size
        )

    def forward(self, input_values: torch.Tensor, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        features = outputs.last_hidden_state
        
        if self.adapter is not None:
            features = self.adapter(features)

        logits = self.custom_head(features)
        return logits

# --- Load tokenizer manually ---
def load_tokenizer(tokenizer_dir: str):
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    return tokenizer

# --- Load processor and model ---
print("Loading tokenizer and processor...")
tokenizer = load_tokenizer(TOKENIZER_DIR)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, 
    sampling_rate=SAMPLING_RATE, 
    padding_value=0.0, 
    do_normalize=True, 
    return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

print("Loading custom model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = W2V2WithCustomHead(
    pretrained_name=PRETRAINED_W2V,
    processor=processor,
    tokenizer=tokenizer,
    freeze_w2v=True,
    use_adapters=True,
    adapter_dim=128,
    trans_layers=4,
    trans_dim=768,
    trans_heads=8,
    dropout=0.1
)

# Try to load the trained model weights
try:
    # Try multiple possible file names
    possible_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]
    model_loaded = False
    
    for filename in possible_files:
        model_state_path = os.path.join(MODEL_PATH, filename)
        if os.path.exists(model_state_path):
            try:
                if filename.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_state_path)
                else:
                    state_dict = torch.load(model_state_path, map_location=device)
                
                # Remove wrapper prefixes if they exist
                new_state_dict = {}
                for k, v in state_dict.items():
                    key = k
                    for prefix in ['inner.', 'model.', 'module.']:
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                            break
                    new_state_dict[key] = v
                
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                print(f"✅ Loaded trained model weights from {filename}")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                model_loaded = True
                break
                
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue

    if not model_loaded:
        print("⚠️ No trained weights found, using base model")

except Exception as e:
    print(f"Error loading model: {e}")

model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")
print(f"Vocabulary size: {len(tokenizer.get_vocab())}")

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
    if len(audio) == 0:
        return ""
    
    # Process audio
    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Get logits from custom model
        logits = model(inputs["input_values"])
        
        # CTC decoding - get most likely tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Custom CTC decoding (remove repeats and blank tokens)
        decoded_chars = []
        prev_id = None
        
        vocab = tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        
        for token_id in predicted_ids[0].cpu().numpy():
            # Skip blank token (ID=0) and repeated tokens
            if token_id != 0 and token_id != prev_id:
                if token_id in id_to_token:
                    char = id_to_token[token_id]
                    # Skip special tokens
                    if char not in ["<unk>", "<pad>", "<blank>"]:
                        decoded_chars.append(char)
            prev_id = token_id
    
    # Join characters and clean up
    decoded_text = ''.join(decoded_chars)
    # Replace word delimiter with space
    decoded_text = decoded_text.replace("|", " ")
    
    return decoded_text.strip()

if __name__ == "__main__":
    print("🟢 Nepali Real-Time Speech Recognition with Custom Model (Ctrl+C to stop)")
    try:
        while True:
            audio = record_until_silence()
            if len(audio) > 0:
                text = transcribe(audio)
                if text:
                    print("🗣️ You said:", text)
                else:
                    print("🔇 No transcription produced")
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")