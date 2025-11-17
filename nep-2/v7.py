# %%
# ========================================
# IMPORTS
# ========================================

import os
import math
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Trainer,
    TrainingArguments,
    AutoProcessor,
    logging as hf_logging,
)
from evaluate import load

# %%
# ========================================
# LIBRARY CHECKS
# ========================================

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

print(f"Librosa available: {LIBROSA_AVAILABLE}")
print(f"Torchaudio available: {TORCHAUDIO_AVAILABLE}")

# %%
# ========================================
# CONFIGURATIONS
# ========================================

# CONFIGURATIONS
TSV_PATH = "/home/archy_sahil/Downloads/Dataset(Major)/Combined/TSV/Final-combined.tsv"
AUDIO_BASE = "/home/archy_sahil/Downloads/Dataset(Major)/Combined/Audios"
TOKENIZER_DIR = "./tokenizer_nepali-2"
# Use the model id and read from local cache
PRETRAINED_W2V = "facebook/wav2vec2-base"
CACHE_DIR = "./cache"
OUTPUT_DIR = "./wav2vec2_custom_head-2"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"TSV Path: {TSV_PATH}")
print(f"Audio Base: {AUDIO_BASE}")
print(f"HF Cache Dir: {os.path.abspath(CACHE_DIR)}")

# %%
# ========================================
# MODEL CONFIGURATION FLAGS
# ========================================

FREEZE_W2V = True
USE_ADAPTERS = True
ADAPTER_BOTTLENECK = 128
CUSTOM_TRANSFORMER_LAYERS = 4
CUSTOM_TRANSFORMER_DIM = 768
CUSTOM_TRANSFORMER_HEADS = 8
DROPOUT = 0.1
USE_SPEC_AUG = True
USE_AUDIO_AUG = True

print(f"Freeze Wav2Vec: {FREEZE_W2V}")
print(f"Use Adapters: {USE_ADAPTERS}")

# %%
# ========================================
# TRAINING HYPERPARAMETERS
# ========================================

EPOCHS = 20
LR = 5e-5
BATCH_SIZE = 1

print(f"Epochs: {EPOCHS}, LR: {LR}, Batch Size: {BATCH_SIZE}")

# %%
# ========================================
# TOKENIZER UTILITY FUNCTION
# ========================================

def load_or_create_tokenizer(tsv_path: str, tokenizer_dir: str):
    os.makedirs(tokenizer_dir, exist_ok=True)
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    
    if os.path.exists(vocab_file):
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
            
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file,
                unk_token="<unk>",
                pad_token="<pad>",
                word_delimiter_token="|"
            )
            print(f"✅ Loaded existing tokenizer from {vocab_file}")
            return tokenizer
        except Exception as e:
            print(f"Loading existing tokenizer failed: {e}")

    # Create character-level tokenizer for CTC
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    
    print(f"Available columns in TSV: {df.columns.tolist()}")
    
    # Find text column
    text_column = None
    possible_text_cols = ['text', 'transcription', 'transcript', 'sentence', 'label']
    for col in possible_text_cols:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None and len(df.columns) >= 2:
        text_column = df.columns[1]
        print(f"Using second column '{text_column}' as text data")
    
    texts = df[text_column].astype(str).tolist()
    
    # Build character-level vocabulary
    chars = set()
    for text in texts:
        chars.update(text.lower())
    
    # Ensure word delimiter is included for decoding/joining
    chars.add('|')
    
    # Create vocabulary with proper CTC blank token
    # Order matters: <blank>=0, <unk>=1, <pad>=2
    vocab_list = ['<blank>', '<unk>', '<pad>'] + sorted(list(chars))
    vocab_dict = {char: i for i, char in enumerate(vocab_list)}
    
    print(f"Created vocabulary with {len(vocab_dict)} characters")
    
    # Save vocabulary
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    # Create tokenizer with proper blank token
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    
    print(f"✅ Created new tokenizer and saved to {vocab_file}")
    return tokenizer

# %%
# ========================================
# DATA AUGMENTATION FUNCTIONS
# ========================================

def speed_perturb(wave: np.ndarray, sr: int, factors=(0.9, 1.0, 1.1)) -> np.ndarray:
    if not LIBROSA_AVAILABLE:
        return wave
    f = random.choice(factors)
    if f == 1.0:
        return wave
    return librosa.effects.time_stretch(wave, rate=f)

def pitch_shift(wave: np.ndarray, sr: int, n_steps=(-2, 0, 2)) -> np.ndarray:
    if not LIBROSA_AVAILABLE:
        return wave
    step = random.choice(n_steps)
    return librosa.effects.pitch_shift(wave, sr=sr, n_steps=step)

def add_background_noise(wave: np.ndarray, snr_db_min=5, snr_db_max=20) -> np.ndarray:
    rms = np.sqrt(np.mean(wave**2))
    if rms == 0:
        return wave
    snr_db = random.uniform(snr_db_min, snr_db_max)
    snr = 10 ** (snr_db / 20.0)
    noise_rms = rms / snr
    noise = np.random.normal(0, noise_rms, wave.shape)
    return wave + noise

def spec_augment(features: torch.Tensor, time_mask_param=30, freq_mask_param=13, num_time_masks=2, num_freq_masks=2):
    B, T, D = features.shape
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, T - t)) if T - t > 0 else 0
        features[:, t0:t0+t, :] = 0
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, D - f)) if D - f > 0 else 0
        features[:, :, f0:f0+f] = 0
    return features

print("✅ Data augmentation functions defined")

# %%
# ========================================
# ADAPTER MODULE
# ========================================

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

print("✅ Adapter module defined")

# %%
# ========================================
# CUSTOM TRANSFORMER CTC HEAD
# ========================================

class CustomTransformerCTCHead(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_layers: int, num_heads: int, dropout: float, vocab_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        
        # Project input to model dimension if needed
        if input_dim != model_dim:
            self.input_proj = nn.Linear(input_dim, model_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.output_proj(x)
        return logits

print("✅ Custom Transformer CTC Head defined")

# %%
# ========================================
# COMBINED MODEL WRAPPER
# ========================================

class W2V2WithCustomHead(nn.Module):
    def __init__(self, pretrained_name: str, processor: Wav2Vec2Processor, tokenizer: Wav2Vec2CTCTokenizer,
                 freeze_w2v: bool=True, use_adapters: bool=True, adapter_dim: int=128,
                 trans_layers: int=4, trans_dim: int=768, trans_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.get_vocab())
        # Pad token ID is 2
        self.pad_token_id = tokenizer.pad_token_id
        
        # FIX: Get blank token ID by token string, not by attribute.
        self.blank_token_id = tokenizer.convert_tokens_to_ids('<blank>') 

        # Base wav2vec model
        print(f"   Loading Wav2Vec2 from: {pretrained_name} (cache: {os.path.abspath(CACHE_DIR)})")
        # FIX: Removed local_files_only=True to allow downloading if cache is empty
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            pretrained_name,
            cache_dir=CACHE_DIR,
        )
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

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, 
            labels: Optional[torch.Tensor]=None):
        
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        features = outputs.last_hidden_state
        
        if self.adapter is not None:
            features = self.adapter(features)

        logits = self.custom_head(features)

        loss = None
        if labels is not None:
            labels_copy = labels.clone()
            labels_copy[labels_copy == -100] = self.pad_token_id

            log_probs = F.log_softmax(logits, dim=-1)
            
            input_lengths = torch.full(
                size=(log_probs.shape[0],), 
                fill_value=log_probs.shape[1], 
                dtype=torch.long, 
                device=log_probs.device
            ).clamp(min=1)
            
            target_lengths = (labels_copy != self.pad_token_id).sum(dim=-1).clamp(min=1)
            
            targets = labels_copy[labels_copy != self.pad_token_id]
            
            if targets.numel() > 0 and target_lengths.sum() > 0:
                
                ctc_loss_fn = nn.CTCLoss(
                    blank=self.blank_token_id, # Should be 0
                    zero_infinity=True, 
                    reduction='mean'
                )
                
                log_probs_t = log_probs.transpose(0, 1) # CTC expects T x B x D

                try:
                    loss = ctc_loss_fn(log_probs_t, targets, input_lengths, target_lengths)
                    
                    if torch.isnan(loss):
                        print("Warning: CTC loss NaN. Replacing with 0.0 loss (from graph).")
                        # FIX 1: Replace leaf tensor with graph-attached 0.0 loss
                        loss = logits.sum() * 0.0
                        
                except Exception as e:
                    print(f"CTC loss computation failed: {e}. Replacing with 0.0 loss (from graph).")
                    # FIX 2: Replace leaf tensor with graph-attached 0.0 loss
                    loss = logits.sum() * 0.0
            else:
                # Fallback for empty or problematic batch
                print("Warning: Empty batch. Replacing with 0.0 loss (from graph).")
                # FIX 3: Replace leaf tensor with graph-attached 0.0 loss
                loss = logits.sum() * 0.0

        return {
            'loss': loss,
            'logits': logits,
        }

print("✅ W2V2WithCustomHead model defined")

# %%
# ========================================
# DATA COLLATOR
# ========================================

@dataclass
class DataCollatorCTCWithAugment:
    processor: Wav2Vec2Processor
    tokenizer: Wav2Vec2CTCTokenizer
    sample_rate: int = SAMPLE_RATE
    padding: bool = True
    apply_spec_augment: bool = True
    apply_audio_aug: bool = True
    min_audio_length: int = 1600 # Used for reference, actual check is more lenient
    
    # Get pad token ID (2) and unk token ID (1)
    pad_token_id: int = field(init=False)
    unk_token_id: int = field(init=False)

    def __post_init__(self):
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_values = []
        attention_masks = []
        labels = []

        def _load_audio(path: str):
            if LIBROSA_AVAILABLE:
                # librosa automatically handles resampling and mono conversion
                arr, sr = librosa.load(path, sr=self.sample_rate)
                return arr, sr
            if TORCHAUDIO_AVAILABLE:
                import torchaudio
                wave, sr = torchaudio.load(path)
                # mixdown to mono
                if wave.dim() == 2 and wave.size(0) > 1:
                    wave = wave.mean(dim=0)
                else:
                    wave = wave.squeeze(0)
                wave = wave.numpy()
                
                # Resample if needed
                if sr and sr != self.sample_rate:
                    try:
                        arr_t = torch.tensor(wave, dtype=torch.float32)
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                        wave = resampler(arr_t.unsqueeze(0)).squeeze(0).numpy()
                    except Exception:
                        pass
                return wave, sr
            return None, None

        for f in features:
            try:
                audio_path = f.get("audio_path", "")
                if not audio_path or not os.path.exists(audio_path):
                    continue

                arr, sr = _load_audio(audio_path)
                if arr is None:
                    continue

                if isinstance(arr, np.ndarray) and arr.ndim > 1:
                    arr = arr.flatten()

                # Minimum length check
                if len(arr) < int(1.0 * self.sample_rate):
                    continue

                # Optional audio augmentation
                if self.apply_audio_aug:
                    if LIBROSA_AVAILABLE:
                        if random.random() < 0.3: arr = speed_perturb(arr, self.sample_rate)
                        if random.random() < 0.3: arr = pitch_shift(arr, self.sample_rate)
                    if random.random() < 0.3: arr = add_background_noise(arr)

                # Process with wav2vec2
                inputs = self.processor(
                    arr,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )
                
                input_tensor = inputs.input_values.squeeze(0)
                attn_mask = inputs.attention_mask.squeeze(0) if hasattr(inputs, 'attention_mask') else torch.ones_like(input_tensor, dtype=torch.long)
                if input_tensor.dim() == 0:
                    input_tensor = input_tensor.unsqueeze(0)
                    attn_mask = torch.ones_like(input_tensor, dtype=torch.long)
                input_values.append(input_tensor)
                attention_masks.append(attn_mask)

                # Process text
                text = str(f.get("text", "")).lower().strip()
                if not text:
                    continue

                # Character-level tokenization
                token_ids = []
                vocab = self.tokenizer.get_vocab()
                
                for char in text:
                    if char in vocab:
                        token_ids.append(vocab[char])
                    else:
                        token_ids.append(self.unk_token_id)

                if token_ids:
                    labels.append(torch.tensor(token_ids, dtype=torch.long))

            except Exception as e:
                # print(f"Error processing item: {e}")
                continue

        # Handle empty batch
        if len(input_values) == 0 or len(labels) == 0:
            dummy_audio = torch.zeros(self.sample_rate // 2)
            dummy_mask = torch.ones_like(dummy_audio, dtype=torch.long)
            # Dummy label must be padded with -100
            dummy_labels = torch.tensor([self.pad_token_id], dtype=torch.long)
            dummy_labels[dummy_labels == self.pad_token_id] = -100
            
            return {
                "input_values": dummy_audio.unsqueeze(0),
                "attention_mask": dummy_mask.unsqueeze(0),
                "labels": dummy_labels.unsqueeze(0),
            }

        # Pad sequences
        input_values = nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
        # FIX: Pad with the pad_token_id (2), then replace with -100 for CTC loss compatibility in Trainer
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)
        labels[labels == self.pad_token_id] = -100

        return {
            "input_values": input_values,
            "attention_mask": attention_masks,
            "labels": labels, # <- Padded with -100
        }

print("✅ DataCollatorCTCWithAugment defined")

# %%
# ========================================
# DATASET PREPARATION
# ========================================

def prepare_dataset(tsv_path: str, audio_base: str) -> Dataset:
    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    
    # Handle non-standard column names
    columns = df.columns.tolist()
    if len(columns) >= 2:
        df = df.rename(columns={columns[0]: 'path', columns[1]: 'text'})
    
    if 'path' not in df.columns:
        raise ValueError('TSV must have at least 2 columns: audio path and text')

    # Remove rows with NaN values in path or text columns
    df = df.dropna(subset=['path', 'text'])
    
    # Convert path and text to strings
    df['path'] = df['path'].astype(str)
    df['text'] = df['text'].astype(str)
    
    # Create full audio paths
    def get_full_path(p):
        if not p or p == 'nan':
            return None
            
        if os.path.isabs(p):
            return p
        else:
            base_path = os.path.join(audio_base, p)
            if os.path.exists(base_path):
                return base_path
            for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                if os.path.exists(base_path + ext):
                    return base_path + ext
            return None # Return None if file not found with common extensions
    
    df['audio_path'] = df['path'].apply(get_full_path)
    
    # Remove rows where audio_path is None
    df = df[df['audio_path'].notna()]
    
    # Filter valid files
    initial_count = len(df)
    df = df[df['audio_path'].apply(os.path.exists)]
    final_count = len(df)
    
    print(f"Filtered dataset: {initial_count} -> {final_count} samples")
    
    if final_count == 0:
        raise ValueError("No valid audio files found!")
    
    dataset = Dataset.from_pandas(df[['audio_path', 'text']])
    
    return dataset

print("✅ prepare_dataset function defined")

# %%
# ========================================
# HUGGINGFACE WRAPPER
# ========================================

@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

class HFWrapperModel(nn.Module):
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model
    
    def forward(self, input_values=None, labels=None, attention_mask=None, **kwargs):
        outputs = self.inner(input_values=input_values, attention_mask=attention_mask, labels=labels)
        # Return a plain dict for Trainer compatibility
        return {"loss": outputs['loss'], "logits": outputs['logits']}

print("✅ HFWrapperModel defined")

# %%
# ========================================
# METRICS
# ========================================

wer_metric = load('wer')

def compute_metrics(pred):
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    
    if logits is None:
        return {"wer": 1.0}
        
    pred_ids = np.argmax(logits, axis=-1)
    # Label IDs were padded with -100 by the DataCollator
    label_ids = pred.label_ids 
    
    pred_texts = []
    ref_texts = []
    
    # Get tokenizer from global scope (assumes it's defined, which it is in __main__)
    global tokenizer 
    
    # CTC decoding for predictions
    for i in range(pred_ids.shape[0]):
        # Decode predictions (remove repeats, blank=0, and padding)
        pred_seq = []
        prev_id = None
        for token_id in pred_ids[i]:
            # Token ID 0 is '<blank>'
            if token_id != prev_id and token_id != 0: 
                pred_seq.append(int(token_id))
            prev_id = int(token_id)
        
        if len(pred_seq) == 0:
            pred_texts.append("")
        else:
            # Use tokenizer to convert IDs to text
            pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
            pred_texts.append(pred_text)
        
        # Decode references (mask -100)
        ref_seq = [int(t) for t in label_ids[i] if t != -100] 
        
        if len(ref_seq) == 0:
            ref_texts.append("")
        else:
            ref_text = tokenizer.decode(ref_seq, skip_special_tokens=True)
            ref_texts.append(ref_text)
    
    if not pred_texts or not ref_texts:
        return {"wer": 1.0}
    
    wer_score = wer_metric.compute(predictions=pred_texts, references=ref_texts)
    return {"wer": wer_score}

print("✅ Metrics function defined")

# %%
# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 STARTING WAV2VEC2 TRAINING")
    print("="*50 + "\n")
    
    # Set seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Reduce HF verbosity a bit
    try:
        hf_logging.set_verbosity_warning()
    except Exception:
        pass
    
    # Load tokenizer
    print("📝 Loading/Creating tokenizer...")
    tokenizer = load_or_create_tokenizer(TSV_PATH, TOKENIZER_DIR)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=SAMPLE_RATE, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    print(f"✅ Tokenizer vocabulary size: {len(tokenizer.get_vocab())}")
    
    # Prepare dataset
    print("\n📂 Loading dataset...")
    dataset = prepare_dataset(TSV_PATH, AUDIO_BASE)
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"✅ Train dataset size: {len(train_dataset)}")
    print(f"✅ Eval dataset size: {len(eval_dataset)}")
    
    # Create model
    print("\n🏗️ Creating model...")
    try:
        model = W2V2WithCustomHead(
            pretrained_name=PRETRAINED_W2V,
            processor=processor,
            tokenizer=tokenizer,
            freeze_w2v=FREEZE_W2V,
            use_adapters=USE_ADAPTERS,
            adapter_dim=ADAPTER_BOTTLENECK,
            trans_layers=CUSTOM_TRANSFORMER_LAYERS,
            trans_dim=CUSTOM_TRANSFORMER_DIM,
            trans_heads=CUSTOM_TRANSFORMER_HEADS,
            dropout=DROPOUT
        )
        
        model.to(DEVICE)
        print(f"✅ Model loaded on {DEVICE}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Create data collator
    print("\n📦 Creating data collator...")
    data_collator = DataCollatorCTCWithAugment(
        processor=processor,
        tokenizer=tokenizer,
        sample_rate=SAMPLE_RATE,
        apply_spec_augment=USE_SPEC_AUG,
        apply_audio_aug=USE_AUDIO_AUG
    )
    print("✅ Data collator created")
    
    # Wrap model for HuggingFace
    wrapped_model = HFWrapperModel(model)
    print("✅ Model wrapped for HuggingFace Trainer")
    
    # Training arguments
    print("\n⚙️ Configuring training arguments...")
    
    # This block should be correct now, using 'eval_strategy'
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=500,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        
        eval_strategy="steps",
        save_strategy="steps",
        
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        group_by_length=False,
        # fp16=torch.cuda.is_available(),
        fp16=False,
        push_to_hub=False,
        report_to=None,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=SEED,
    )
    print("✅ Training arguments configured")
    print(f"   Effective batch size: {BATCH_SIZE * 4}")
    print(f"   FP16 enabled: {torch.cuda.is_available()}")
    
    # Create trainer
    print("\n👨‍🏫 Creating trainer...")
    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("✅ Trainer created")
    
    # Start training
    print("\n" + "="*50)
    print("🏋️ STARTING TRAINING")
    print("="*50 + "\n")
    
    train_result = trainer.train()
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETED")
    print("="*50 + "\n")
    
    # Final evaluation
    print("📊 Running final evaluation...")
    metrics = trainer.evaluate()
    
    print("\n" + "="*50)
    print("FINAL METRICS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    print(f"✅ Processor saved to {OUTPUT_DIR}")
    print(f"✅ Tokenizer saved to {OUTPUT_DIR}")
    
    print("\n" + "="*50)
    print("🎉 ALL DONE!")
    print("="*50)