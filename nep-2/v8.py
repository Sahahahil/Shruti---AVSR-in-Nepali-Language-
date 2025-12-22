# %% [markdown]
# # Imports

# %%
import os
import math
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# %%
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    logging as hf_logging,
)
from evaluate import load

# %% [markdown]
# # Library Checks

# %%
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

# %% [markdown]
# # Configurations

# %%
TSV_PATH = "/home/archy_sahil/MajorProject/College_dataset/Dataset_College_Server/dataset-dec-5-a/07-Dec.tsv"
AUDIO_BASE = "/home/archy_sahil/MajorProject/College_dataset/Dataset_College_Server/dataset-dec-5-a/audio"
TOKENIZER_DIR = "./tokenizer_nepali-3"
PRETRAINED_W2V = "/home/archy_sahil/MajorProject/Models & Processors/wav2vec2-nepali-finetuned-v2"
CACHE_DIR = "./cache"
OUTPUT_DIR = "./wav2vec2_custom_head-2-5Dec"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"TSV Path: {TSV_PATH}")
print(f"Audio Base: {AUDIO_BASE}")
print(f"HF Cache Dir: {os.path.abspath(CACHE_DIR)}")

# %% [markdown]
# # Model Config Flags

# %%
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

# %% [markdown]
# # Training Hyperparameters

# %%
EPOCHS = 5
# Lowered learning rate for stability
LR = 3e-5 
BATCH_SIZE = 1

print(f"Epochs: {EPOCHS}, LR: {LR}, Batch Size: {BATCH_SIZE}")

# %% [markdown]
# # Metrics Tracking Callback

# %%
class MetricsCallback(TrainerCallback):
    """Custom callback to track training and validation metrics"""
    
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.wer_scores = []
        self.cer_scores = [] # New: Track Character Error Rate
        self.steps = []
        self.eval_steps = []
        self.trainer = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens"""
        if logs is not None:
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
                self.steps.append(state.global_step)
            
            if 'eval_loss' in logs:
                self.validation_loss.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
            
            if 'eval_wer' in logs:
                self.wer_scores.append(logs['eval_wer'])
            
            if 'eval_cer' in logs: # New: Track CER
                self.cer_scores.append(logs['eval_cer'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        pass

    def plot_metrics(self, save_path=None):
        """Plot all metrics"""
        try:
            # Create two plots: Loss and Error Rates (WER/CER)
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Loss
            if self.training_loss and self.validation_loss:
                axes[0].plot(self.steps, self.training_loss,
                            label='Training Loss', linewidth=2, alpha=0.7, color='blue')
                axes[0].plot(self.eval_steps, self.validation_loss,
                            marker='o', label='Validation Loss', linewidth=2, color='orange')
                axes[0].set_xlabel('Steps')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training and Validation Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot 2: WER and CER
            if self.wer_scores:
                axes[1].plot(self.eval_steps, self.wer_scores, 
                            marker='o', color='red', label='WER', linewidth=2)
            if self.cer_scores:
                axes[1].plot(self.eval_steps, self.cer_scores, 
                            marker='x', color='green', label='CER', linewidth=2, linestyle='--')

            if self.wer_scores or self.cer_scores:
                axes[1].set_xlabel('Steps')
                axes[1].set_ylabel('Error Rate (Lower is Better)')
                axes[1].set_title('Word Error Rate (WER) and Character Error Rate (CER)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Metrics plot saved to {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def print_summary(self):
        print("\n" + "="*50)
        print("TRAINING METRICS SUMMARY")
        print("="*50)
        if self.validation_loss:
            print(f"Final Val Loss: {self.validation_loss[-1]:.4f}")
        if self.wer_scores:
            print(f"Final WER: {self.wer_scores[-1]:.4f}")
        if self.cer_scores:
            print(f"Final CER: {self.cer_scores[-1]:.4f}")
        print("="*50 + "\n")

# %% [markdown]
# # Tokenizer Utility Function

# %%
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

    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    
    text_column = None
    possible_text_cols = ['text', 'transcription', 'transcript', 'sentence', 'label']
    for col in possible_text_cols:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None and len(df.columns) >= 2:
        text_column = df.columns[1]
    
    texts = df[text_column].astype(str).tolist()
    
    chars = set()
    for text in texts:
        chars.update(text.lower())
    
    chars.add('|')
    
    vocab_list = ['<blank>', '<unk>', '<pad>'] + sorted(list(chars))
    vocab_dict = {char: i for i, char in enumerate(vocab_list)}
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    
    print(f"✅ Created new tokenizer and saved to {vocab_file}")
    return tokenizer

# %% [markdown]
# # Data Augmentation Functions

# %%
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

# %% [markdown]
# # Adapter Module

# %%
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.relu = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)
        return x + residual

# %% [markdown]
# # Custom Transformer CTC Head

# %%
class CustomTransformerCTCHead(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_layers: int, 
                 num_heads: int, dropout: float, vocab_size: int):
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

        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent exploding gradients"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, features: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(features)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_proj(x)
        return logits

# %% [markdown]
# # Combined Model Wrapper

# %%
class W2V2WithCustomHead(nn.Module):
    def __init__(self, pretrained_name: str, processor: Wav2Vec2Processor, 
                 tokenizer: Wav2Vec2CTCTokenizer, freeze_w2v: bool=True, 
                 use_adapters: bool=True, adapter_dim: int=128,
                 trans_layers: int=4, trans_dim: int=768, trans_heads: int=8, 
                 dropout: float=0.1):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id
        
        print(f"   Loading Wav2Vec2 from: {pretrained_name}")
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
        
        # 1. Forward Wav2Vec2
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        features = outputs.last_hidden_state
        
        # 2. Compute padding mask for Transformer
        transformer_mask = None
        if attention_mask is not None:
            batch_size, input_len = input_values.shape
            output_len = features.shape[1]
            reduced_mask_len = (attention_mask.sum(dim=1) / input_len * output_len).long()
            transformer_mask = torch.zeros((batch_size, output_len), dtype=torch.bool, device=features.device)
            for i in range(batch_size):
                valid_len = reduced_mask_len[i].item()
                if valid_len < output_len:
                    transformer_mask[i, valid_len:] = True

        # 3. Adapters
        if self.adapter is not None:
            features = self.adapter(features)

        # 4. Custom Head
        logits = self.custom_head(features, src_key_padding_mask=transformer_mask)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Ensure model output is float32 for stable log_softmax
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_probs_t = log_probs.transpose(0, 1).contiguous()
            
            # Input lengths (Time steps)
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)
            if transformer_mask is not None:
                input_lengths = (~transformer_mask).sum(dim=1)

            # Target lengths
            pad_id = self.pad_token_id
            labels_mask = (labels != -100) & (labels != pad_id)
            target_lengths = labels_mask.sum(dim=-1)
            targets = labels[labels_mask]

            # FIX: CLIP TARGET LENGTHS to ensure target length <= input length (CTC requirement)
            target_lengths = torch.clamp(target_lengths, max=input_lengths)
            
            ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
            loss = ctc_loss(log_probs_t, targets, input_lengths, target_lengths)

        return {
            'loss': loss,
            'logits': logits,
        }

# %% [markdown]
# # Data Collator

# %%
@dataclass
class DataCollatorCTCWithAugment:
    processor: Wav2Vec2Processor
    tokenizer: Wav2Vec2CTCTokenizer
    sample_rate: int = SAMPLE_RATE
    padding: bool = True
    apply_spec_augment: bool = True
    apply_audio_aug: bool = True
    # Minimal length restriction to allow "whatever data"
    min_audio_length: int = 10 
    
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
                try:
                    arr, sr = librosa.load(path, sr=self.sample_rate)
                    return arr, sr
                except:
                    return None, None
            return None, None

        for f in features:
            try:
                audio_path = f.get("audio_path", "")
                text = str(f.get("text", "")).lower().strip()
                if not audio_path or not text: continue

                arr, sr = _load_audio(audio_path)
                if arr is None: continue
                if len(arr) < self.min_audio_length: continue 

                if self.apply_audio_aug and random.random() < 0.3:
                     arr = speed_perturb(arr, self.sample_rate)

                inputs = self.processor(arr, sampling_rate=self.sample_rate, return_tensors="pt")
                input_tensor = inputs.input_values.squeeze(0)
                
                token_ids = []
                vocab = self.tokenizer.get_vocab()
                for char in text:
                    token_ids.append(vocab.get(char, self.unk_token_id))

                if not token_ids: continue

                input_values.append(input_tensor)
                attention_masks.append(inputs.attention_mask.squeeze(0) if hasattr(inputs, 'attention_mask') else torch.ones_like(input_tensor, dtype=torch.long))
                labels.append(torch.tensor(token_ids, dtype=torch.long))

            except Exception:
                continue

        if len(input_values) == 0:
            dummy = torch.zeros(self.sample_rate)
            return {
                "input_values": dummy.unsqueeze(0),
                "attention_mask": torch.ones_like(dummy, dtype=torch.long).unsqueeze(0),
                "labels": torch.tensor([-100]).unsqueeze(0)
            }

        input_values = nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)
        labels[labels == self.pad_token_id] = -100

        return {
            "input_values": input_values,
            "attention_mask": attention_masks,
            "labels": labels,
        }

# %% [markdown]
# # Dataset Preparation

# %%
def prepare_dataset(tsv_path: str, audio_base: str) -> Dataset:
    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    
    cols = df.columns.tolist()
    if len(cols) >= 2: df = df.rename(columns={cols[0]: 'path', cols[1]: 'text'})
    
    df = df.dropna(subset=['path', 'text'])
    df['path'] = df['path'].astype(str)
    df['text'] = df['text'].astype(str)
    
    def get_full_path(p):
        if os.path.isabs(p): return p
        full = os.path.join(audio_base, p)
        if os.path.exists(full): return full
        if os.path.exists(full + '.wav'): return full + '.wav'
        return None
    
    df['audio_path'] = df['path'].apply(get_full_path)
    df = df[df['audio_path'].notna()]
    
    print(f"✅ Final dataset size: {len(df)} samples")
    dataset = Dataset.from_pandas(df[['audio_path', 'text']])
    return dataset

# %% [markdown]
# # HF Wrapper

# %%
class HFWrapperModel(nn.Module):
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model
    
    def forward(self, input_values=None, labels=None, attention_mask=None, **kwargs):
        outputs = self.inner(input_values=input_values, attention_mask=attention_mask, labels=labels)
        return {"loss": outputs['loss'], "logits": outputs['logits']}

# %% [markdown]
# # Metrics

# %%
wer_metric = load('wer')
# Load CER metric
cer_metric = load('cer')

def compute_metrics(pred):
    logits = pred.predictions
    if isinstance(logits, tuple): logits = logits[0]
    
    # Decode
    pred_ids = np.argmax(logits, axis=-1)
    label_ids = pred.label_ids 
    
    # Replace -100 with pad token for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(label_ids, group_tokens=False)
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Compute CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    print("\n🚀 STARTING TRAINING")
    
    # Tokenizer
    tokenizer = load_or_create_tokenizer(TSV_PATH, TOKENIZER_DIR)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Dataset
    dataset = prepare_dataset(TSV_PATH, AUDIO_BASE)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Model
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
    # Reduce peak memory: enable gradient checkpointing on wav2vec backbone if available
    try:
        if hasattr(model, 'wav2vec') and hasattr(model.wav2vec, 'gradient_checkpointing_enable'):
            model.wav2vec.gradient_checkpointing_enable()
            print("✅ Enabled gradient checkpointing on wav2vec backbone (reduces memory, increases compute).")
    except Exception:
        pass

    # Clear CUDA cache after moving model to device to reduce fragmentation
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    # Collator
    data_collator = DataCollatorCTCWithAugment(
        processor=processor,
        tokenizer=tokenizer,
        sample_rate=SAMPLE_RATE,
        apply_spec_augment=USE_SPEC_AUG,
        apply_audio_aug=USE_AUDIO_AUG
    )
    
    # Initialize the callback
    metrics_callback = MetricsCallback()

    # Trainer
    trainer = Trainer(
        model=HFWrapperModel(model),
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            # Lower accumulation reduces peak memory on low-memory GPUs
            gradient_accumulation_steps=1,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            # Use mixed precision to reduce memory footprint. If you encounter instability, set to False.
            fp16=True,
            save_total_limit=2,
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            eval_strategy="steps",
            report_to=None,
            # CRITICAL FIX: Retains audio_path and text columns for DataCollator
            remove_unused_columns=False,
            # Added for gradient stability
            max_grad_norm=0.5,
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )
    
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
    # Save processor and tokenizer as well
    processor.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Model, Processor, and Tokenizer saved to {OUTPUT_DIR}")
    
    # Print metrics summary and generate plots
    print("\n📊 Generating training metrics visualization...")
    metrics_callback.print_summary()
    metrics_callback.plot_metrics(save_path=os.path.join(OUTPUT_DIR, "training_metrics.png"))
    
    print("\n" + "="*50)
    print("ALL DONE.")
    print("="*50)