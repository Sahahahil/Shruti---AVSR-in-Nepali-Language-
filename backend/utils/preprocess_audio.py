import torch
import torchaudio

def preprocess_audio(audio_path, target_sr=16000):
    """
    Loads audio file, resamples if needed, and converts to tensor.
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    # Wav2Vec2 expects shape [batch, time], so remove channel dim if mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze(0)  # [time]
