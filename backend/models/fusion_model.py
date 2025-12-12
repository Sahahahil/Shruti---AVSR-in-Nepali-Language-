import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, audio_dim=256, video_dim=256, hidden_dim=512, vocab_size=100):
        super(FusionModel, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, audio_feat, video_feat):
        batch_size, seq_len, _ = audio_feat.shape
        # return dummy logits [batch, seq_len, vocab_size]
        return torch.randn(batch_size, seq_len, self.vocab_size)
