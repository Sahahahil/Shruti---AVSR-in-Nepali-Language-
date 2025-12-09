import torch
import torch.nn as nn

class AudioModel(nn.Module):
    def __init__(self, wav2vec_path=None, output_dim=256):
        super(AudioModel, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        # x: [batch, time]
        batch_size, seq_len = x.shape[0], x.shape[1] if x.dim() > 1 else x.shape[0]
        # return dummy features [batch, seq_len, output_dim]
        return torch.randn(batch_size, seq_len, self.output_dim)
