import torch
import torch.nn as nn

class VideoModel(nn.Module):
    def __init__(self, input_dim=468*3, hidden_dim=256, output_dim=256):
        super(VideoModel, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        return torch.randn(batch_size, seq_len, self.output_dim)
