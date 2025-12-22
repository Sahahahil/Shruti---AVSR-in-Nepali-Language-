import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= CONFIG =================
TSV_PATH = "/home/archy_sahil/MajorProject/College_dataset/Dataset_College_Server/dataset-dec-5-a/07-Dec.tsv"
FRAMES_PATH = "/home/archy_sahil/MajorProject/College_dataset/Dataset_College_Server/dataset-dec-5-a/frames"

SEQUENCE_LENGTH = 40
IMAGE_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 30
LR = 3e-4
EMBED_DIM = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD DATA =================
df = pd.read_csv(TSV_PATH, sep="\t", names=["sample_id", "label"])
print(f"Total samples: {len(df)}")

# ================= DATASET =================
class TripletVideoDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.labels = df["label"].values
        self.label_to_indices = {}

        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.indices = list(range(len(self.df)))

    def _load_video(self, sample_id):
        frames_dir = os.path.join(FRAMES_PATH, str(sample_id))
        frame_files = sorted(os.listdir(frames_dir))[:SEQUENCE_LENGTH]

        frames = []
        for f in frame_files:
            img = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img / 255.0
            frames.append(img)

        frames = np.stack(frames)
        frames = torch.tensor(frames).unsqueeze(1)  # (T,1,H,W)
        return frames.float()

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]
        anchor_video = self._load_video(anchor_row.sample_id)
        anchor_label = anchor_row.label

        # POSITIVE: same sample (since only one exists)
        positive_video = anchor_video.clone()

        # NEGATIVE: different label
        neg_label = random.choice(
            [l for l in self.label_to_indices.keys() if l != anchor_label]
        )
        neg_idx = random.choice(self.label_to_indices[neg_label])
        neg_row = self.df.iloc[neg_idx]
        negative_video = self._load_video(neg_row.sample_id)

        return anchor_video, positive_video, negative_video

    def __len__(self):
        return len(self.df)

dataset = TripletVideoDataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
class CNNLSTMEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.lstm = nn.LSTM(
            input_size=128 * 8 * 8,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(256, EMBED_DIM)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return nn.functional.normalize(x, dim=1)

model = CNNLSTMEmbedder().to(device)
print("Model ready")

# ================= TRAINING =================
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("\nStarting Triplet Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        a_emb = model(anchor)
        p_emb = model(positive)
        n_emb = model(negative)

        loss = criterion(a_emb, p_emb, n_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Triplet Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "lip_embedding_triplet.pth")
print("\nEmbedding model saved ✔")
