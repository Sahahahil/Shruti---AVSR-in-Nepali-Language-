# Facebook Data Collector (Nepali AVSR dataset)

This repository contains tools to download Facebook videos, extract audio and lip-region frames, and align audio with text using Whisper word timestamps.

Main script:
- [file5.py](file5.py) — entrypoint for dataset collection and alignment. Key functions: [`get_whisper_model`](file5.py), [`download_fb_video`](file5.py), [`extract_audio`](file5.py), [`extract_lip_frames`](file5.py), [`align_audio_text`](file5.py), [`process_sample`](file5.py).

Legacy notebooks:
- [file1.ipynb](file1.ipynb)
- [file2.ipynb](file2.ipynb)

Quickstart (Linux / macOS)
1. Install system packages (FFmpeg required):
```sh
# Debian/Ubuntu
sudo apt update
sudo apt install -y ffmpeg
```

2. Install Python packages:
```sh
pip install yt_dlp opencv-python mediapipe aeneas tqdm
```

3. Download and process videos:
```sh
python file5.py --fb-link https://www.facebook.com/... --title "..."
```