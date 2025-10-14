# Data collector from Facebook 
1. Takes in facebook post link and title/status.
2. Download videos.
3. Extracts audios.
4. Extracts lip frames from the video.
5. Aligns audio and text.
6. Processes the sample and outputs .json file.

## Requirements
### 2nd line is only for Arch-Linux, Manager youself for Windows.
```
pip install yt_dlp opencv-python mediapipe aeneas tqdm
sudo apt install ffmpeg espeak
```