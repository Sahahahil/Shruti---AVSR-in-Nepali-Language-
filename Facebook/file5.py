#!/usr/bin/env python3
"""
Nepali Lip-Reading Dataset Collection Script
Uses Whisper with word timestamps (supports Nepali natively)
Run as: python datacollection.py

REQUIREMENTS:
pip install yt-dlp opencv-python mediapipe faster-whisper torch tqdm
"""

import os
import json
import subprocess
import cv2
import gc
import time
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
from faster_whisper import WhisperModel
import torch

# Force MediaPipe to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU for Whisper
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # CPU for MediaPipe

# ---------- CONFIG ----------
DATA_ROOT = "dataset-oct-21-b"
os.makedirs(DATA_ROOT, exist_ok=True)

# Detect device for Whisper
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*70}")
print(f"DEVICE CONFIGURATION")
print(f"{'='*70}")
print(f"Whisper Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"MediaPipe: CPU (forced)")
print(f"{'='*70}\n")

# ---------- WHISPER MODEL ----------
_whisper_model = None

def get_whisper_model():
    """Initialize Whisper model"""
    global _whisper_model
    if _whisper_model is None:
        print("🔄 Loading Whisper model...")
        try:
            if DEVICE == "cuda":
                _whisper_model = WhisperModel(
                    "large-v3",
                    device="cuda",
                    compute_type="float16",
                    num_workers=1,
                )
                print(f"✅ Whisper 'large-v3' loaded on GPU")
            else:
                _whisper_model = WhisperModel(
                    "medium",
                    device="cpu",
                    compute_type="int8",
                    num_workers=4,
                )
                print(f"✅ Whisper 'medium' loaded on CPU")
        except Exception as e:
            print(f"⚠️ Failed to load on {DEVICE}: {e}")
            print("🔄 Falling back to CPU...")
            _whisper_model = WhisperModel(
                "medium",
                device="cpu",
                compute_type="int8",
                num_workers=4,
            )
            print(f"✅ Whisper 'medium' loaded on CPU")
    
    return _whisper_model

# ---------- DOWNLOAD ----------
def download_fb_video(url, out_dir):
    """Download Facebook video using yt-dlp"""
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "best[ext=mp4]",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        return os.path.join(out_dir, f"{info['id']}.mp4"), info["id"]
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

# ---------- AUDIO EXTRACTION ----------
def extract_audio(video_path, out_dir, start_time=0, end_time=None):
    """Extract audio segment using FFmpeg with normalization"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}.wav")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-filter:a", "loudnorm",
        "-loglevel", "error"
    ]
    if start_time > 0:
        cmd += ["-ss", str(start_time)]
    if end_time:
        cmd += ["-to", str(end_time)]
    cmd += [out_path]

    result = subprocess.run(cmd, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f"FFmpeg failed: {result.stderr.decode()}")
    
    return out_path

# ---------- AUDIO VALIDATION ----------
def validate_audio(audio_path, min_duration=0.5):
    """Validate audio file"""
    import wave
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / sample_rate
            
            print(f"  📊 Audio: {duration:.2f}s @ {sample_rate}Hz")
            
            return {
                "valid": duration >= min_duration,
                "duration": duration,
                "sample_rate": sample_rate
            }
    except Exception as e:
        print(f"  ❌ Audio validation error: {e}")
        return {"valid": False, "duration": 0}

# ---------- LIP FRAMES EXTRACTION (CPU ONLY) ----------
def extract_lip_frames(video_path, out_dir, start_time=0, end_time=None, verbose=True):
    """Extract lip-region frames using MediaPipe (CPU)"""
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if verbose:
        print(f"  🎹 Video: {fps:.2f}fps | {total_frames} frames | {duration:.2f}s")

    if end_time is None or end_time > duration:
        end_time = duration

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    frame_idx = 0
    detected = 0

    # MediaPipe on CPU
    face_det = None
    try:
        face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3
        )
        
        while True:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end_time:
                break
            
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960 / w
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_det.process(rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    # Focus on lower face (lips)
                    y1 = y1 + int(bh * 0.4)
                    bh = int(bh * 0.6)
                    x1 = max(x1 - 10, 0)
                    y1 = max(y1 - 10, 0)
                    x2 = min(x1 + bw + 20, w)
                    y2 = min(y1 + bh + 20, h)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.jpg"), crop)
                        detected += 1
            
            frame_idx += 1
            
            # Periodic cleanup
            if frame_idx % 200 == 0:
                gc.collect()
    
    finally:
        if face_det is not None:
            face_det.close()
        cap.release()
        gc.collect()

    if verbose:
        print(f"  🎬 Extracted {detected}/{frame_idx} frames")
        if detected == 0:
            print(f"     ⚠️ WARNING: No faces detected!")

    return detected > 0

# ---------- TEXT PREPROCESSING ----------
def preprocess_nepali_text(text):
    """Clean and normalize Nepali text"""
    import re
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def match_words_fuzzy(transcribed_words, ground_truth_text):
    """Match transcribed words to ground truth using fuzzy matching"""
    from difflib import SequenceMatcher
    
    gt_words = ground_truth_text.split()
    matched_words = []
    
    for tw in transcribed_words:
        best_match = None
        best_ratio = 0.0
        
        for gt_word in gt_words:
            ratio = SequenceMatcher(None, tw['word'].strip(), gt_word).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = gt_word
        
        # If match is good enough, use ground truth word
        if best_ratio > 0.6:
            matched_words.append({
                'start': tw['start'],
                'end': tw['end'],
                'word': best_match,
                'original': tw['word'],
                'confidence': best_ratio
            })
        else:
            # Use transcribed word if no good match
            matched_words.append({
                'start': tw['start'],
                'end': tw['end'],
                'word': tw['word'].strip(),
                'original': tw['word'],
                'confidence': 0.0
            })
    
    return matched_words

# ---------- ALIGNMENT (WHISPER + MATCHING) ----------
def align_audio_text(audio_path, text, out_path, debug=True):
    """Align audio with text using Whisper word timestamps"""
    try:
        print(f"\n  🔍 Processing: {os.path.basename(audio_path)}")
        
        # Validate audio
        audio_info = validate_audio(audio_path, min_duration=0.5)
        if not audio_info["valid"]:
            print(f"  ⚠️ Audio too short or invalid")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# Audio too short or invalid\n")
            return False
        
        # Preprocess text
        ground_truth = preprocess_nepali_text(text)
        print(f"  📝 Ground truth: {ground_truth[:80]}...")
        
        # Get Whisper model
        whisper_model = get_whisper_model()
        
        # Clear GPU cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        print(f"  🎤 Transcribing with word timestamps...")
        start_time = time.time()
        
        # Transcribe with word timestamps
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=7,
            word_timestamps=True,
            language="ne",  # Nepali
            initial_prompt=ground_truth,  # Guide with ground truth
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.4,
                min_speech_duration_ms=200,
                min_silence_duration_ms=200
            ),
            condition_on_previous_text=False,
            temperature=0.0
        )
        
        # Convert to list
        segments_list = list(segments)
        elapsed = time.time() - start_time
        
        print(f"  ⏱️ Transcription completed in {elapsed:.1f}s")
        
        if debug:
            print(f"  📋 Language: {info.language} (confidence: {info.language_probability:.2%})")
            print(f"     Segments: {len(segments_list)}")
        
        if not segments_list:
            print(f"  ⚠️ No speech detected")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# No speech detected\n")
            return False
        
        # Extract all words with timestamps
        all_words = []
        for seg in segments_list:
            if seg.words:
                for w in seg.words:
                    if w.word.strip():
                        all_words.append({
                            'start': w.start,
                            'end': w.end,
                            'word': w.word.strip()
                        })
        
        if not all_words:
            print(f"  ⚠️ No words found in transcription")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# No words in transcription\n")
            return False
        
        print(f"  🔍 Transcribed {len(all_words)} words")
        if debug:
            transcribed_text = " ".join([w['word'] for w in all_words[:10]])
            print(f"     Sample: {transcribed_text}...")
        
        # Match with ground truth
        print(f"  🎯 Matching with ground truth...")
        matched_words = match_words_fuzzy(all_words, ground_truth)
        
        # Write TSV
        print(f"  💾 Writing alignment file...")
        word_count = 0
        high_conf_count = 0
        
        with open(out_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("# start\tend\tword\n")
            
            for w in matched_words:
                f.write(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['word']}\n")
                word_count += 1
                if w['confidence'] > 0.6:
                    high_conf_count += 1
        
        # Show sample
        if debug and word_count > 0:
            print(f"  📄 Sample output:")
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if not l.startswith('#')][:5]
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        print(f"       {parts[0]}-{parts[1]}s: {parts[2]}")
        
        # Cleanup
        del segments_list
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if word_count > 0:
            match_rate = (high_conf_count / word_count * 100) if word_count > 0 else 0
            print(f"  ✅ Aligned {word_count} words ({high_conf_count} high confidence, {match_rate:.1f}%)")
            print(f"     File: {out_path}")
            return True
        else:
            print(f"  ⚠️ No words aligned")
            return False
            
    except Exception as e:
        print(f"  ❌ Alignment error: {e}")
        import traceback
        traceback.print_exc()
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Error: {str(e)}\n")
        
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return False

# ---------- PROCESS ONE SAMPLE ----------
def process_sample(url, title, start_time, end_time):
    """Process a single video sample"""
    vid_id = None
    try:
        print(f"\n{'='*70}")
        print(f"📥 PROCESSING VIDEO")
        print(f"{'='*70}")
        print(f"URL: {url}")
        print(f"Title: {title[:80]}...")
        print(f"Time Range: {start_time}s - {end_time}s")
        
        # 1. Download video
        print(f"\n📥 Downloading...")
        video_path, vid_id = download_fb_video(url, os.path.join(DATA_ROOT, "videos"))
        print(f"✅ Downloaded: {vid_id}")
        
        # 2. Extract audio
        print(f"\n📊 Extracting audio...")
        audio_path = extract_audio(
            video_path,
            os.path.join(DATA_ROOT, "audio"),
            start_time,
            end_time
        )
        print(f"✅ Audio saved: {os.path.basename(audio_path)}")
        
        # 3. Extract lip frames (CPU - MediaPipe)
        print(f"\n🎬 Extracting lip frames (MediaPipe on CPU)...")
        frame_dir = os.path.join(DATA_ROOT, "frames", vid_id)
        frames_ok = extract_lip_frames(video_path, frame_dir, start_time, end_time)
        
        # Force cleanup before GPU work
        gc.collect()
        
        # 4. Align audio with text (GPU - Whisper)
        print(f"\n🎯 Aligning audio with text (Whisper on {DEVICE.upper()})...")
        align_path = os.path.join(DATA_ROOT, "alignments", f"{vid_id}.tsv")
        os.makedirs(os.path.dirname(align_path), exist_ok=True)
        align_success = align_audio_text(audio_path, title, align_path, debug=True)
        
        # 5. Verify output
        if os.path.exists(align_path):
            file_size = os.path.getsize(align_path)
            with open(align_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f if not line.startswith('#'))
            print(f"\n  📄 TSV File: {file_size} bytes | {line_count} lines")
        
        # 6. Save metadata
        meta = {
            "id": vid_id,
            "url": url,
            "text": title,
            "video": video_path,
            "audio": audio_path,
            "frames_dir": frame_dir,
            "alignment": align_path,
            "start_time": start_time,
            "end_time": end_time,
            "transcription_success": align_success,
            "frames_extracted": frames_ok,
        }
        meta_path = os.path.join(DATA_ROOT, f"{vid_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Metadata: {meta_path}")
        print(f"{'='*70}\n")
        
        # Final cleanup
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return vid_id
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {vid_id or 'unknown'}")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return None

# ---------- MAIN ----------
def main():
    """Main execution"""
    
    # Video list
    videos = [
        ("https://www.facebook.com/share/v/1Z6bCPVy49/", "गृहमन्त्री ओमप्रकाश अर्यालले नेपाल प्रहरीलाई पर्याप्त श्रोत, साधन उपलब्ध नभए आगामी प्रतिनिधि सभा सदस्यको निर्वाचन गराउने कार्य प्रभावित हुनेमा सरकार शतर्क रहेको बताउनुभएको छ", 0.0, 11.0),
        ("https://www.facebook.com/share/v/19adT5wZE5/", "बी डिभिजन विजेता प्लानिङ ब्वाइज युनाइटेड बेलबारीमा जारी मेजर स्वर्गिय एन. बि. रुङमाहाङ अन्तराष्ट्रिय आमन्त्रण रनिङ गोल्डकपको उपाधि नजिक पुगेको छ", 0.0, 8.7),
        ("https://www.facebook.com/share/v/1FgSon4ZEG/", "लेडिज सर्कल विराटनगरले स्तन क्यान्सर विषयमा सचेतनामूलक कार्यक्रम सम्पन्न गरेको छ", 0.0, 5.7),
        ("https://www.facebook.com/share/v/1DjWB1asuQ/", "सरकारले केपी ओली नेतृत्वको सरकारले नियुक्त गरेका विभिन्न ११ देशका नेपाली आवासीय राजदूतहरूलाई फिर्ता बोलाएको छ", 0.0, 7.5),
        ("https://www.facebook.com/share/v/1D31nKkj8T/", "नेकपा माओवादी केन्द्रका महाधिवेशन आयोजक समिति संयोजक पुष्पकमल दाहालले फागुन २१ गतेको निवार्चनका लागि तयारीमा जुट्न कार्यकर्ता समक्ष आह्वान गर्नु भएको छ", 0.0, 11.3),
    ]
    
    print(f"\n{'#'*70}")
    print(f"# DATASET BUILD STARTED")
    print(f"# Total Videos: {len(videos)}")
    print(f"# Output Directory: {DATA_ROOT}")
    print(f"{'#'*70}\n")
    
    # Initialize Whisper model once
    print("🔄 Initializing Whisper model...")
    try:
        _ = get_whisper_model()
        print()
    except Exception as e:
        print(f"\n❌ Failed to initialize Whisper: {e}")
        return
    
    # Error logging
    error_log = os.path.join(DATA_ROOT, 'errors.log')
    successful = 0
    failed = 0
    
    # Process each video
    for idx, (url, title, start, end) in enumerate(videos, 1):
        try:
            print(f"\n{'*'*70}")
            print(f"VIDEO {idx}/{len(videos)}")
            print(f"{'*'*70}")
            
            result = process_sample(url, title, start, end)
            
            if result:
                successful += 1
                print(f"✅ SUCCESS: {result}")
            else:
                failed += 1
                print(f"⚠️ FAILED: Returned None")
                with open(error_log, 'a', encoding='utf-8') as ef:
                    ef.write(f"{url}\tReturned None\n")
            
            print(f"\n📊 Progress: {successful} successful | {failed} failed")
            
            # Cleanup between videos
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            failed += 1
            print(f"\n❌ CRITICAL ERROR: {e}")
            with open(error_log, 'a', encoding='utf-8') as ef:
                ef.write(f"{url}\t{repr(e)}\n")
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# DATASET BUILD COMPLETE")
    print(f"# Successful: {successful}/{len(videos)}")
    print(f"# Failed: {failed}/{len(videos)}")
    print(f"{'#'*70}\n")
    
    if failed > 0:
        print(f"⚠️ Check error log: {error_log}\n")

if __name__ == "__main__":
    main()