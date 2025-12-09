import ffmpeg

def extract_audio(video_path, audio_path):
    """
    Extracts audio from video and saves as WAV.
    """
    try:
        (
            ffmpeg.input(video_path)
            .output(audio_path, format="wav", acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        return audio_path
    except ffmpeg.Error as e:
        print("Error extracting audio:", e)
        return None
