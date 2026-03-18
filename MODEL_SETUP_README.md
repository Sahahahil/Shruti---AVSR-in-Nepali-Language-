# Model Setup Guide

This project expects large model files to exist locally and not be committed to git.

## 1) Where to place models

Place your model files in the following folders:

- VSR model file:
  - `backend/video_model/best_model.pth`

- ASR wav2vec2 checkpoint folder:
  - `backend/audio_model/asr_weights/wav2vec2-weights/checkpoint-9200/`

- ASR processor folder:
  - `backend/audio_model/asr_weights/wav2vec2-weights/wav2vec2-nepali-processor-20260315T054231Z-1-001/wav2vec2-nepali-processor/`

Keep the folder names exactly as above unless you also update the code paths.

## 2) Where to edit model paths

If your models are in different locations, edit these files:

- `backend/video_model/infer_march.py`
  - In class `Config`, update:
    - `VSR_CHECKPOINT`
    - `WAV2VEC_PATH`
    - `WAV2VEC_PROCESSOR_PATH`

- `backend/inference.py`
  - If you use custom weight loading logic, update paths inside `load_models()`.

## 3) Quick verification

From `backend/` run:

```bash
python3 -c "import app; print('Backend import OK')"
```

If imports succeed, the basic dependency/path setup is valid.

## 4) Notes for contributors

- Model weights and checkpoints are intentionally ignored in git.
- Do not commit `.pth`, checkpoint folders, or generated cache/output data.
- Use local absolute or relative paths in your own environment as needed.