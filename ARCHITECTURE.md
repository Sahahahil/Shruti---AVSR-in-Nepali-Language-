# Shruti AVSR - Project Architecture

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (User)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │    Next.js Frontend (3000)      │
         │   ┌──────────────────────────┐  │
         │   │  Dashboard Component     │  │
         │   │  ┌────────────────────┐  │  │
         │   │  │  Tab 1: AVSR       │  │  │
         │   │  ├────────────────────┤  │  │
         │   │  │  Tab 2: VSR+ASR    │  │  │
         │   │  ├────────────────────┤  │  │
         │   │  │  Tab 3: ASR Only   │  │  │
         │   │  └────────────────────┘  │  │
         │   └──────────────────────────┘  │
         │                                 │
         │  State: Zustand Store           │
         │  HTTP: Axios Client             │
         │  Animation: Framer Motion       │
         │  Style: SASS Modules            │
         └────────────┬────────────────────┘
                      │ HTTP/REST
                      │ (Multipart Form Data)
                      ▼
         ┌──────────────────────────────────────┐
         │  FastAPI Backend (8000)              │
         │                                      │
         │  ┌────────────────────────────────┐  │
         │  │  API Endpoints                 │  │
         │  ├─  POST /api/avsr/realtime      │  │
         │  ├─  POST /api/vsr-asr/realtime   │  │
         │  └─  POST /api/asr-only/video     │  │
         │  └────────────────────────────────┘  │
         │                                      │
         │  ┌────────────────────────────────┐  │
         │  │  Inference Engine              │  │
         │  │  (inference.py)                │  │
         │  ├─  AVSR Mode Processing        │  │
         │  ├─  VSR+ASR Mode Processing     │  │
         │  └─  ASR Only Mode Processing    │  │
         │  └────────────────────────────────┘  │
         │                                      │
         │  ┌────────────────────────────────┐  │
         │  │  Models                        │  │
         │  ├─  Audio Model (wav2vec2)       │  │
         │  ├─  Video Model (CNN)            │  │
         │  └─  Fusion Model (RNN)           │  │
         │  └────────────────────────────────┘  │
         │                                      │
         │  ┌────────────────────────────────┐  │
         │  │  Utilities                     │  │
         │  ├─  Audio Extraction (FFmpeg)    │  │
         │  ├─  Frame Extraction (OpenCV)    │  │
         │  ├─  Audio Preprocessing          │  │
         │  ├─  Video Preprocessing          │  │
         │  └─  CTC Decoding                 │  │
         │  └────────────────────────────────┘  │
         └──────────────────────────────────────┘
```

## 🔄 Data Flow

### Tab 1: AVSR Mode

```
Video Upload
    │
    ▼
[Frontend] FileUpload Component
    │
    ▼
[API] POST /api/avsr/realtime (multipart/form-data)
    │
    ▼
[Backend] app.py - avsr_realtime()
    │
    ├─ Save file to uploads/
    │
    ▼
[Backend] inference.py - transcribe_video(mode="avsr")
    │
    ├─ extract_audio() → audio.wav
    ├─ extract_frames() → frames/*.jpg
    │
    ├─ preprocess_audio() → audio_tensor
    ├─ preprocess_video() → video_tensor
    │
    ├─ audio_model(audio_tensor) → audio_features
    ├─ video_model(video_tensor) → video_features
    │
    └─ fusion_model(audio_feat, video_feat) → logits
        │
        ▼
    ctc_decode(logits) → "नमस्ते"
        │
        ▼
    Return: {transcription, confidence, processing_time}
    │
    ▼
[Frontend] ResultDisplay Component
    │
    ▼
Display transcription to user
```

### Tab 2: VSR+ASR Mode

```
Video Upload
    │
    ├─ extract_audio() & extract_frames()
    │
    ├─ PARALLEL PROCESSING ──┬──────────────────┐
    │                        │                  │
    │                   VSR Path           ASR Path
    │                        │                  │
    │              video_model()         audio_model()
    │                        │                  │
    │              vsr_logits          asr_logits
    │                        │                  │
    │           ctc_decode()            ctc_decode()
    │                        │                  │
    │              vsr_result           asr_result
    │                        │                  │
    └────────────┬───────────┘──────────────────┘
                 │
                 ▼
        Return both results
                 │
                 ▼
        Display side-by-side
```

### Tab 3: ASR Only Mode

```
Video Upload
    │
    ├─ extract_audio() → audio.wav
    │
    ├─ preprocess_audio() → audio_tensor
    │
    ├─ audio_model(audio_tensor) → logits
    │
    └─ ctc_decode(logits) → "नमस्ते"
        │
        ▼
    Return: {transcription, confidence, processing_time}
        │
        ▼
    Display result
```

## 📁 Frontend Component Tree

```
App (Layout)
└── Dashboard
    ├── Tabs (Tab Navigation)
    │   ├── Tab 1 Button
    │   ├── Tab 2 Button
    │   └── Tab 3 Button
    │
    └── Tab Content (Animated)
        ├── RealtimeAVSR
        │   ├── FileUpload
        │   ├── ProgressBar
        │   └── ResultDisplay
        │       └── Copy Button
        │
        ├── RealtimeVSR_ASR
        │   ├── FileUpload
        │   ├── ProgressBar
        │   └── ResultDisplay (Dual)
        │       ├── ASR Result + Copy
        │       └── VSR Result + Copy
        │
        └── RealtimeASR_VideoInput
            ├── FileUpload
            ├── ProgressBar
            └── ResultDisplay
                └── Copy Button
```

## 🔌 API Endpoint Structure

```
app.py (FastAPI Application)
│
├── CORS Middleware
├── Error Handlers
├── Request Validators
│
└── Routes
    │
    ├── GET /health                    ← Health Check
    ├── GET /api/config                ← Configuration
    │
    ├── POST /api/avsr/realtime        ← Tab 1
    │   ├── Validate file
    │   ├── Save file
    │   ├── Call inference.transcribe_video(mode="avsr")
    │   └── Return response
    │
    ├── POST /api/vsr-asr/realtime     ← Tab 2
    │   ├── Validate file
    │   ├── Save file
    │   ├── Call inference.transcribe_video(mode="vsr_asr")
    │   └── Return response
    │
    └── POST /api/asr-only/video       ← Tab 3
        ├── Validate file
        ├── Save file
        ├── Call inference.transcribe_video(mode="asr_only")
        └── Return response
```

## 🧠 Inference Pipeline

```
inference.py
│
├── load_models()
│   ├── AudioModel
│   ├── VideoModel
│   └── FusionModel
│
├── transcribe_video(video_path, mode)
│   │
│   ├── extract_audio(video_path) → audio.wav
│   ├── extract_frames(video_path) → frames/
│   │
│   ├── preprocess_audio(audio.wav) → tensor
│   ├── preprocess_video(frames/) → tensor
│   │
│   ├── if mode == "avsr"
│   │   └── _process_avsr()
│   │       ├── audio_model() → features
│   │       ├── video_model() → features
│   │       ├── fusion_model() → logits
│   │       └── ctc_decode() → text
│   │
│   ├── elif mode == "vsr_asr"
│   │   └── _process_vsr_asr()
│   │       ├── audio_model() → asr_logits
│   │       ├── video_model() → vsr_logits
│   │       ├── ctc_decode(asr_logits)
│   │       └── ctc_decode(vsr_logits)
│   │
│   └── elif mode == "asr_only"
│       └── _process_asr_only()
│           ├── audio_model() → logits
│           └── ctc_decode() → text
│
└── cleanup_files()
    ├── Remove audio.wav
    └── Remove frames/
```

## 🗂️ File Organization

```
src/
├── app/
│   ├── layout.tsx           ← Root layout with SASS
│   └── page.tsx             ← Dashboard entry point
│
├── components/
│   ├── Dashboard.tsx        ← Tab management & layout
│   ├── RealtimeAVSR.tsx     ← Tab 1 (AVSR)
│   ├── RealtimeVSR_ASR.tsx  ← Tab 2 (Separated)
│   ├── RealtimeASR_VideoInput.tsx ← Tab 3 (ASR Only)
│   ├── FileUpload.tsx       ← Upload component
│   ├── ResultDisplay.tsx    ← Results display
│   └── ProcessingIndicator.tsx ← Loading animation
│
├── lib/
│   ├── api-client.ts        ← Axios instances & methods
│   └── store.ts             ← Zustand state store
│
└── styles/
    ├── dashboard.module.scss ← Dashboard styling
    ├── tab-content.module.scss ← Tab content styling
    ├── components.module.scss ← Components styling
    └── globals.scss         ← Global styling
```

## 🔐 Error Flow

```
User Action (Upload)
    │
    ▼
FileUpload Component
    │
    ├─ Validate file type
    │  └─ Error → Show alert
    │
    ├─ Validate file size
    │  └─ Error → Show alert
    │
    ▼
API Client (axios)
    │
    ├─ Network Error → Catch & Show message
    ├─ 400 Bad Request → Show validation error
    ├─ 500 Server Error → Show server error
    │
    ▼
Backend Processing
    │
    ├─ File processing error → Return error response
    ├─ Model inference error → Return error response
    │
    └─ Success → Return results

Frontend Error Boundary
└─ Log errors in console
└─ Show user-friendly messages
```

## 🎨 State Management Flow

```
Zustand Store (lib/store.ts)
│
├── isLoading (boolean)
├── uploadProgress (0-100)
├── error (string | null)
├── result (object | null)
└── currentTab ('avsr' | 'vsr_asr' | 'asr_only')

Components Subscribe to Store
│
├── Dashboard → currentTab
├── RealtimeAVSR → isLoading, uploadProgress, error, result
├── RealtimeVSR_ASR → isLoading, uploadProgress, error, result
├── RealtimeASR_VideoInput → isLoading, uploadProgress, error, result
└── ResultDisplay → result

Store Mutations
│
├── setLoading() → Update isLoading
├── setProgress() → Update uploadProgress
├── setError() → Update error message
├── setResult() → Update transcription result
├── setCurrentTab() → Update active tab
└── reset() → Clear all state
```

## 🚀 Deployment Architecture

```
Production Deployment
│
├── Backend (Docker Container)
│   ├── FastAPI Server on :8000
│   ├── Model files mounted
│   └── GPU support (optional)
│
├── Frontend (Next.js on Vercel/Docker)
│   ├── Deployed on :3000
│   └── CDN for assets
│
└── Networking
    ├── Both services on same network
    └── Frontend env var points to Backend URL
```

---

This architecture ensures:
✅ Clean separation of concerns
✅ Reusable components
✅ Scalable backend
✅ Responsive frontend
✅ Efficient data flow
✅ Good error handling
