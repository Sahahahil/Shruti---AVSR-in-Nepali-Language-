# Quick Start Guide

## 🚀 Fastest Way to Run the Project

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start both backend and frontend
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup (Development)

#### Terminal 1: Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2: Start Frontend
```bash
cd frontend/nepali-avsr-frontend
npm install
npm run dev
```

Then open `http://localhost:3000` in your browser.

## 📖 Understanding the Application

### What You'll See

1. **Main Dashboard** with 3 tabs at the top
2. **Upload area** on the left side
3. **Results area** on the right side

### How to Use Each Tab

#### Tab 1: Audio-Visual Speech Recognition (AVSR)
1. Click on Tab 1
2. Drag and drop or click to upload a video
3. Wait for processing
4. See the combined transcription result

#### Tab 2: Visual + Audio (Separated)
1. Click on Tab 2
2. Upload a video
3. Wait for processing
4. See both VSR and ASR results side by side
5. Compare confidence scores

#### Tab 3: Audio Only (Video Input)
1. Click on Tab 3
2. Upload a video
3. Backend extracts audio automatically
4. See the audio-only transcription result

## 🔧 Configuration

### Change Backend URL (if running on different server)

Edit `frontend/nepali-avsr-frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://your-backend-server:8000
```

### Supported Video Formats
- MP4, WebM, MOV, AVI
- Max file size: 100MB

## 📊 Testing the API Directly

### Using cURL

```bash
# Tab 1: AVSR
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/avsr/realtime

# Tab 2: VSR+ASR
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/vsr-asr/realtime

# Tab 3: ASR Only
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/asr-only/video

# Health Check
curl http://localhost:8000/health

# Config
curl http://localhost:8000/api/config
```

### Using Postman
1. Create new POST request
2. URL: `http://localhost:8000/api/avsr/realtime`
3. Body → form-data → file (type: File)
4. Select your video file
5. Send

## 🛠️ Troubleshooting

### Issue: "Cannot connect to backend"
- ✅ Check backend is running on port 8000
- ✅ Check `.env.local` has correct API URL
- ✅ Check firewall isn't blocking port 8000

### Issue: "File upload fails"
- ✅ Check file size (max 100MB)
- ✅ Check file format (mp4, webm, mov, avi)
- ✅ Check server has write permissions for uploads folder

### Issue: "Processing takes very long"
- ✅ Large files take longer to process
- ✅ First run might be slower (model loading)
- ✅ Close other applications to free up resources

## 📚 Key Directories

```
frontend/nepali-avsr-frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.tsx          ← Main tab container
│   │   ├── RealtimeAVSR.tsx      ← Tab 1 logic
│   │   ├── RealtimeVSR_ASR.tsx   ← Tab 2 logic
│   │   └── RealtimeASR_VideoInput.tsx ← Tab 3 logic
│   ├── lib/
│   │   ├── api-client.ts          ← API calls
│   │   └── store.ts               ← State management
│   └── styles/
│       ├── dashboard.module.scss  ← Dashboard styles
│       ├── tab-content.module.scss ← Tabs styles
│       └── components.module.scss ← Components styles

backend/
├── app.py                          ← 3 main endpoints
├── inference.py                    ← Processing logic (3 modes)
├── models/                         ← ML models
│   ├── audio_model.py
│   ├── video_model.py
│   └── fusion_model.py
└── utils/                          ← Helper functions
    ├── extract_audio.py
    ├── extract_frames.py
    ├── preprocess_audio.py
    ├── preprocess_video.py
    └── decode_ctc.py
```

## 🎯 Next Steps

1. **Add your models**: Replace placeholder models in `backend/models/`
2. **Train models**: Use training notebooks in project root
3. **Customize UI**: Edit SASS files in `frontend/nepali-avsr-frontend/src/styles/`
4. **Add more features**: Follow component structure to add new features

## 📞 Help

For detailed setup instructions, see `SETUP_AND_RUN.md`

---

**Enjoy using Shruti AVSR! 🎤📹**
