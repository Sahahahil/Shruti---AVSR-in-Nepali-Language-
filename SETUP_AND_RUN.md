# Shruti: Nepali Audio-Visual Speech Recognition (AVSR)

A complete Full-Stack Application for Audio-Visual Speech Recognition in Nepali language with modern architecture and three different processing modes.

## 📋 Project Structure

```
.
├── backend/               # FastAPI Backend Server
│   ├── app.py            # Main FastAPI application
│   ├── inference.py      # Model inference logic (3 modes)
│   ├── requirements.txt   # Python dependencies
│   ├── models/           # ML Model definitions
│   ├── utils/            # Utility functions (audio/video processing)
│   └── uploads/          # Temporary upload storage
│
└── frontend/              # Next.js Frontend Web Application
    └── nepali-avsr-frontend/
        ├── src/
        │   ├── app/              # Next.js pages
        │   ├── components/       # React components
        │   │   ├── Dashboard.tsx        # Main tab container
        │   │   ├── RealtimeAVSR.tsx     # Tab 1: AVSR mode
        │   │   ├── RealtimeVSR_ASR.tsx  # Tab 2: Separated VSR+ASR
        │   │   ├── RealtimeASR_VideoInput.tsx  # Tab 3: ASR only
        │   │   ├── FileUpload.tsx       # File upload component
        │   │   ├── ResultDisplay.tsx    # Results display
        │   │   └── ProcessingIndicator.tsx  # Loading indicator
        │   ├── lib/
        │   │   ├── api-client.ts  # Axios API client
        │   │   └── store.ts       # Zustand state management
        │   └── styles/            # SASS modules
        ├── package.json
        └── .env.local            # Environment configuration
```

## 🚀 Features

### Three Processing Modes

1. **Tab 1: Audio-Visual Speech Recognition (AVSR)**
   - Combines both audio and video information
   - Uses CNN/RNN fusion model
   - Best for normal environments
   - Higher accuracy with visual cues

2. **Tab 2: Separated VSR + ASR**
   - Visual Speech Recognition from lip movements
   - Audio Speech Recognition from sound
   - Compare accuracy of both modalities
   - Useful for research and analysis

3. **Tab 3: ASR Only (Video Input)**
   - Takes video input but uses only audio extraction
   - Uses audio speech recognition model
   - Simpler and faster processing
   - Good for audio-focused scenarios

### Technology Stack

**Frontend:**
- Next.js 15+ with React 19
- TypeScript for type safety
- TanStack Query for data fetching
- Axios for HTTP requests
- Framer Motion for animations
- SASS for styling
- Zustand for state management

**Backend:**
- FastAPI for modern async API
- PyTorch for model inference
- OpenCV for video processing
- FFmpeg for audio extraction
- Python 3.8+

## 📦 Installation & Setup

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+
- FFmpeg (for audio extraction)

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   # Ensure uploads and processed directories exist
   mkdir -p uploads processed
   ```

3. **Start FastAPI server:**
   ```bash
   cd backend
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   The API will be available at `http://localhost:8000`

   **API Endpoints:**
   - `GET /health` - Health check
   - `GET /api/config` - Get API configuration
   - `POST /api/avsr/realtime` - Tab 1: AVSR processing
   - `POST /api/vsr-asr/realtime` - Tab 2: VSR+ASR processing
   - `POST /api/asr-only/video` - Tab 3: ASR only processing

### Frontend Setup

1. **Install Node dependencies:**
   ```bash
   cd frontend/nepali-avsr-frontend
   npm install
   ```

2. **Configure environment:**
   
   Edit `.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   Navigate to `http://localhost:3000`

4. **Build for production:**
   ```bash
   npm run build
   npm start
   ```

## 🔧 API Reference

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "service": "Nepali AVSR Backend"
}
```

### Tab 1: AVSR Realtime
```bash
POST /api/avsr/realtime
Content-Type: multipart/form-data

# Body: video file
```
Response:
```json
{
  "status": "success",
  "mode": "AVSR (Audio + Video)",
  "transcription": "नमस्ते",
  "confidence": 0.85,
  "processing_time": 2.34
}
```

### Tab 2: VSR + ASR Realtime
```bash
POST /api/vsr-asr/realtime
Content-Type: multipart/form-data

# Body: video file
```
Response:
```json
{
  "status": "success",
  "mode": "VSR + ASR (Separated)",
  "vsr_transcription": "नमस्ते",
  "asr_transcription": "नमस्ते",
  "vsr_confidence": 0.78,
  "asr_confidence": 0.82,
  "processing_time": 2.15
}
```

### Tab 3: ASR Only with Video Input
```bash
POST /api/asr-only/video
Content-Type: multipart/form-data

# Body: video file
```
Response:
```json
{
  "status": "success",
  "mode": "ASR Only (Video Input)",
  "transcription": "नमस्ते",
  "confidence": 0.88,
  "processing_time": 1.95
}
```

### Get Configuration
```bash
GET /api/config
```
Response:
```json
{
  "supported_modes": ["AVSR", "VSR+ASR", "ASR_Only"],
  "allowed_formats": ["webm", "mp4", "webp", "avi", "mov"],
  "max_file_size_mb": 100,
  "language": "Nepali"
}
```

## 🎨 Frontend Components

### Main Components

**Dashboard.tsx**
- Tab navigation and switching
- Manages active tab state
- Animated transitions

**RealtimeAVSR.tsx** (Tab 1)
- File upload interface
- AVSR processing
- Result display

**RealtimeVSR_ASR.tsx** (Tab 2)
- File upload interface
- Separated VSR and ASR processing
- Dual result display

**RealtimeASR_VideoInput.tsx** (Tab 3)
- File upload interface
- Audio-only processing
- Single result display

### Shared Components

**FileUpload.tsx**
- Drag and drop file upload
- File validation
- Progress tracking

**ResultDisplay.tsx**
- Transcription display
- Confidence scores
- Processing statistics
- Copy to clipboard functionality

**ProcessingIndicator.tsx**
- Animated loading spinner
- Status messages
- Progress indication

## 🔐 Error Handling

The application includes comprehensive error handling:

1. **Bad File Format:**
   ```json
   { "error": "File type not allowed. Allowed: webm, mp4, webp, avi" }
   ```

2. **Large File:** Automatically handled with progress tracking

3. **Processing Error:**
   ```json
   { "error": "Processing failed. Please try again." }
   ```

4. **Backend Connection Error:**
   ```json
   { "error": "Failed to connect to backend service" }
   ```

## 📱 Responsive Design

- Mobile-first approach
- Breakpoints: 768px, 1024px
- Touch-friendly interface
- Optimized for all devices

## 🎯 Performance Optimizations

- Lazy loading of components
- Memoization of expensive computations
- Efficient state management with Zustand
- Progress tracking for large file uploads
- Optimized SASS compilation

## 🔍 Development Tips

### Adding New Features

1. **New processing mode:** Update `inference.py` and add new endpoint in `app.py`
2. **New component:** Create in `src/components/` and import in Dashboard
3. **New API call:** Add method to `lib/api-client.ts`
4. **State management:** Add to Zustand store in `lib/store.ts`

### Debugging

- Frontend: Check browser console and Network tab
- Backend: Check terminal output and add print statements
- API: Use tools like Postman or curl to test endpoints

### Common Issues

1. **CORS Error:** Ensure backend has CORS middleware enabled
2. **File Upload Fails:** Check file size and format
3. **API Connection Error:** Verify `NEXT_PUBLIC_API_URL` in `.env.local`

## 📊 Model Integration

To integrate your trained models:

1. **Update model paths** in `backend/models/`
2. **Modify preprocessing** in `backend/utils/`
3. **Update character mapping** in `backend/inference.py`
4. **Test with sample videos** before deployment

## 🚀 Deployment

### Backend Deployment (AWS/DigitalOcean)

```bash
# Build Docker image
docker build -t avsr-backend .

# Push to registry
docker push your-registry/avsr-backend

# Run container
docker run -p 8000:8000 your-registry/avsr-backend
```

### Frontend Deployment (Vercel/Netlify)

```bash
# Build production
npm run build

# Deploy to Vercel
vercel deploy --prod

# Deploy to Netlify
netlify deploy --prod
```

## 📝 License

This project is part of Major Project 2025.

## 👥 Contributors

- Shruti AVSR Team

## 📧 Support

For issues and questions, please create an issue in the repository.

---

**Happy Speech Recognition! 🎤📹**
