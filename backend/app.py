import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from inference import transcribe_video
from realtime_engine import RealtimeSession


app = FastAPI(
    title='Nepali AVSR API',
    description='Audio-Visual Speech Recognition for Nepali Language',
    version='2.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

UPLOAD_FOLDER = Path('uploads')
PROCESSED_FOLDER = Path('processed')
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'webm', 'mp4', 'webp', 'avi', 'mov'}
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get('/health')
async def health_check():
    return {'status': 'healthy', 'service': 'Nepali AVSR Backend'}


@app.post('/api/avsr/realtime')
async def avsr_realtime(file: UploadFile = File(...)):
    file_path = None
    try:
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        file_path = UPLOAD_FOLDER / file.filename
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        result = transcribe_video(str(file_path), mode='avsr')
        return {
            'status': 'success',
            'mode': 'AVSR (Audio + Video)',
            'transcription': result.get('transcription', ''),
            'confidence': result.get('confidence', 0.0),
            'processing_time': result.get('processing_time', 0.0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and file_path.exists():
            file_path.unlink()


@app.post('/api/vsr-asr/realtime')
async def vsr_asr_realtime(file: UploadFile = File(...)):
    file_path = None
    try:
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        file_path = UPLOAD_FOLDER / file.filename
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        result = transcribe_video(str(file_path), mode='vsr_asr')
        return {
            'status': 'success',
            'mode': 'VSR + ASR (Separated)',
            'vsr_transcription': result.get('vsr_transcription', ''),
            'asr_transcription': result.get('asr_transcription', ''),
            'vsr_confidence': result.get('vsr_confidence', 0.0),
            'asr_confidence': result.get('asr_confidence', 0.0),
            'processing_time': result.get('processing_time', 0.0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and file_path.exists():
            file_path.unlink()


@app.post('/api/asr-only/video')
async def asr_only_video(file: UploadFile = File(...)):
    file_path = None
    try:
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        file_path = UPLOAD_FOLDER / file.filename
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        result = transcribe_video(str(file_path), mode='asr_only')
        return {
            'status': 'success',
            'mode': 'ASR Only (Video Input)',
            'transcription': result.get('transcription', ''),
            'confidence': result.get('confidence', 0.0),
            'processing_time': result.get('processing_time', 0.0),
            'classification': result.get('classification', result.get('transcription', '')),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and file_path.exists():
            file_path.unlink()


@app.websocket('/ws/realtime')
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    state = RealtimeSession()

    try:
        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)
            msg_type = msg.get('type')

            if msg_type == 'config':
                state.set_mode(msg.get('mode', 'avsr'))
                continue

            if msg_type == 'reset':
                state.reset()
                continue

            if msg_type == 'frame':
                data_url = msg.get('data', '')
                if ',' not in data_url:
                    continue

                frame_bytes = base64.b64decode(data_url.split(',', 1)[1])
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                out = state.add_frame(frame)
                if out is not None:
                    await websocket.send_text(json.dumps(out, ensure_ascii=False))
                continue

            if msg_type == 'audio':
                out = state.add_audio(msg.get('data', []), msg.get('sample_rate'))
                if out is not None:
                    await websocket.send_text(json.dumps(out, ensure_ascii=False))
                continue

    except WebSocketDisconnect:
        return
    finally:
        state.close()


@app.get('/api/config')
async def get_config():
    return {
        'supported_tabs': ['realtime_avsr', 'realtime_vsr_only', 'asr_only_plus_video_audio_input'],
        'allowed_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 100,
        'language': 'Nepali',
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
