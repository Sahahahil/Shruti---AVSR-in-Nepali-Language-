import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from inference import RealtimeInferencer, transcribe_video


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


class RealtimeState:
    def __init__(self) -> None:
        self.mode = 'avsr'
        self.engine = RealtimeInferencer(audio_window_s=2.0)
        self.last_emit_t = 0.0
        self.emit_interval_s = 0.45

    def set_mode(self, mode: str) -> None:
        if mode in {'avsr', 'vsr_asr', 'vsr_only', 'asr_only'}:
            self.mode = mode

    def reset(self) -> None:
        self.engine.reset()
        self.last_emit_t = 0.0

    def close(self) -> None:
        self.engine.close()

    def add_frame(self, frame_bgr: np.ndarray) -> dict | None:
        self.engine.add_frame(frame_bgr)
        return self._predict_if_ready()

    def add_audio(self, samples: list[float]) -> dict | None:
        if not samples:
            return None

        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0:
            return None

        self.engine.add_audio(arr)
        return self._predict_if_ready()

    def _predict_if_ready(self) -> dict | None:
        now = time.time()
        if now - self.last_emit_t < self.emit_interval_s:
            return None

        if self.mode == 'vsr_only' and not self.engine.has_vsr_context():
            return None

        if self.mode == 'asr_only' and not self.engine.has_asr_context():
            return None

        if self.mode in {'avsr', 'vsr_asr'} and not (self.engine.has_vsr_context() and self.engine.has_asr_context()):
            return None

        self.last_emit_t = now
        t0 = time.time()
        pred = self.engine.predict(self.mode)
        pred.update({
            'type': 'prediction',
            'mode': self.mode,
            'latency': int((time.time() - t0) * 1000),
        })
        return pred


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


@app.post('/api/vsr-only/video')
async def vsr_only_video(file: UploadFile = File(...)):
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

        result = transcribe_video(str(file_path), mode='vsr_only')
        return {
            'status': 'success',
            'mode': 'VSR Only (Video Input)',
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
    state: RealtimeState | None = None

    try:
        state = RealtimeState()
        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)
            msg_type = msg.get('type')

            if msg_type == 'config':
                if state is not None:
                    state.set_mode(msg.get('mode', 'avsr'))
                continue

            if msg_type == 'reset':
                if state is not None:
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

                # Keep realtime processing in selfie orientation to match frontend preview.
                frame = cv2.flip(frame, 1)

                if state is None:
                    continue

                out = state.add_frame(frame)
                if out is not None:
                    await websocket.send_text(json.dumps(out, ensure_ascii=False))
                continue

            if msg_type == 'audio':
                if state is None:
                    continue

                out = state.add_audio(msg.get('data', []))
                if out is not None:
                    await websocket.send_text(json.dumps(out, ensure_ascii=False))
                continue

    except WebSocketDisconnect:
        return
    finally:
        if state is not None:
            state.close()


@app.get('/api/config')
async def get_config():
    return {
        'supported_tabs': ['avsr', 'vsr_only', 'asr_only'],
        'allowed_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 100,
        'language': 'Nepali',
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
