'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import RealtimeAudio from './RealtimeAudio';
import RealtimeWebcam from './RealtimeWebcam';
import LipContourVisualizer from './LipContourVisualizer';
import styles from '@/styles/tab-content.module.scss';

type WsMode = 'avsr' | 'vsr_only' | 'asr_only';

interface LivePrediction {
  prediction: string;
  confidence: number;
  latency: number;
  vsr_prediction?: string;
  vsr_confidence?: number;
  asr_prediction?: string;
  asr_confidence?: number;
  vsr_weight?: number;
  asr_weight?: number;
}

interface RealtimeStreamPanelProps {
  mode: WsMode;
  title: string;
  description: string;
  useWebcam: boolean;
  useMic: boolean;
  enableUploadVideoAudioOnly?: boolean;
}

const TRAINED_WORD_CLASSES = [
  '\u0905\u0917\u093e\u0921\u093f',
  '\u091c\u093e\u090a',
  '\u0924\u0932',
  '\u0924\u093f\u092e\u0940',
  '\u0926\u093e\u092f\u093e\u0901',
  '\u092a\u091b\u093e\u0921\u093f',
  '\u092c\u093e\u092f\u093e\u0901',
  '\u092e\u093e\u0925\u093f',
  '\u0930\u094b\u0915',
];

const RealtimeStreamPanel: React.FC<RealtimeStreamPanelProps> = ({
  mode,
  title,
  description,
  useWebcam,
  useMic,
  enableUploadVideoAudioOnly = false,
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const audioStepRef = useRef(0);

  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<LivePrediction | null>(null);
  const [history, setHistory] = useState<LivePrediction[]>([]);

  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{
    transcription?: string;
    confidence?: number;
    classification?: string;
  } | null>(null);

  const wsUrl = useMemo(() => {
    const base = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    return `${base.replace(/^http/, 'ws')}/ws/realtime`;
  }, []);

  const closeSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    setStatus('idle');
    closeSocket();
  }, [closeSocket]);

  useEffect(() => {
    return () => {
      closeSocket();
    };
  }, [closeSocket]);

  const startStreaming = useCallback(() => {
    setError(null);
    setUploadResult(null);
    setStatus('connecting');

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      setIsStreaming(true);
      ws.send(JSON.stringify({ type: 'config', mode }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type !== 'prediction') {
          return;
        }

        const pred: LivePrediction = {
          prediction: msg.prediction || '-',
          confidence: Number(msg.confidence || 0),
          latency: Number(msg.latency || 0),
          vsr_prediction: msg.vsr_prediction,
          vsr_confidence: msg.vsr_confidence,
          asr_prediction: msg.asr_prediction,
          asr_confidence: msg.asr_confidence,
          vsr_weight: msg.vsr_weight,
          asr_weight: msg.asr_weight,
        };

        setLive(pred);
        setHistory((prev) => [pred, ...prev].slice(0, 6));
      } catch {
        setError('Failed to parse realtime response');
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection failed');
      setStatus('error');
      setIsStreaming(false);
    };

    ws.onclose = () => {
      setIsStreaming(false);
      setStatus((prev) => (prev === 'error' ? 'error' : 'idle'));
    };
  }, [mode, wsUrl]);

  const sendFrame = useCallback((dataUrl: string) => {
    if (!isStreaming || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    // Skip sending video frames in ASR-only mode (audio only)
    if (mode === 'asr_only') {
      return;
    }
    wsRef.current.send(JSON.stringify({ type: 'frame', data: dataUrl }));
  }, [isStreaming, mode]);

  const sendAudio = useCallback((samples: Float32Array, sampleRate: number) => {
    if (!isStreaming || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    audioStepRef.current += 1;
    if (audioStepRef.current % 3 !== 0) {
      return;
    }

    wsRef.current.send(JSON.stringify({
      type: 'audio',
      data: Array.from(samples),
      sample_rate: sampleRate,
    }));
  }, [isStreaming]);

  const handleUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploading(true);
    setError(null);
    try {
      const res = await apiClient.uploadForASR_Only(file);
      setUploadResult({
        transcription: res.transcription,
        confidence: res.confidence,
        classification: (res as { classification?: string }).classification,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Upload failed';
      setError(msg);
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  }, []);

  return (
    <div className={styles.realtimePanel}>
      <div className={styles.leftColumn}>
        <div className={styles.card}>
          <h3 className={styles.sectionTitle}>{title}</h3>
          <p className={styles.sectionDescription}>{description}</p>

          <div className={styles.controls}>
            {!isStreaming ? (
              <button className={styles.startBtn} onClick={startStreaming}>Start Realtime</button>
            ) : (
              <button className={styles.stopBtn} onClick={stopStreaming}>Stop</button>
            )}
            <span className={styles.statusBadge} data-state={status}>{status}</span>
          </div>

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
            {useWebcam && <RealtimeWebcam isStreaming={isStreaming} onFrame={sendFrame} />}
            {useWebcam && (
              <div>
                <p className={styles.smallNote}>Live Lip Contour (Cropped)</p>
                <LipContourVisualizer isStreaming={isStreaming} width={128} height={96} />
              </div>
            )}
          </div>
          {useMic && <RealtimeAudio isStreaming={isStreaming} onAudio={sendAudio} />}

          {enableUploadVideoAudioOnly && (
            <div className={styles.uploadOnlyCard}>
              <h4 className={styles.featuresTitle}>Video + Audio Input (Audio Processing)</h4>
              <p className={styles.sectionDescription}>
                Upload a video file. The backend extracts audio and returns audio-only classification/transcription.
              </p>
              <input
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
                onChange={handleUpload}
                className={styles.fileInput}
                disabled={uploading}
              />
              {uploading && <p className={styles.smallNote}>Processing uploaded video...</p>}
              {uploadResult && (
                <div className={styles.uploadResult}>
                  <p><strong>Transcription:</strong> {uploadResult.transcription || '-'}</p>
                  <p><strong>Classification:</strong> {uploadResult.classification || '-'}</p>
                  <p><strong>Confidence:</strong> {((uploadResult.confidence || 0) * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>
          )}

          {error && <div className={styles.errorBox}>{error}</div>}
        </div>
      </div>

      <div className={styles.rightColumn}>
        <div className={styles.card}>
          <h4 className={styles.featuresTitle}>
            {mode === 'asr_only' ? 'Live ASR Output' : 'Live Transcription'}
          </h4>
          <div className={styles.liveOutput}>
            {live ? (
              <>
                <p className={styles.livePrediction}>{live.prediction}</p>
                <p className={styles.smallNote}>Confidence: {(live.confidence * 100).toFixed(1)}%</p>
                <p className={styles.smallNote}>Latency: {live.latency} ms</p>
                {mode === 'avsr' && (
                  <div className={styles.breakdown}>
                    <p>VSR: {live.vsr_prediction || '-'} ({((live.vsr_confidence || 0) * 100).toFixed(1)}%)</p>
                    <p>ASR: {live.asr_prediction || '-'} ({((live.asr_confidence || 0) * 100).toFixed(1)}%)</p>
                    <p>Fusion Weights: VSR {(live.vsr_weight ?? 0.3).toFixed(2)} | ASR {(live.asr_weight ?? 0.7).toFixed(2)}</p>
                  </div>
                )}
              </>
            ) : (
              <p className={styles.smallNote}>Start streaming to receive live transcriptions.</p>
            )}
          </div>

          {(mode === 'avsr' || mode === 'vsr_only') ? (
            <>
              <h4 className={styles.featuresTitle}>Trained Word Classes</h4>
              <div className={styles.historyList}>
                {TRAINED_WORD_CLASSES.map((word) => (
                  <div className={styles.historyItem} key={word}>
                    <span>{word}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>
              <h4 className={styles.featuresTitle}>Recent Predictions</h4>
              <div className={styles.historyList}>
                {history.length === 0 && <p className={styles.smallNote}>No predictions yet.</p>}
                {history.map((item, idx) => (
                  <div className={styles.historyItem} key={`${item.prediction}-${idx}`}>
                    <span>{item.prediction}</span>
                    <span>{(item.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RealtimeStreamPanel;
