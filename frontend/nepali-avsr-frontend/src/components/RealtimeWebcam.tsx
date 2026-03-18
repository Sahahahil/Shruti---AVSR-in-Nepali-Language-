'use client';

import React, { useEffect, useRef, useState } from 'react';
import styles from '@/styles/tab-content.module.scss';

interface RealtimeWebcamProps {
  isStreaming: boolean;
  onFrame: (dataUrl: string) => void;
}

const RealtimeWebcam: React.FC<RealtimeWebcamProps> = ({ isStreaming, onFrame }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!isStreaming) {
      setReady(false);
      setError(null);
      return;
    }

    let stream: MediaStream | null = null;
    let timer: number | null = null;
    const videoEl = videoRef.current;

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });

        if (!videoEl) {
          return;
        }

        videoEl.srcObject = stream;
        await videoEl.play();
        setReady(true);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        timer = window.setInterval(() => {
          if (!ctx || !videoEl || videoEl.readyState < 2) {
            return;
          }
          canvas.width = videoEl.videoWidth || 640;
          canvas.height = videoEl.videoHeight || 480;
          ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

          onFrame(canvas.toDataURL('image/jpeg', 0.65));
        }, 100);
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Unable to access webcam';
        setError(msg);
      }
    };

    start();

    return () => {
      if (timer) {
        window.clearInterval(timer);
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
    };
  }, [isStreaming, onFrame]);

  return (
    <div className={styles.webcamWrap}>
      <video ref={videoRef} className={styles.webcamVideo} playsInline muted autoPlay />
      {!ready && isStreaming && <div className={styles.webcamOverlay}>Starting camera...</div>}
      {error && <div className={styles.errorBox}>{error}</div>}
    </div>
  );
};

export default RealtimeWebcam;
