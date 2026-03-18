'use client';

import React, { useEffect, useRef, useState } from 'react';
import styles from '@/styles/tab-content.module.scss';

interface RealtimeWebcamProps {
  isStreaming: boolean;
  onFrame: (dataUrl: string) => void;
  onLipMovementChange?: (active: boolean, score: number) => void;
}

const RealtimeWebcam: React.FC<RealtimeWebcamProps> = ({
  isStreaming,
  onFrame,
  onLipMovementChange,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const prevLipRoiRef = useRef<Uint8ClampedArray | null>(null);
  const noMotionFramesRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setReady(false);
      setError(null);
      prevLipRoiRef.current = null;
      noMotionFramesRef.current = 0;
      onLipMovementChange?.(false, 0);
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

          const roiX = Math.floor(canvas.width * 0.28);
          const roiY = Math.floor(canvas.height * 0.58);
          const roiW = Math.floor(canvas.width * 0.44);
          const roiH = Math.floor(canvas.height * 0.24);

          const roi = ctx.getImageData(roiX, roiY, roiW, roiH).data;
          let movementScore = 0;

          if (prevLipRoiRef.current && prevLipRoiRef.current.length === roi.length) {
            let diffSum = 0;
            for (let i = 0; i < roi.length; i += 4) {
              const prevGray =
                0.299 * prevLipRoiRef.current[i] +
                0.587 * prevLipRoiRef.current[i + 1] +
                0.114 * prevLipRoiRef.current[i + 2];
              const currGray = 0.299 * roi[i] + 0.587 * roi[i + 1] + 0.114 * roi[i + 2];
              diffSum += Math.abs(currGray - prevGray);
            }
            movementScore = diffSum / (roi.length / 4);
          }

          prevLipRoiRef.current = new Uint8ClampedArray(roi);

          const movingNow = movementScore >= 6;
          if (movingNow) {
            noMotionFramesRef.current = 0;
          } else {
            noMotionFramesRef.current += 1;
          }

          // Keep active for a few frames to avoid jitter between words.
          const active = movingNow || noMotionFramesRef.current < 4;
          onLipMovementChange?.(active, movementScore);

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
      prevLipRoiRef.current = null;
      noMotionFramesRef.current = 0;
      onLipMovementChange?.(false, 0);
    };
  }, [isStreaming, onFrame, onLipMovementChange]);

  return (
    <div className={styles.webcamWrap}>
      <video ref={videoRef} className={styles.webcamVideo} playsInline muted autoPlay />
      {!ready && isStreaming && <div className={styles.webcamOverlay}>Starting camera...</div>}
      {error && <div className={styles.errorBox}>{error}</div>}
    </div>
  );
};

export default RealtimeWebcam;
