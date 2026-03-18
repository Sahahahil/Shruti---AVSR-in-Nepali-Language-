'use client';

import React, { useEffect, useRef, useCallback } from 'react';
import * as Facemesh from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

interface LipContourVisualizerProps {
  isStreaming: boolean;
  width?: number;
  height?: number;
}

const OUTER_LIPS = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];

const INNER_LIPS = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

const LIP_PAD_X = 30;
const LIP_PAD_Y = 15;
const LIP_ASPECT_RATIO = 2.0;
const MOUTH_OPEN_REF_RATIO = 0.25;
const ADAPTIVE_PAD_ENABLED = true;
const MIN_PAD_SCALE = 0.85;
const MAX_PAD_SCALE = 1.45;

const LipContourVisualizer: React.FC<LipContourVisualizerProps> = ({
  isStreaming,
  width = 320,
  height = 240,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const facemeshRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cameraRef = useRef<any>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onResults = useCallback((results: any) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear with white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      return;
    }

    const landmarks = results.multiFaceLandmarks[0];
    const videoW = results.image.width;
    const videoH = results.image.height;

    // Extract outer and inner lip points
    const outerPts = OUTER_LIPS.map((idx) => {
      const lm = landmarks[idx];
      return [lm.x * videoW, lm.y * videoH];
    }) as [number, number][];

    const innerPts = INNER_LIPS.map((idx) => {
      const lm = landmarks[idx];
      return [lm.x * videoW, lm.y * videoH];
    }) as [number, number][];

    // Compute bounding box from outer contour (like infer_march.py)
    let minX = videoW, maxX = 0, minY = videoH, maxY = 0;
    outerPts.forEach(([px, py]) => {
      minX = Math.min(minX, px);
      maxX = Math.max(maxX, px);
      minY = Math.min(minY, py);
      maxY = Math.max(maxY, py);
    });

    let x = minX;
    let y = minY;
    let bw = maxX - minX;
    let bh = maxY - minY;

    // Mouth open ratio for adaptive padding (like infer_march.py)
    const INNER_UPPER_IDX = 5;
    const INNER_LOWER_IDX = 15;
    const upperY = innerPts[INNER_UPPER_IDX][1];
    const lowerY = innerPts[INNER_LOWER_IDX][1];
    const mouthOpenPx = Math.abs(lowerY - upperY);
    const mouthOpenRatio = mouthOpenPx / Math.max(bh, 1.0);

    // Compute adaptive pad scale
    let padScale = 1.0;
    if (ADAPTIVE_PAD_ENABLED) {
      const normalized = mouthOpenRatio / Math.max(MOUTH_OPEN_REF_RATIO, 1e-6);
      padScale = 0.85 + 0.35 * normalized;
      padScale = Math.max(MIN_PAD_SCALE, Math.min(MAX_PAD_SCALE, padScale));
    }

    const padX = Math.max(1, Math.round(LIP_PAD_X * padScale));
    const padY = Math.max(1, Math.round(LIP_PAD_Y * padScale));

    x = Math.max(0, x - padX);
    y = Math.max(0, y - padY);
    bw = Math.min(videoW - x, bw + 2 * padX);
    bh = Math.min(videoH - y, bh + 2 * padY);

    // Apply aspect ratio constraint
    const desiredBw = LIP_ASPECT_RATIO * bh;
    const dw = desiredBw - bw;
    x = Math.max(0, x - dw / 2);
    bw = Math.min(videoW - x, desiredBw);

    // Crop from video and draw on canvas
    if (bw > 0 && bh > 0) {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
        data[i + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);

      // Draw cropped lip region
      const scaleX = canvas.width / bw;
      const scaleY = canvas.height / bh;

      // Draw outer lip contour (gray)
      ctx.fillStyle = 'rgb(180, 180, 180)';
      ctx.beginPath();
      const scaledOuter = outerPts.map(([px, py]) => [
        (px - x) * scaleX,
        (py - y) * scaleY,
      ]);
      if (scaledOuter.length > 0) {
        ctx.moveTo(scaledOuter[0][0], scaledOuter[0][1]);
        for (let i = 1; i < scaledOuter.length; i++) {
          ctx.lineTo(scaledOuter[i][0], scaledOuter[i][1]);
        }
        ctx.closePath();
        ctx.fill();
      }

      // Draw inner mouth (black)
      ctx.fillStyle = 'black';
      ctx.beginPath();
      const scaledInner = innerPts.map(([px, py]) => [
        (px - x) * scaleX,
        (py - y) * scaleY,
      ]);
      if (scaledInner.length > 0) {
        ctx.moveTo(scaledInner[0][0], scaledInner[0][1]);
        for (let i = 1; i < scaledInner.length; i++) {
          ctx.lineTo(scaledInner[i][0], scaledInner[i][1]);
        }
        ctx.closePath();
        ctx.fill();
      }
    }
  }, []);

  useEffect(() => {
    if (!isStreaming || !videoRef.current || !canvasRef.current) return;

    const setupFacemesh = async () => {
      const videoEl = videoRef.current;
      if (!videoEl) return;

      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const facemesh = new (Facemesh.FaceMesh as any)({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });

        facemesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.4,
          minTrackingConfidence: 0.6,
        });

        facemesh.onResults(onResults);

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const camera = new (Camera as any)(videoEl, {
          onFrame: async () => {
            if (videoEl) {
              await facemesh.send({ image: videoEl });
            }
          },
          width: 640,
          height: 480,
        });

        camera.start();
        facemeshRef.current = facemesh;
        cameraRef.current = camera;
      } catch (error) {
        console.error('Failed to setup FaceMesh:', error);
      }
    };

    setupFacemesh();

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
      if (facemeshRef.current) {
        facemeshRef.current.close();
      }
    };
  }, [isStreaming, onResults]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <video
        ref={videoRef}
        style={{ display: 'none', width: 640, height: 480 }}
      />
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          border: '2px solid #ec4899',
          borderRadius: '8px',
          background: 'white',
        }}
      />
    </div>
  );
};

export default LipContourVisualizer;
