'use client';

import React, { useEffect, useRef, useCallback } from 'react';

declare global {
  interface Window {
    FaceMesh?: new (config: { locateFile?: (file: string) => string }) => {
      setOptions: (options: Record<string, unknown>) => void;
      onResults: (callback: (results: unknown) => void) => void;
      send: (input: { image: HTMLVideoElement }) => Promise<void>;
      close: () => void;
    };
    Camera?: new (
      video: HTMLVideoElement,
      options: {
        onFrame: () => Promise<void>;
        width?: number;
        height?: number;
        facingMode?: 'user' | 'environment';
      }
    ) => {
      start: () => Promise<void>;
      stop: () => Promise<void>;
    };
  }
}

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

const MEDIAPIPE_FACE_MESH_SCRIPT = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js';
const MEDIAPIPE_CAMERA_UTILS_SCRIPT = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js';

const loadScriptOnce = (src: string): Promise<void> => {
  if (typeof window === 'undefined') {
    return Promise.resolve();
  }

  const existing = document.querySelector(`script[src="${src}"]`) as HTMLScriptElement | null;
  if (existing) {
    if (existing.dataset.loaded === 'true') {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load script: ${src}`)), {
        once: true,
      });
    });
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.onload = () => {
      script.dataset.loaded = 'true';
      resolve();
    };
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.body.appendChild(script);
  });
};

const LipContourVisualizer: React.FC<LipContourVisualizerProps> = ({
  isStreaming,
  width = 320,
  height = 240,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const facemeshRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cameraRef = useRef<any>(null);

  const getOffscreenCanvas = useCallback((w: number, h: number) => {
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }
    const c = offscreenRef.current;
    c.width = Math.max(1, Math.round(w));
    c.height = Math.max(1, Math.round(h));
    return c;
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onResults = useCallback((results: any) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Always paint a white background first for stable contour visibility.
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

    // Draw contour map in the same style as infer_march: white bg, gray lips, black mouth.
    if (bw > 0 && bh > 0) {
      const offscreen = getOffscreenCanvas(canvas.width, canvas.height);
      const offCtx = offscreen.getContext('2d');
      if (!offCtx) return;

      offCtx.fillStyle = 'white';
      offCtx.fillRect(0, 0, offscreen.width, offscreen.height);

      const scaleX = offscreen.width / bw;
      const scaleY = offscreen.height / bh;

      offCtx.fillStyle = 'rgb(180, 180, 180)';
      offCtx.beginPath();
      const scaledOuter = outerPts.map(([px, py]) => [
        (px - x) * scaleX,
        (py - y) * scaleY,
      ]);
      if (scaledOuter.length > 0) {
        offCtx.moveTo(scaledOuter[0][0], scaledOuter[0][1]);
        for (let i = 1; i < scaledOuter.length; i++) {
          offCtx.lineTo(scaledOuter[i][0], scaledOuter[i][1]);
        }
        offCtx.closePath();
        offCtx.fill();
      }

      offCtx.fillStyle = 'black';
      offCtx.beginPath();
      const scaledInner = innerPts.map(([px, py]) => [
        (px - x) * scaleX,
        (py - y) * scaleY,
      ]);
      if (scaledInner.length > 0) {
        offCtx.moveTo(scaledInner[0][0], scaledInner[0][1]);
        for (let i = 1; i < scaledInner.length; i++) {
          offCtx.lineTo(scaledInner[i][0], scaledInner[i][1]);
        }
        offCtx.closePath();
        offCtx.fill();
      }

      // Mirror to match selfie webcam preview.
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }
  }, [getOffscreenCanvas]);

  useEffect(() => {
    if (!isStreaming || !videoRef.current || !canvasRef.current) return;

    let cancelled = false;
    let sendInFlight = false;
    let localFacemesh: {
      setOptions: (options: Record<string, unknown>) => void;
      onResults: (callback: (results: unknown) => void) => void;
      send: (input: { image: HTMLVideoElement }) => Promise<void>;
      close: () => void;
    } | null = null;
    let localCamera: {
      start: () => Promise<void>;
      stop: () => Promise<void>;
    } | null = null;

    const setupFacemesh = async () => {
      const videoEl = videoRef.current;
      if (!videoEl) return;

      try {
        await loadScriptOnce(MEDIAPIPE_FACE_MESH_SCRIPT);
        await loadScriptOnce(MEDIAPIPE_CAMERA_UTILS_SCRIPT);

        const FaceMeshCtor = window.FaceMesh;
        const CameraCtor = window.Camera;

        if (typeof FaceMeshCtor !== 'function' || typeof CameraCtor !== 'function') {
          throw new Error('MediaPipe FaceMesh/Camera constructor not available');
        }

        const facemesh = new FaceMeshCtor({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });
        localFacemesh = facemesh;

        facemesh.setOptions({
          maxNumFaces: 1,
          selfieMode: true,
          refineLandmarks: true,
          minDetectionConfidence: 0.35,
          minTrackingConfidence: 0.35,
        });

        facemesh.onResults((results: unknown) => {
          if (cancelled) {
            return;
          }
          onResults(results);
        });

        const camera = new CameraCtor(videoEl, {
          onFrame: async () => {
            if (cancelled || !videoEl || sendInFlight) {
              return;
            }

            try {
              sendInFlight = true;
              await facemesh.send({ image: videoEl });
            } catch (err) {
              // Ignore teardown races where wasm resources are being released.
              if (!cancelled) {
                console.debug('FaceMesh frame send skipped:', err);
              }
            } finally {
              sendInFlight = false;
            }
          },
          width: 640,
          height: 480,
          facingMode: 'user',
        });
        localCamera = camera;

        if (cancelled) {
          return;
        }

        await camera.start();
        facemeshRef.current = facemesh;
        cameraRef.current = camera;
      } catch (error) {
        console.error('Failed to setup FaceMesh:', error);
      }
    };

    setupFacemesh();

    return () => {
      cancelled = true;

      const teardown = async () => {
        if (localCamera) {
          try {
            await localCamera.stop();
          } catch {
            // ignore stop errors during rapid mode switches
          }
        }

        // Give any in-flight send call a brief chance to settle before close.
        let retries = 0;
        while (sendInFlight && retries < 8) {
          await new Promise((resolve) => setTimeout(resolve, 25));
          retries += 1;
        }

        if (localFacemesh) {
          try {
            localFacemesh.close();
          } catch {
            // ignore close errors during rapid unmount/remount cycles
          }
        }

        if (cameraRef.current === localCamera) {
          cameraRef.current = null;
        }
        if (facemeshRef.current === localFacemesh) {
          facemeshRef.current = null;
        }
      };

      void teardown();
    };
  }, [isStreaming, onResults]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <video
        ref={videoRef}
        style={{
          position: 'absolute',
          width: 640,
          height: 480,
          opacity: 0,
          pointerEvents: 'none',
        }}
        playsInline
        muted
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
