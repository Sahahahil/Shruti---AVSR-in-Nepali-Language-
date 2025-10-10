"use client";
import { useEffect, useRef } from "react";

interface VideoStreamProps {
  videoRef: React.RefObject<HTMLVideoElement| null> ;
  socket: WebSocket | null;
  isStreaming: boolean;
}

export default function VideoStream({ videoRef, socket, isStreaming }: VideoStreamProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isStreaming && videoRef.current) {
      navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;

        interval = setInterval(() => {
          if (!canvasRef.current || !videoRef.current || !socket) return;
          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) return;

          ctx.drawImage(videoRef.current, 0, 0, 224, 224);
          const frame = canvasRef.current.toDataURL("image/jpeg", 0.5);
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "video_frame", data: frame }));
          }
        }, 200);
      });
    }

    return () => clearInterval(interval);
  }, [isStreaming, socket]);

  return (
    <div className="video-wrapper">
      <video ref={videoRef} autoPlay playsInline muted />
      <canvas ref={canvasRef} width={224} height={224} style={{ display: "none" }} />
    </div>
  );
}
