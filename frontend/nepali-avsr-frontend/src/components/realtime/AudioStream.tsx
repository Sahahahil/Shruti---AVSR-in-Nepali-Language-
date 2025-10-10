"use client";
import { useEffect, useRef } from "react";

interface AudioStreamProps {
  socket: WebSocket | null;
  isStreaming: boolean;
}

export default function AudioStream({ socket, isStreaming }: AudioStreamProps) {
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const Float32ArrayToBase64 = (buffer: Float32Array) => {
    let binary = "";
    const bytes = new Uint8Array(buffer.buffer);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize) as any);
    }
    return btoa(binary);
  };

  useEffect(() => {
    if (!isStreaming) return;

    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      audioContextRef.current = new AudioContext();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      processorRef.current = audioContextRef.current.createScriptProcessor(2048, 1, 1);

      source.connect(processorRef.current);
      processorRef.current.connect(audioContextRef.current.destination);

      processorRef.current.onaudioprocess = (e) => {
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        const float32Data = e.inputBuffer.getChannelData(0);
        const base64 = Float32ArrayToBase64(float32Data);
        socket.send(JSON.stringify({ type: "audio_chunk", data: base64 }));
      };
    });

    return () => {
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, [isStreaming]);

  return null;
}
