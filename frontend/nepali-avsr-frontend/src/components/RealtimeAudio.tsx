'use client';

import React, { useEffect, useRef, useState } from 'react';
import styles from '@/styles/tab-content.module.scss';

interface RealtimeAudioProps {
  isStreaming: boolean;
  onAudio: (samples: Float32Array, sampleRate: number) => void;
}

const RealtimeAudio: React.FC<RealtimeAudioProps> = ({ isStreaming, onAudio }) => {
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const contextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (!isStreaming) {
      setLevel(0);
      setError(null);
      if (contextRef.current) {
        contextRef.current.close();
        contextRef.current = null;
      }
      return;
    }

    let stream: MediaStream | null = null;
    let processor: ScriptProcessorNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true },
          video: false,
        });

        const Ctor =
          window.AudioContext ||
          (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!Ctor) {
          throw new Error('AudioContext is not supported in this browser');
        }

        const audioContext = new Ctor();
        contextRef.current = audioContext;

        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(2048, 1, 1);

        processor.onaudioprocess = (event) => {
          const input = event.inputBuffer.getChannelData(0);
          const samples = new Float32Array(input);

          let sum = 0;
          for (let i = 0; i < samples.length; i++) {
            sum += Math.abs(samples[i]);
          }
          setLevel(Math.min(100, (sum / samples.length) * 600));
          onAudio(samples, audioContext.sampleRate);
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Unable to access microphone';
        setError(msg);
      }
    };

    start();

    return () => {
      if (processor) {
        processor.disconnect();
      }
      if (source) {
        source.disconnect();
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (contextRef.current) {
        contextRef.current.close();
        contextRef.current = null;
      }
    };
  }, [isStreaming, onAudio]);

  return (
    <div className={styles.audioPanel}>
      <div className={styles.audioLabel}>Microphone Level</div>
      <div className={styles.audioBar}>
        <div className={styles.audioFill} style={{ width: `${level}%` }} />
      </div>
      {error && <div className={styles.errorBox}>{error}</div>}
    </div>
  );
};

export default RealtimeAudio;
