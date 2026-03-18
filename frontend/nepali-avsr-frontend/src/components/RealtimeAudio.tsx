'use client';

import React, { useEffect, useRef, useState } from 'react';
import styles from '@/styles/tab-content.module.scss';

interface RealtimeAudioProps {
  isStreaming: boolean;
  onAudio: (samples: Float32Array) => void;
  onSoundActivityChange?: (active: boolean, level: number) => void;
}

const RealtimeAudio: React.FC<RealtimeAudioProps> = ({
  isStreaming,
  onAudio,
  onSoundActivityChange,
}) => {
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const silenceFramesRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setLevel(0);
      setError(null);
      silenceFramesRef.current = 0;
      onSoundActivityChange?.(false, 0);
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
          const newLevel = Math.min(100, (sum / samples.length) * 600);
          setLevel(newLevel);

          const speakingNow = newLevel >= 7;
          if (speakingNow) {
            silenceFramesRef.current = 0;
          } else {
            silenceFramesRef.current += 1;
          }

          // Add a small hangover so short pauses do not immediately stop predictions.
          const active = speakingNow || silenceFramesRef.current < 4;
          onSoundActivityChange?.(active, newLevel);
          onAudio(samples);
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
      onSoundActivityChange?.(false, 0);
    };
  }, [isStreaming, onAudio, onSoundActivityChange]);

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
