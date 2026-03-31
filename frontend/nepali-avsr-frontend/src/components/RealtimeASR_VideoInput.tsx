'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeASR_VideoInput: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="asr_only"
      title="ASR Only"
      description="Live character-level wav2vec2 transcription from microphone audio (about every 2 seconds)."
      useWebcam={false}
      useMic
      enableUploadVideoAudioOnly={false}
    />
  );
};

export default RealtimeASR_VideoInput;
