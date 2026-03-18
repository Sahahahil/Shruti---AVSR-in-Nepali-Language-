'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeASR_VideoInput: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="asr_only"
      title="Realtime ASR + Video Input"
      description="Live microphone audio transcription. Video is displayed but only audio is processed by the backend."
      useWebcam
      useMic
      enableUploadVideoAudioOnly={false}
    />
  );
};

export default RealtimeASR_VideoInput;
