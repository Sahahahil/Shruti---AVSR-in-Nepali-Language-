'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeASR_VideoInput: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="asr_only"
      title="ASR Only"
      description="Realtime microphone ASR with uploaded video audio-only processing."
      useWebcam={false}
      useMic
      enableUploadVideoAudioOnly
    />
  );
};

export default RealtimeASR_VideoInput;
