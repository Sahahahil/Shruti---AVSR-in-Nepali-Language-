'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeVSR_ASR: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="vsr_only"
      title="VSR Only"
      description="Realtime webcam lip-reading with uploaded video VSR support."
      useWebcam
      useMic={false}
      enableUploadVideoAudioOnly
    />
  );
};

export default RealtimeVSR_ASR;
