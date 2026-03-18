'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeVSR_ASR: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="vsr_only"
      title="Realtime VSR Only"
      description="Live webcam lip-reading without microphone input."
      useWebcam
      useMic={false}
    />
  );
};

export default RealtimeVSR_ASR;
