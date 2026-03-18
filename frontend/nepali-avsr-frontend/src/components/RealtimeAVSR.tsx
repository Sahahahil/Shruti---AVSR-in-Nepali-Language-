'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeAVSR: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="avsr"
      title="Realtime AVSR"
      description="Live webcam + microphone fusion transcription for Nepali speech."
      useWebcam
      useMic
    />
  );
};

export default RealtimeAVSR;
