'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeAVSR: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="avsr"
      title="AVSR"
      description="Realtime webcam + microphone fusion."
      useWebcam
      useMic
    />
  );
};

export default RealtimeAVSR;
