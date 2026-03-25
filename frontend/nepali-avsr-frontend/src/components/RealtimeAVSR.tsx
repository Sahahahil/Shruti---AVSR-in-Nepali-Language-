'use client';

import React from 'react';
import RealtimeStreamPanel from './RealtimeStreamPanel';

const RealtimeAVSR: React.FC = () => {
  return (
    <RealtimeStreamPanel
      mode="avsr"
      title="AVSR"
      description="Realtime webcam + microphone fusion with uploaded video AVSR support."
      useWebcam
      useMic
      enableUploadVideoAudioOnly
    />
  );
};

export default RealtimeAVSR;
