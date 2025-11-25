// "use client";
// import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

// export interface RecorderHandle {
//   startRecording: () => Promise<void>;
//   stopRecording: () => Promise<Blob | null>;
//   startCamera: () => Promise<void>;
//   stopCamera: () => void;
// }

// interface Props {
//   isRecording: boolean;
// }

// const CameraRecorder = forwardRef<RecorderHandle, Props>(({ isRecording }, ref) => {
//   const videoRef = useRef<HTMLVideoElement>(null);
//   const streamRef = useRef<MediaStream | null>(null);
//   const recorderRef = useRef<MediaRecorder | null>(null);
//   const chunks = useRef<BlobPart[]>([]);

//   // ---- CAMERA CONTROLS ----
//   const startCamera = async () => {
//     if (streamRef.current) return;

//     const stream = await navigator.mediaDevices.getUserMedia({
//       video: true,
//       audio: true,
//     });

//     streamRef.current = stream;
//     if (videoRef.current) videoRef.current.srcObject = stream;
//   };

//   const stopCamera = () => {
//     if (!streamRef.current) return;
//     streamRef.current.getTracks().forEach(t => t.stop());
//     streamRef.current = null;

//     if (videoRef.current) {
//       videoRef.current.srcObject = null;
//       videoRef.current.load();
//     }
//   };

//   // ---- RECORDING CONTROLS ----
//   const startRecording = async () => {
//     if (!streamRef.current) await startCamera();

//     chunks.current = [];
//     recorderRef.current = new MediaRecorder(streamRef.current!, {
//       mimeType: "video/webm;codecs=vp8,opus",
//     });

//     recorderRef.current.ondataavailable = e => {
//       if (e.data.size > 0) chunks.current.push(e.data);
//     };
//   };

//   const stopRecording = async (): Promise<Blob | null> => {
//     return new Promise(resolve => {
//       const recorder = recorderRef.current;
//       if (!recorder || recorder.state === "inactive") return resolve(null);

//       recorder.onstop = () => {
//         const blob = new Blob(chunks.current, { type: "video/webm" });
//         stopCamera();
//         resolve(blob);
//       };

//       recorder.stop();
//     });
//   };

//   // Expose methods to parent
//   useImperativeHandle(ref, () => ({
//     startRecording,
//     stopRecording,
//     startCamera,
//     stopCamera,
//   }));

//   // Auto-start/stop recording
//   useEffect(() => {
//     if (isRecording) {
//       recorderRef.current?.start();
//     }
//   }, [isRecording]);

//   return (
//     <video ref={videoRef} autoPlay playsInline muted className="live-video" />
//   );
// });

// export default CameraRecorder;

"use client";
import { forwardRef, useImperativeHandle, useRef, useEffect } from "react";

export interface RecorderHandle {
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<Blob | null>;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
}

interface Props {
  isRecording: boolean;
}

const CameraRecorder = forwardRef<RecorderHandle, Props>(({ isRecording }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  // -----------------------------------
  // START CAMERA
  // -----------------------------------
  const startCamera = async () => {
    if (streamRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true,
    });

    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }
  };

  // -----------------------------------
  // STOP CAMERA
  // -----------------------------------
  const stopCamera = () => {
    if (!streamRef.current) return;

    streamRef.current.getTracks().forEach(t => t.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.load();
    }
  };

  // -----------------------------------
  // START RECORDING
  // -----------------------------------
  const startRecording = async () => {
    if (!streamRef.current) await startCamera();

    chunksRef.current = [];
    mediaRecorderRef.current = new MediaRecorder(streamRef.current!, {
      mimeType: "video/webm",
    });

    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
  };

  // -----------------------------------
  // STOP RECORDING → RETURN VIDEO BLOB
  // -----------------------------------
  const stopRecording = async (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;

      if (!recorder || recorder.state === "inactive")
        return resolve(null);

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        stopCamera();
        resolve(blob);
      };

      recorder.stop();
    });
  };

  useImperativeHandle(ref, () => ({
    startRecording,
    stopRecording,
    startCamera,
    stopCamera,
  }));

  // Actually start MediaRecorder here
  useEffect(() => {
    if (isRecording) {
      mediaRecorderRef.current?.start();
    }
  }, [isRecording]);

  return (
    <video ref={videoRef} autoPlay playsInline muted className="live-video" />
  );
});

export default CameraRecorder;
