// "use client";
// import { useEffect, useRef } from "react";

// interface RecorderProps {
//   isRecording: boolean;
//   onStop: () => void;
//   onRecordingComplete: (blob: Blob) => void;
// }

// export default function Recorder({
//   isRecording,
//   onStop,
//   onRecordingComplete,
// }: RecorderProps) {
//   const videoRef = useRef<HTMLVideoElement>(null);
//   const mediaRecorderRef = useRef<MediaRecorder | null>(null);
//   const recordedChunksRef = useRef<BlobPart[]>([]);
//   const streamRef = useRef<MediaStream | null>(null);

//   useEffect(() => {
//     const startRecording = async () => {
//       try {
//         const stream = await navigator.mediaDevices.getUserMedia({
//           video: {
//             width: { ideal: 1280 },
//             height: { ideal: 720 },
//           },
//           audio: true,
//         });

//         streamRef.current = stream;

//         if (videoRef.current) {
//           videoRef.current.srcObject = stream;
//         }

//         const mediaRecorder = new MediaRecorder(stream, {
//           mimeType: "video/webm;codecs=vp8,opus",
//         });
//         recordedChunksRef.current = [];

//         mediaRecorder.ondataavailable = (event) => {
//           if (event.data.size > 0) recordedChunksRef.current.push(event.data);
//         };

//         mediaRecorder.onstop = () => {
//           const blob = new Blob(recordedChunksRef.current, {
//             type: "video/webm",
//           });
//           onRecordingComplete(blob);

//           // OPTIONAL: auto-download for debugging
//           // const url = URL.createObjectURL(blob);
//           // const a = document.createElement("a");
//           // a.href = url;
//           // a.download = "recorded_avsr_video.webm";
//           // a.click();
//           // URL.revokeObjectURL(url);

//           // Stop camera and mic
//           stream.getTracks().forEach((track) => track.stop());
//         };

//         mediaRecorder.start();
//         mediaRecorderRef.current = mediaRecorder;
//       } catch (error) {
//         console.error("Error accessing camera or microphone:", error);
//       }
//     };

//     const stopRecording = () => {
//       mediaRecorderRef.current?.stop();
//     };

//     if (isRecording) {
//       startRecording();
//     } else if (mediaRecorderRef.current) {
//       stopRecording();
//     }

//     return () => {
//       streamRef.current?.getTracks().forEach((track) => track.stop());
//     };
//   }, [isRecording]);

//   return (
//     <div className="video-container">
//       <video
//         ref={videoRef}
//         autoPlay
//         playsInline
//         muted
//         className="live-video"
//       />
//     </div>
//   );
// }



"use client";
import { useEffect, useRef } from "react";

interface RecorderProps {
  isRecording: boolean;
  onStop: () => void;
  onRecordingComplete: (blob: Blob) => void;
}

export default function Recorder({
  isRecording,
  onStop,
  onRecordingComplete,
}: RecorderProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // --- Utility: hard stop everything ---
  const stopAllMediaTracks = async () => {
    console.log("🛑 Stopping all media tracks...");

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.load();
    }

    // Chromium sometimes holds on to camera streams — force release
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      devices
        .filter((d) => d.kind === "videoinput" || d.kind === "audioinput")
        .forEach((d) => console.log("🎥 Device:", d.label || d.deviceId));
    } catch (err) {
      console.warn("Failed to enumerate devices:", err);
    }
  };

  useEffect(() => {
    const startRecording = async () => {
      try {
        await stopAllMediaTracks(); // ensure clean state

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: true,
        });

        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;

        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: "video/webm;codecs=vp8,opus",
        });

        recordedChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) recordedChunksRef.current.push(event.data);
        };

        mediaRecorder.onstop = async () => {
          const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
          onRecordingComplete(blob);
          if (videoRef.current) {
            videoRef.current.srcObject = null;
            videoRef.current.load(); // flush video element
          }
          
          await stopAllMediaTracks();
          mediaRecorderRef.current = null;
          onStop();
        };

        mediaRecorder.start();
        mediaRecorderRef.current = mediaRecorder;
      } catch (error) {
        console.error("Error accessing camera/microphone:", error);
      }
    };

    const stopRecording = async () => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      } else {
        await stopAllMediaTracks();
      }
    };

    if (isRecording) {
      startRecording();
    } else {
      stopRecording();
    }

    return () => {
      stopAllMediaTracks();
    };
  }, [isRecording, onStop, onRecordingComplete]);

  return (
    <div className="video-container">
      <video ref={videoRef} autoPlay playsInline muted className="live-video" />
    </div>
  );
}
