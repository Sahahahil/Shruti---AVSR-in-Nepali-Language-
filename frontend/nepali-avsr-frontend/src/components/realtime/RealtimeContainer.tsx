// "use client";
// import { useState, useRef } from "react";
// import VideoStream from "./VideoStream";
// import AudioStream from "./AudioStream";
// import TranscriptionBox from "./TranscriptionBox";
// import Controls from "./Controls";


// export default function RealtimeContainer() {
//   const [socket, setSocket] = useState<WebSocket | null>(null);
//   const [transcript, setTranscript] = useState("");
//   const [isStreaming, setIsStreaming] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const videoRef = useRef<HTMLVideoElement | null>(null);

//   const startStream = async () => {
//     try {
//       const ws = new WebSocket("ws://localhost:8000/ws/realtime");
//       ws.onmessage = (event) => {
//         const data = JSON.parse(event.data);
//         if (data.transcript) setTranscript(data.transcript);
//       };
//       ws.onerror = () => setError("WebSocket connection error");
//       ws.onclose = () => stopStream();

//       setSocket(ws);
//       setIsStreaming(true);
//     } catch (err) {
//       console.error(err);
//       setError("Could not start stream.");
//     }
//   };

//   const stopStream = () => {
//     if (socket) socket.close();
//     if (videoRef.current && videoRef.current.srcObject) {
//       const stream = videoRef.current.srcObject as MediaStream;
//       stream.getTracks().forEach((t) => t.stop());
//     }
//     setIsStreaming(false);
//   };

//   return (
//     <div className="realtime-container">
//       <h1>Realtime AVSR</h1>

//       <VideoStream videoRef={videoRef} socket={socket} isStreaming={isStreaming} />
//       <AudioStream socket={socket} isStreaming={isStreaming} />

//       <Controls
//         isStreaming={isStreaming}
//         onStart={startStream}
//         onStop={stopStream}
//         error={error}
//       />

//       <TranscriptionBox isStreaming={isStreaming} transcript={transcript} />
//     </div>
//   );
// }

"use client";
import { useState } from "react";
import Recorder from "./Recorder";
import Countdown from "./Countdown";

export default function RealtimeContainer() {
  const [isRecording, setIsRecording] = useState(false);
  const [showCountdown, setShowCountdown] = useState(false);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

  const handleStartRecording = () => {
    setShowCountdown(true);
    setRecordedUrl(null);
  };

  const handleCountdownFinish = () => {
    setShowCountdown(false);
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
  };

  const handleRecordingComplete = (blob: Blob) => {
    const url = URL.createObjectURL(blob);
    setRecordedUrl(url);
    setRecordedBlob(blob);
  };

  const handleSend = () => {
    if (!recordedBlob) return alert("No recording to send yet!");
    console.log("Ready to send file:", recordedBlob);
  };

  return (
    <div className="realtime-container">
      <h1>Realtime AVSR Test</h1>

      <div className="video-container">
        {showCountdown ? (
          <Countdown onFinish={handleCountdownFinish} />
        ) : recordedUrl ? (
          <video
            src={recordedUrl}
            controls
            className="recorded-preview"
          />
        ) : (
          <Recorder
            isRecording={isRecording}
            onStop={handleStopRecording}
            onRecordingComplete={handleRecordingComplete}
          />
        )}
      </div>

      <div className="controls">
        {!isRecording && !showCountdown && !recordedUrl && (
          <button onClick={handleStartRecording}>Start Recording</button>
        )}
        {isRecording && <button onClick={handleStopRecording}>Stop Recording</button>}
        {recordedUrl && (
          <>
            <a href={recordedUrl} download="recording.webm">
              <button>Download</button>
            </a>
            <button onClick={handleSend}>Send</button>
            <button onClick={handleStartRecording}>Record Again</button>
          </>
        )}
      </div>
    </div>
  );
}

// "use client";
// import { useState } from "react";
// import Recorder from "./Recorder";
// import Countdown from "./Countdown";

// export default function RealtimeContainer() {
//   const [isRecording, setIsRecording] = useState(false);
//   const [showCountdown, setShowCountdown] = useState(false);
//   const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
//   const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

//   const handleStartRecording = () => {
//     setShowCountdown(true);
//   };

//   const handleCountdownFinish = () => {
//     setShowCountdown(false);
//     setIsRecording(true);
//   };

//   const handleStopRecording = () => {
//     setIsRecording(false);
//   };

//   const handleRecordingComplete = (blob: Blob) => {
//     const url = URL.createObjectURL(blob);
//     setRecordedUrl(url);
//     setRecordedBlob(blob);
//   };

//   const handleSend = () => {
//     if (!recordedBlob) return alert("No recording to send yet!");
//     // TODO: upload blob to backend later
//     console.log("Ready to send file:", recordedBlob);
//   };

//   return (
//     <div className="realtime-container">
//       <h1>Realtime AVSR Test</h1>

//       {showCountdown ? (
//         <Countdown onFinish={handleCountdownFinish} />
//       ) : (
//         <Recorder
//           isRecording={isRecording}
//           onStop={handleStopRecording}
//           onRecordingComplete={handleRecordingComplete}
//         />
//       )}

//       <div className="controls">
//         {!isRecording && !showCountdown && (
//           <button onClick={handleStartRecording}>Start Recording</button>
//         )}
//         {isRecording && <button onClick={handleStopRecording}>Stop Recording</button>}
//         {recordedUrl && (
//           <>
//             <a href={recordedUrl} download="recording.webm">
//               <button>Download</button>
//             </a>
//             <button onClick={handleSend}>Send</button>
//           </>
//         )}
//       </div>

//       {recordedUrl && (
//         <video src={recordedUrl} controls className="recorded-preview" />
//       )}
//     </div>
//   );
// }
