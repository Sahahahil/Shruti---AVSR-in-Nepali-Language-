// "use client";
// import { useRef, useState } from "react";
// import CameraRecorder, { RecorderHandle } from "./CameraRecorder";
// import Countdown from "./Countdown";

// export default function RealtimeRecorder() {
//     const recorderRef = useRef<RecorderHandle>(null);
//     // const [showPopup, setShowPopup] = useState(false);
//     const [cameraOn, setCameraOn] = useState(false);
//     const [countdown, setCountdown] = useState(false);
//     const [isRecording, setIsRecording] = useState(false);
//     const [previewUrl, setPreviewUrl] = useState<string | null>(null);
//     const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

//     // ---- CAMERA TOGGLE ----
//     const handleToggleCamera = async () => {
//         if (cameraOn) {
//             recorderRef.current?.stopCamera();
//             setCameraOn(false);
//         } else {
//             await recorderRef.current?.startCamera();
//             setCameraOn(true);
//         }
//     };

//     // ---- START RECORDING ----
//     const startRecording = async () => {
//         if (!cameraOn) {
//             alert("Please turn on the camera first.");
//             return;
//         }
//         if (!cameraOn) {
//             await recorderRef.current?.startCamera();
//             setCameraOn(true);
//         }
//         setPreviewUrl(null);
//         setRecordedBlob(null);
//         setCountdown(true);
//     };
//     const retakeRecording = async () => {
//         // Auto turn on camera if off
//         if (!cameraOn) {
//             await recorderRef.current?.startCamera();
//             setCameraOn(true);
//         }

//         setPreviewUrl(null);
//         setRecordedBlob(null);
//         setCountdown(true);
//     };

//     const onCountdownFinish = async () => {
//         setCountdown(false);
//         await recorderRef.current?.startRecording();
//         setIsRecording(true);
//     };

//     // ---- STOP RECORDING ----
//     const stopRecording = async () => {
//         setIsRecording(false);
//         const blob = await recorderRef.current?.stopRecording();
//         if (blob) {
//             setRecordedBlob(blob);
//             setPreviewUrl(URL.createObjectURL(blob));
//         }
//         setCameraOn(false);
//     };

//     // ---- SEND TO BACKEND ----
//     const sendToBackend = () => {
//         if (!recordedBlob) return;
//         console.log("Sending video:", recordedBlob);
//     };

//     return (
//         <div className="realtime-container">
//             <h1>Realtime AVSR Recording</h1>

//             {/* VIDEO AREA */}
//             <div className="video-container">
//                 {countdown && <Countdown onFinish={onCountdownFinish} />}

//                 {!previewUrl && (
//                     <CameraRecorder
//                         ref={recorderRef}
//                         isRecording={isRecording}
//                     />
//                 )}

//                 {previewUrl && (
//                     <video
//                         src={previewUrl}
//                         controls
//                         className="preview-video"
//                     />
//                 )}
//             </div>

//             {/* CONTROLS */}
//             <div className="controls">
//                 {!previewUrl && !isRecording && !countdown && (
//                     <>
//                         <button onClick={handleToggleCamera}>
//                             {cameraOn ? "Turn Camera Off" : "Turn Camera On"}
//                         </button>

//                         <button onClick={startRecording}>
//                             Start Recording
//                         </button>
//                     </>
//                 )}

//                 {isRecording && (
//                     <button onClick={stopRecording}>Stop Recording</button>
//                 )}

//                 {previewUrl && (
//                     <>
//                         <a href={previewUrl} download="recording.webm">
//                             <button>Download</button>
//                         </a>

//                         <button onClick={sendToBackend}>Send</button>

//                         <button onClick={retakeRecording}>Retake</button>
//                     </>
//                 )}
//             </div>
//         </div>
//     );
// }

"use client";
import { useRef, useState } from "react";
import CameraRecorder, { RecorderHandle } from "./CameraRecorder";
import Countdown from "./Countdown";

export default function RealtimeRecorder() {
    const recorderRef = useRef<RecorderHandle>(null);

    const [cameraOn, setCameraOn] = useState(false);
    const [countdown, setCountdown] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

    // -----------------------------
    // TOGGLE CAMERA
    // -----------------------------
    const handleToggleCamera = async () => {
        if (cameraOn) {
            recorderRef.current?.stopCamera();
            setCameraOn(false);
        } else {
            await recorderRef.current?.startCamera();
            setCameraOn(true);
        }
    };

    // -----------------------------
    // START RECORDING (NO AUTO CAMERA)
    // -----------------------------
    const startRecording = () => {
        if (!cameraOn) {
            alert("Please turn on the camera first.");
            return;
        }

        setPreviewUrl(null);
        setRecordedBlob(null);
        setCountdown(true);
    };

    // -----------------------------
    // RETAKE (AUTO TURN CAMERA ON)
    // -----------------------------
    const retakeRecording = async () => {
        await recorderRef.current?.startCamera();
        setCameraOn(true);

        setPreviewUrl(null);
        setRecordedBlob(null);
        setCountdown(true);
    };

    const onCountdownFinish = async () => {
        setCountdown(false);
        await recorderRef.current?.startRecording();
        setIsRecording(true);
    };

    // -----------------------------
    // STOP RECORD (GENERATE FULL VIDEO)
    // -----------------------------
    const stopRecording = async () => {
        setIsRecording(false);
        const blob = await recorderRef.current?.stopRecording();

        if (blob) {
            setRecordedBlob(blob);
            setPreviewUrl(URL.createObjectURL(blob));
        }

        setCameraOn(false);
    };

    // -----------------------------
    // SEND
    // -----------------------------
    const sendToBackend = () => {
        if (!recordedBlob) return;
        console.log("Sending video:", recordedBlob);
    };

    return (
        <div className="realtime-container">
            <h1>Realtime AVSR Recording</h1>

            <div className="video-container">
                {countdown && <Countdown onFinish={onCountdownFinish} />}

                {!previewUrl && (
                    <CameraRecorder
                        ref={recorderRef}
                        isRecording={isRecording}
                    />
                )}

                {previewUrl && (
                    <video
                        src={previewUrl}
                        controls
                        autoPlay
                        playsInline
                        className="preview-video"
                    />
                )}
            </div>

            <div className="controls">
                {!previewUrl && !isRecording && !countdown && (
                    <>
                        <button onClick={handleToggleCamera}>
                            {cameraOn ? "Turn Camera Off" : "Turn Camera On"}
                        </button>

                        <button onClick={startRecording}>
                            Start Recording
                        </button>
                    </>
                )}

                {isRecording && (
                    <button onClick={stopRecording}>Stop Recording</button>
                )}

                {previewUrl && (
                    <>
                        <a href={previewUrl} download="recording.webm">
                            <button>Download</button>
                        </a>

                        <button onClick={sendToBackend}>Send</button>

                        <button onClick={retakeRecording}>Retake</button>
                    </>
                )}
            </div>
        </div>
    );
}
