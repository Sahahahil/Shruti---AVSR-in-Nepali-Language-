"use client";
import { useState, useRef, useEffect } from "react";
import Recorder, { RecorderHandle } from "./Recorder";
import Countdown from "./Countdown";

export default function RealtimeContainer() {
    const [mounted, setMounted] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [showCountdown, setShowCountdown] = useState(false);
    const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
    const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
    const [cameraOn, setCameraOn] = useState(false);

    const recorderRef = useRef<RecorderHandle>(null);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    const handleStartRecording = () => {
        setShowCountdown(true);
        setRecordedUrl(null);
        setRecordedBlob(null);
    };

    const handleCountdownFinish = () => {
        setShowCountdown(false);
        setIsRecording(true);
    };

    const handleStopRecording = async () => {
        recorderRef.current?.stop(); // stops MediaRecorder
        await recorderRef.current?.stopCamera(); // ensures camera is off
        setIsRecording(false);
        setCameraOn(false);
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

    const handleToggleCamera = async () => {
        if (cameraOn) {
            await recorderRef.current?.stopCamera();
            setCameraOn(false);
        } else {
            await recorderRef.current?.startCamera();
            setCameraOn(true);
        }
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
                        ref={recorderRef}
                        isRecording={isRecording}
                        onStop={handleStopRecording}
                        onRecordingComplete={handleRecordingComplete}
                    />
                )}
            </div>

            <div className="controls">
                {/* <button onClick={handleToggleCamera}>
                    {cameraOn ? "Turn Camera Off" : "Turn Camera On"}
                </button> */}

                <button
                    onClick={handleToggleCamera}
                    disabled={isRecording || recordedUrl !== null}
                >
                    {cameraOn ? "Turn Camera Off" : "Turn Camera On"}
                </button>
                {!isRecording && !showCountdown && !recordedUrl && (
                    <button onClick={handleStartRecording}>
                        Start Recording
                    </button>
                )}
                {isRecording && (
                    <button onClick={handleStopRecording}>
                        Stop Recording
                    </button>
                )}
                {recordedUrl && (
                    <>
                        <a href={recordedUrl} download="recording.webm">
                            <button>Download</button>
                        </a>
                        <button onClick={handleSend}>Send</button>
                        <button onClick={handleStartRecording}>
                            Record Again
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}
