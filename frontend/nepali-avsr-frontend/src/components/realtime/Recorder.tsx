// "use client";
// import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";

// interface RecorderProps {
//     isRecording: boolean;
//     onStop: () => void;
//     onRecordingComplete: (blob: Blob) => void;
// }

// export interface RecorderHandle {
//     stop: () => void;
// }

// const Recorder = forwardRef<RecorderHandle, RecorderProps>(
//     ({ isRecording, onStop, onRecordingComplete }, ref) => {
//         const videoRef = useRef<HTMLVideoElement>(null);
//         const mediaRecorderRef = useRef<MediaRecorder | null>(null);
//         const recordedChunksRef = useRef<BlobPart[]>([]);
//         const streamRef = useRef<MediaStream | null>(null);

//         // Expose stop() method to parent
//         useImperativeHandle(ref, () => ({
//             stop: () => {
//                 const recorder = mediaRecorderRef.current;
//                 if (recorder && recorder.state !== "inactive") {
//                     recorder.stop();
//                 }
//             },
//         }));

//         const stopAllMediaTracks = async () => {
//             if (streamRef.current) {
//                 streamRef.current.getTracks().forEach((track) => track.stop());
//                 streamRef.current = null;
//             }

//             if (videoRef.current) {
//                 videoRef.current.pause();
//                 videoRef.current.srcObject = null;
//                 videoRef.current.load();
//             }
//         };

//         const startRecording = async () => {
//             if (streamRef.current) return; // already recording

//             try {
//                 const stream = await navigator.mediaDevices.getUserMedia({
//                     video: { width: { ideal: 1280 }, height: { ideal: 720 } },
//                     audio: true,
//                 });

//                 streamRef.current = stream;
//                 if (videoRef.current) videoRef.current.srcObject = stream;

//                 const mediaRecorder = new MediaRecorder(stream, {
//                     mimeType: "video/webm;codecs=vp8,opus",
//                 });

//                 recordedChunksRef.current = [];

//                 mediaRecorder.ondataavailable = (event) => {
//                     if (event.data.size > 0) recordedChunksRef.current.push(event.data);
//                 };

//                 mediaRecorder.onstop = async () => {
//                     const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
//                     onRecordingComplete(blob);
//                     await stopAllMediaTracks();
//                     mediaRecorderRef.current = null;
//                     onStop();
//                 };

//                 mediaRecorder.start();
//                 mediaRecorderRef.current = mediaRecorder;
//             } catch (err) {
//                 console.error("Error accessing camera/microphone:", err);
//             }
//         };

//         const stopRecording = async () => {
//             const recorder = mediaRecorderRef.current;
//             if (recorder && recorder.state !== "inactive") {
//                 recorder.stop();
//             } else {
//                 await stopAllMediaTracks();
//             }
//         };

//         useEffect(() => {
//             if (isRecording) startRecording();
//             else stopRecording();

//             return () => {
//                 stopAllMediaTracks().catch(console.error);
//             };
//             // eslint-disable-next-line react-hooks/exhaustive-deps
//         }, [isRecording]);

//         return (
//             <div className="video-container">
//                 <video ref={videoRef} autoPlay playsInline muted className="live-video" />
//             </div>
//         );
//     }
// );

// export default Recorder;


"use client";
import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";

interface RecorderProps {
    isRecording: boolean;
    onStop: () => void;
    onRecordingComplete: (blob: Blob) => void;
}

export interface RecorderHandle {
    stop: () => void;
    startCamera: () => void;
    stopCamera: () => void;
}

const Recorder = forwardRef<RecorderHandle, RecorderProps>(
    ({ isRecording, onStop, onRecordingComplete }, ref) => {
        const videoRef = useRef<HTMLVideoElement>(null);
        const mediaRecorderRef = useRef<MediaRecorder | null>(null);
        const recordedChunksRef = useRef<BlobPart[]>([]);
        const streamRef = useRef<MediaStream | null>(null);

        // Expose methods to parent
        useImperativeHandle(ref, () => ({
            stop: () => {
                const recorder = mediaRecorderRef.current;
                if (recorder && recorder.state !== "inactive") {
                    recorder.stop();
                }
            },
            startCamera: async () => {
                if (!streamRef.current) await startRecording(true);
            },
            stopCamera: async () => {
                await stopAllMediaTracks();
            },
        }));

        const stopAllMediaTracks = async () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
                streamRef.current = null;
            }
            if (videoRef.current) {
                videoRef.current.pause();
                videoRef.current.srcObject = null;
                videoRef.current.load();
            }
        };

        const startRecording = async (cameraOnly = false) => {
            if (streamRef.current) return;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: cameraOnly ? false : true,
                });

                streamRef.current = stream;
                if (videoRef.current) videoRef.current.srcObject = stream;

                if (!cameraOnly) {
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
                        await stopAllMediaTracks();
                        mediaRecorderRef.current = null;
                        onStop();
                    };

                    mediaRecorder.start();
                    mediaRecorderRef.current = mediaRecorder;
                }
            } catch (err) {
                console.error("Error accessing camera/microphone:", err);
            }
        };

        const stopRecording = async () => {
            const recorder = mediaRecorderRef.current;
            if (recorder && recorder.state !== "inactive") {
                recorder.stop();
            } else {
                await stopAllMediaTracks();
            }
        };

        useEffect(() => {
            if (isRecording) startRecording();
            else stopRecording();

            return () => {
                stopAllMediaTracks().catch(console.error);
            };
        }, [isRecording]);

        return (
            <div className="video-container">
                <video ref={videoRef} autoPlay playsInline muted className="live-video" />
            </div>
        );
    }
);

export default Recorder;
