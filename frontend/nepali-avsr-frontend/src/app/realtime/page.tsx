"use client";
import "@/styles/components/_realtime.scss";
// import { useRef, useState, useEffect } from "react";
import RealtimeContainer from "@/components/realtime/RealtimeContainer";
import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";

export default function RealtimePage() {
    // const videoRef = useRef<HTMLVideoElement | null>(null);
    // const [error, setError] = useState<string | null>(null);
    // const [isStreaming, setIsStreaming] = useState(false);

    // const startStream = async () => {
    //     try {
    //         const stream = await navigator.mediaDevices.getUserMedia({
    //             video: true,
    //             audio: true,
    //         });
    //         if (videoRef.current) {
    //             videoRef.current.srcObject = stream;
    //         }
    //         setIsStreaming(true);
    //         setError(null);
    //     } catch (err) {
    //         console.error("Error accessing media devices:", err);
    //         setError(
    //             "Cannot access camera or microphone.Please check permissions."
    //         );
    //     }
    // };
    // const stopStream = () => {
    //     if (videoRef.current && videoRef.current.srcObject) {
    //         const stream = videoRef.current.srcObject as MediaStream;
    //         stream.getTracks().forEach((track) => track.stop());
    //         videoRef.current.srcObject = null;
    //         setIsStreaming(false);
    //     }
    // };
    // useEffect(() => {
    //     return () => stopStream(); // cleanup when page unmounts
    // }, []);

    // return (
    //     <div className="realtime-container">
    //         <h1>Realtime AVSR</h1>

    //         <div className="video-wrapper">
    //             <video
    //                 ref={videoRef}
    //                 autoPlay
    //                 playsInline
    //                 muted={!isStreaming}
    //             />
    //         </div>

    //         <div className="controls">
    //             {!isStreaming ? (
    //                 <button onClick={startStream}>Start Realtime</button>
    //             ) : (
    //                 <button onClick={stopStream}>Stop Realtime</button>
    //             )}
    //         </div>

    //         {error && <p className="error">{error}</p>}
    //     </div>
    // );

    return (
        <>
            <Navbar></Navbar>
            <RealtimeContainer></RealtimeContainer>
            <Footer></Footer>
        </>
    );
}
