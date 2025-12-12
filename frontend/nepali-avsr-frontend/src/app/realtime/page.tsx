"use client";
import "@/styles/components/_realtime.scss";
// import { useRef, useState, useEffect } from "react";
import RealtimeContainer from "@/components/realtime/RealtimeContainer";
import RealtimeRecorder from "@/components/realtime_1/RealtimeRecorder";
import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";

export default function RealtimePage() {
 

    return (
        <>
            <Navbar></Navbar>
            {/* <RealtimeContainer></RealtimeContainer> */}
            <RealtimeRecorder></RealtimeRecorder>
            <Footer></Footer>
        </>
    );
}
