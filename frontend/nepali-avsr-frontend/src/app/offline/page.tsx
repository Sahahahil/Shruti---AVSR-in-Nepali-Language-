"use client";
import "@/styles/components/_realtime.scss";
// import { useRef, useState, useEffect } from "react";
// import RealtimeContainer from "@/components/realtime/RealtimeContainer";
import FileUploader from "@/components/offline/FileUploader";
import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";

export default function OfflinetimePage() {
 

    return (
        <>
            <Navbar></Navbar>
            <FileUploader></FileUploader>
            <Footer></Footer>
        </>
    );
}
