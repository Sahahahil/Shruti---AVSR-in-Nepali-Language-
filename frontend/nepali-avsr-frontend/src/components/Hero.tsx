"use client";
import { useState } from "react";

import GetStartedModal from "./GetStartedModal";
import "@/styles/components/_hero.scss"

export default function Hero(){
    const[showPopUp, setShowPopUp]=useState(false);
    return(
        <section className="hero">
            <div className="hero-content">
                <h1>Nepali Audio-Video Speech Recognition</h1>
                <p> Experience real-time and offline Nepali speech recognition using
                cutting-edge AI models.</p>
                <button onClick={()=>setShowPopUp(true)}> Get Started </button>
            </div>
            {showPopUp&& <GetStartedModal onClose={()=> setShowPopUp(false)}></GetStartedModal>}
        </section>
    )
}