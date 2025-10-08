"use client";
import { useState } from "react";


import GetStartedModal from "./GetStartedModal";
import "@/styles/components/_hero.scss"

export default function Hero(){
    const[showPopup, setShowPopup]=useState(false);
    return (
    <section className="hero">
      {/* Background Video */}
      <video className="hero-video" autoPlay muted loop>
        <source src="/videos/hero.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>


      {/* Dark Overlay */}
      <div className="hero-overlay"></div>

      {/* Overlay Content */}
      <div className="hero-content">
        <h1>Nepali Audio-Visual Speech Recognition</h1>
        <p>
          Experience real-time and offline Nepali speech recognition using
          cutting-edge AI models.
        </p>
        <button onClick={() => setShowPopup(true)}>Get Started</button>
      </div>

      {showPopup && <GetStartedModal onClose={() => setShowPopup(false)} />}
    </section>
  );
}