"use client";
import Link from "next/link";
import "@/styles/components/_popup.scss";

export default function GetStartedModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="popup-overlay">
      <div className="popup">
        <h2>Select Mode</h2>
        <div className="buttons">
          <Link href="/realtime" className="btn">Realtime AVSR</Link>
          <Link href="/offline" className="btn">Offline AVSR</Link>
        </div>
        <button className="close" onClick={onClose}>✖</button>
      </div>
    </div>
  );
}
