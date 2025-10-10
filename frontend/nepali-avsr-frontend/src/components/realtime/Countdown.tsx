"use client";
import { useEffect, useState } from "react";

interface CountdownProps {
  onFinish: () => void;
}

export default function Countdown({ onFinish }: CountdownProps) {
  const [count, setCount] = useState(3);

  useEffect(() => {
    if (count === 0) {
      onFinish();
      return;
    }
    const timer = setTimeout(() => setCount((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [count]);

  return (
    <div className="countdown-overlay">
      <h1>{count}</h1>
    </div>
  );
}
