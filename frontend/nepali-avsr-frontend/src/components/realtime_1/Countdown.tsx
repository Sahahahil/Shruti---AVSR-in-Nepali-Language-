"use client";
import { useEffect, useState } from "react";

export default function Countdown({ onFinish }: { onFinish: () => void }) {
  const [count, setCount] = useState(3);

  useEffect(() => {
    if (count === 0) {
      onFinish();
      return;
    }
    const t = setTimeout(() => setCount(c => c - 1), 1000);
    return () => clearTimeout(t);
  }, [count]);

  return (
    <div className="countdown-overlay">
      <h1>{count}</h1>
    </div>
  );
}
