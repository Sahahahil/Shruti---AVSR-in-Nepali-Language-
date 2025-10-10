interface ControlsProps {
  isStreaming: boolean;
  onStart: () => void;
  onStop: () => void;
  error: string | null;
}

export default function Controls({ isStreaming, onStart, onStop, error }: ControlsProps) {
  return (
    <div className="controls">
      {!isStreaming ? (
        <button onClick={onStart}>Start Realtime</button>
      ) : (
        <button onClick={onStop}>Stop Realtime</button>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
