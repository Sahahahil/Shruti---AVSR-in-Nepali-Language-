interface TranscriptBoxProps {
  transcript: string;
  isStreaming: boolean;
}

export default function TranscriptBox({ transcript, isStreaming }: TranscriptBoxProps) {
  if (!isStreaming) return null;
  return (
    <div className="transcript-box">
      <h2>Transcription:</h2>
      <p>{transcript || "Listening..."}</p>
    </div>
  );
}
