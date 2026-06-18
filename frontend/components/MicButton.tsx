"use client";

type MicButtonProps = {
  isRecording: boolean;
  isProcessing: boolean;
  isRunning: boolean;
  durationSeconds: number;
  onStart: () => void;
  onStop: () => void;
};

export function MicButton({
  isRecording,
  isProcessing,
  isRunning,
  durationSeconds,
  onStart,
  onStop,
}: MicButtonProps) {
  const disabled = isProcessing || isRunning;

  const handleClick = () => {
    if (disabled) return;
    if (isRecording) {
      onStop();
    } else {
      onStart();
    }
  };

  const formatTime = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const buttonClass = isRecording
    ? "mic-btn mic-btn-recording"
    : disabled
      ? "mic-btn mic-btn-processing"
      : "mic-btn mic-btn-idle";

  const statusText = isRecording
    ? `Merekam ${formatTime(durationSeconds)}`
    : isRunning
      ? "Memproses..."
      : isProcessing
        ? "Menyiapkan audio..."
        : "Tekan untuk bicara";

  return (
    <div className="flex flex-col items-center gap-4">
      <button
        className={buttonClass}
        onClick={handleClick}
        disabled={disabled}
        data-testid="record-button"
        aria-label={isRecording ? "Stop recording" : "Start recording"}
      >
        {isRecording ? (
          /* Stop icon (square) */
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : disabled ? (
          /* Spinner */
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="animate-spin-slow">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.3" />
            <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        ) : (
          /* Microphone icon */
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="2" width="6" height="11" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="19" x2="12" y2="22" />
          </svg>
        )}
      </button>
      <p
        data-testid="recording-status"
        className="text-sm text-text-secondary font-body text-center"
      >
        {statusText}
      </p>
    </div>
  );
}
