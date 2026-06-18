"use client";

import { useEffect, useRef } from "react";

type MicButtonProps = {
  isRecording: boolean;
  isProcessing: boolean;
  isRunning: boolean;
  durationSeconds: number;
  onStart: () => void;
  onStop: () => void;
  stream: MediaStream | null;
};

export function MicButton({
  isRecording,
  isProcessing,
  isRunning,
  durationSeconds,
  onStart,
  onStop,
  stream,
}: MicButtonProps) {
  const disabled = isProcessing || isRunning;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

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

  // Real-time audio analysis and visualizer draw inside the button
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Handle high-DPI retina screens
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    if (!isRecording || !stream) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      ctx.clearRect(0, 0, width, height);
      return;
    }

    try {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioContextClass();
      const analyser = audioCtx.createAnalyser();
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      analyser.fftSize = 128;
      analyserRef.current = analyser;
      audioCtxRef.current = audioCtx;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      let phase = 0;

      const draw = () => {
        if (!analyserRef.current) return;
        animationRef.current = requestAnimationFrame(draw);

        analyserRef.current.getByteTimeDomainData(dataArray);

        // Calculate voice volume level (RMS)
        let total = 0;
        for (let i = 0; i < bufferLength; i++) {
          const v = (dataArray[i] - 128) / 128;
          total += v * v;
        }
        const rms = Math.sqrt(total / bufferLength);
        const volume = Math.min(rms * 5.0, 1.0); // Boost gain for visual effect

        ctx.clearRect(0, 0, width, height);

        // Draw Siri/Google Assistant styled wave bands in absolute white inside the button
        const drawWave = (
          color: string,
          opac: number,
          lineWidth: number,
          speed: number,
          freq: number,
          ampScale: number
        ) => {
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.globalAlpha = opac;
          ctx.lineWidth = lineWidth;

          const midY = height / 2;
          const maxAmp = (height / 2.3) * (volume + 0.05) * ampScale;

          for (let x = 0; x < width; x++) {
            const angle = (x / width) * Math.PI * 2 * freq + phase * speed;
            // Envelope to clamp edges smoothly
            const envelope = Math.sin((x / width) * Math.PI);
            const y = midY + Math.sin(angle) * maxAmp * envelope;

            if (x === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        };

        // Draw 3 layered organic sines in white for high visibility
        drawWave("#FFFFFF", 0.90, 2.5, 3.2, 1.8, 1.0);
        drawWave("#FFFFFF", 0.45, 1.5, -2.4, 2.6, 0.75);
        drawWave("#FFFFFF", 0.20, 1.0, 1.5, 3.2, 0.45);

        phase += 0.05;
      };

      draw();
    } catch (err) {
      console.error("Audio visualizer error inside MicButton:", err);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, [stream, isRecording]);

  const buttonClass = isRecording
    ? "mic-btn mic-btn-recording overflow-hidden"
    : disabled
      ? "mic-btn mic-btn-processing"
      : "mic-btn mic-btn-idle";

  const statusText = isRecording
    ? `Merekam — ${formatTime(durationSeconds)}`
    : isRunning
      ? "Memproses..."
      : isProcessing
        ? "Menyiapkan..."
        : "Ketuk untuk bicara";

  return (
    <div className="flex flex-col items-center gap-4">
      <button
        className={buttonClass}
        onClick={handleClick}
        disabled={disabled}
        data-testid="record-button"
        aria-label={isRecording ? "Stop recording" : "Start recording"}
      >
        {/* Real-time wave canvas (renders behind foreground stop icon) */}
        {isRecording && (
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none rounded-full"
            style={{ width: "100%", height: "100%" }}
          />
        )}

        {/* Foreground Icon */}
        <div className="relative z-10 flex items-center justify-center text-white">
          {isRecording ? (
            /* Clean stop square */
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : disabled ? (
            /* Spinner */
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="animate-spin-slow text-text-muted">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" opacity="0.3" />
              <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
          ) : (
            /* Microphone icon */
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
              <rect x="9" y="2" width="6" height="11" rx="3" />
              <path d="M5 10a7 7 0 0 0 14 0" />
              <line x1="12" y1="19" x2="12" y2="22" />
            </svg>
          )}
        </div>
      </button>

      <p
        data-testid="recording-status"
        className="text-xs text-text-secondary font-body text-center font-bold tracking-widest uppercase mt-1"
      >
        {statusText}
      </p>
    </div>
  );
}
