"use client";

import type { UseAudioRecorderReturn } from "@/hooks/useAudioRecorder";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import type { KeyboardEvent, PointerEvent } from "react";
import { useEffect, useMemo, useRef } from "react";

interface AudioRecorderProps {
  onRecordingComplete?: (blob: Blob, mimeType: string, durationSeconds: number) => void;
  maxDurationDisplay?: number;
  disabled?: boolean;
  resetSignal?: number;
}

export function AudioRecorder({
  onRecordingComplete,
  maxDurationDisplay = 30,
  disabled = false,
  resetSignal = 0,
}: AudioRecorderProps) {
  const {
    audioBlob,
    mimeType,
    durationSeconds,
    isRecording,
    isProcessing,
    error,
    isSupported,
    startRecording,
    stopRecording,
    resetBlob,
  }: UseAudioRecorderReturn = useAudioRecorder();

  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isHoldGestureRef = useRef(false);

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (disabled) return;
      if (!isRecording) startRecording();
    }
  };

  const handleKeyUp = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (disabled) return;
      if (isRecording) stopRecording();
    }
  };

  const audioUrl = useMemo(() => {
    if (!audioBlob) return null;
    return URL.createObjectURL(audioBlob);
  }, [audioBlob]);

  useEffect(() => {
    if (audioBlob && !isRecording && onRecordingComplete) {
      onRecordingComplete(audioBlob, mimeType, durationSeconds);
    }
  }, [audioBlob, durationSeconds, isRecording, mimeType, onRecordingComplete]);

  useEffect(() => {
    if (!isRecording) resetBlob();
  }, [isRecording, resetBlob, resetSignal]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const handlePointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (disabled) return;
    if (isRecording) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    isHoldGestureRef.current = false;
    holdTimerRef.current = setTimeout(() => {
      isHoldGestureRef.current = true;
      startRecording();
    }, 180);
  };

  const handlePointerUp = () => {
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    if (isHoldGestureRef.current && isRecording) {
      stopRecording();
    }
  };

  const handleClick = () => {
    if (disabled) return;
    if (isHoldGestureRef.current) {
      isHoldGestureRef.current = false;
      return;
    }
    if (isRecording) stopRecording();
    else startRecording();
  };

  const handleReset = () => {
    resetBlob();
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Unsupported browser state
  if (!isSupported) {
    return (
      <div className="flex flex-col items-center gap-4 rounded-[var(--radius-card)] border border-error bg-error-soft p-6">
        <svg className="h-12 w-12 text-error" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p className="text-center text-sm text-error">
          Browser tidak mendukung perekaman audio. Gunakan Chrome, Firefox, atau Edge terbaru.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-6 rounded-[var(--radius-card)] border border-line bg-background/70 p-8">
      <button
        data-testid="record-button"
        type="button"
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        disabled={!isSupported || disabled}
        className={`relative flex h-24 w-24 items-center justify-center rounded-full border transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 focus:ring-accent-soft disabled:cursor-not-allowed disabled:opacity-50 ${
          isRecording
            ? "scale-110 border-error bg-error shadow-[0_18px_42px_color-mix(in_srgb,var(--error)_32%,transparent)]"
            : "border-accent-strong bg-accent shadow-[0_14px_35px_color-mix(in_srgb,var(--accent-strong)_25%,transparent)] hover:scale-105 hover:bg-accent-strong"
        }`}
        aria-label={
          disabled
            ? "Pipeline sedang berjalan"
            : isRecording
              ? "Hentikan rekaman"
              : "Mulai rekaman"
        }
      >
        {disabled ? (
          <span className="pointer-events-none absolute inset-0 rounded-full border border-info bg-info-soft/60" />
        ) : null}
        {isRecording ? (
          <>
            <div className="h-8 w-8 rounded-sm bg-surface" />
            <span className="absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap text-xs font-medium text-ink-muted">
              Lepaskan untuk berhenti
            </span>
          </>
        ) : (
          <>
            <svg className="h-10 w-10 text-surface" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            <span className="absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap text-xs font-medium text-ink-muted">
              {disabled ? "Pipeline berjalan" : "Tahan untuk rekam"}
            </span>
          </>
        )}
      </button>

      {(isRecording || durationSeconds > 0) && (
        <div className="flex flex-col items-center gap-2 mt-8">
          <div className="text-3xl font-semibold text-ink">
            {formatDuration(durationSeconds)}
          </div>
          {isRecording ? (
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-error" />
              <span className="text-sm text-ink-muted">Sedang merekam...</span>
            </div>
          ) : null}
        </div>
      )}

      {isRecording && durationSeconds > 0 && (
        <div className="w-full max-w-xs">
          <div className="h-1 overflow-hidden rounded-full bg-surface-muted">
            <div
              className="h-full bg-error transition-all duration-1000 ease-linear"
              style={{ width: `${(durationSeconds / maxDurationDisplay) * 100}%` }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-xs text-ink-muted">Sisa {Math.max(0, maxDurationDisplay - durationSeconds)}d</span>
          </div>
        </div>
      )}

      {audioBlob && audioUrl && !isRecording && (
        <div className="flex flex-col items-center gap-4 mt-4">
          <audio controls className="w-full max-w-xs rounded-2xl border border-line bg-surface-strong p-2">
            <source src={audioUrl} type={mimeType} />
            Browser tidak mendukung pemutaran audio.
          </audio>
          <button
            type="button"
            onClick={handleReset}
            className="text-sm text-ink-muted underline hover:text-ink"
          >
            Rekam ulang
          </button>
        </div>
      )}

      {isProcessing && (
        <div className="flex flex-col items-center gap-2">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-info border-t-transparent" />
          <p className="text-sm text-ink-muted">Memproses audio...</p>
        </div>
      )}

      {error && (
        <div className="flex max-w-md items-start gap-3 rounded-[var(--radius-card)] border border-error bg-error-soft p-4">
          <svg className="mt-0.5 h-5 w-5 flex-shrink-0 text-error" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="flex-1">
            <p className="text-sm font-medium text-error">Galat rekaman</p>
            <p className="mt-1 text-sm text-error">{error.message}</p>
          </div>
        </div>
      )}

      {!isRecording && !audioBlob && !error && (
        <p className="mt-8 max-w-xs text-center text-sm text-ink-muted">
          {disabled
            ? "Tunggu pipeline selesai sebelum mengirim pertanyaan baru."
            : `Tekan dan tahan tombol mikrofon untuk merekam. Durasi maksimum: ${maxDurationDisplay} detik.`}
        </p>
      )}
    </div>
  );
}
