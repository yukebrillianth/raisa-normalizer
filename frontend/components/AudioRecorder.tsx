"use client";

import { useEffect, useMemo, useRef } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import type { UseAudioRecorderReturn } from "@/hooks/useAudioRecorder";

interface AudioRecorderProps {
  onRecordingComplete?: (blob: Blob, mimeType: string, durationSeconds: number) => void;
  maxDurationDisplay?: number;
}

export function AudioRecorder({
  onRecordingComplete,
  maxDurationDisplay = 30,
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
      if (!isRecording) startRecording();
    }
  };

  const handleKeyUp = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
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
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const handleStartRecording = async () => {
    await startRecording();
  };

  const handleStopRecording = () => {
    stopRecording();
  };

  const handlePointerDown = (e: PointerEvent<HTMLButtonElement>) => {
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
      <div className="flex flex-col items-center gap-4 p-6 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800">
        <svg className="w-12 h-12 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p className="text-sm text-red-700 dark:text-red-300 text-center">
          Your browser does not support audio recording. Please use a modern browser like Chrome, Firefox, or Edge.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-6 p-8 rounded-lg bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
      {/* Recording Button */}
      <button
        type="button"
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        disabled={!isSupported}
        className={`
          relative flex items-center justify-center w-24 h-24 rounded-full
          transition-all duration-200 ease-in-out
          ${isRecording
            ? "bg-red-500 hover:bg-red-600 scale-110 shadow-lg shadow-red-500/50"
            : "bg-blue-500 hover:bg-blue-600 hover:scale-105"
          }
          disabled:opacity-50 disabled:cursor-not-allowed
          focus:outline-none focus:ring-4 focus:ring-blue-500/30
        `}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
      >
        {isRecording ? (
          <>
            <div className="w-8 h-8 bg-white rounded-sm" />
            <span className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs font-medium text-zinc-600 dark:text-zinc-400 whitespace-nowrap">
              Release to stop
            </span>
          </>
        ) : (
          <>
            <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            <span className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs font-medium text-zinc-600 dark:text-zinc-400 whitespace-nowrap">
              Hold to record
            </span>
          </>
        )}
      </button>

      {/* Duration Display */}
      {(isRecording || durationSeconds > 0) && (
        <div className="flex flex-col items-center gap-2 mt-8">
          <div className="text-3xl font-mono font-semibold text-zinc-900 dark:text-zinc-50">
            {formatDuration(durationSeconds)}
          </div>
          {isRecording && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm text-zinc-600 dark:text-zinc-400">Recording...</span>
            </div>
          )}
        </div>
      )}

      {/* Countdown indicator when recording */}
      {isRecording && durationSeconds > 0 && (
        <div className="w-full max-w-xs">
          <div className="h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-red-500 transition-all duration-1000 ease-linear"
              style={{ width: `${(durationSeconds / maxDurationDisplay) * 100}%` }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-xs text-zinc-500 dark:text-zinc-500">
              {maxDurationDisplay - durationSeconds}s remaining
            </span>
          </div>
        </div>
      )}

      {/* Audio Preview */}
      {audioBlob && audioUrl && !isRecording && (
        <div className="flex flex-col items-center gap-4 mt-4">
          <audio controls className="w-full max-w-xs">
            <source src={audioUrl} type={mimeType} />
            Your browser does not support audio playback.
          </audio>
          <button
            type="button"
            onClick={handleReset}
            className="text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 underline"
          >
            Record again
          </button>
        </div>
      )}

      {/* Processing Display */}
      {isProcessing && (
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Processing audio...</p>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 max-w-md">
          <svg className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="flex-1">
            <p className="text-sm font-medium text-red-900 dark:text-red-100">Recording Error</p>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error.message}</p>
          </div>
        </div>
      )}

      {/* Processing State */}
      {isProcessing && (
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Processing audio...</p>
        </div>
      )}

      {/* Help Text */}
      {!isRecording && !audioBlob && !error && (
        <p className="text-sm text-zinc-500 dark:text-zinc-500 text-center max-w-xs mt-8">
          Press and hold the microphone button to start recording. Release to stop. Maximum duration: {maxDurationDisplay} seconds.
        </p>
      )}
    </div>
  );
}
