"use client";

import { AudioRecorder } from "@/components/AudioRecorder";

type RecordControlsProps = {
  onRecordingCompleteAction: (blob: Blob, mimeType: string, durationSeconds: number) => void;
  status: string;
  error?: string;
  isUploading?: boolean;
  isRunning?: boolean;
  resetSignal?: number;
};

export function RecordControls({
  onRecordingCompleteAction,
  status,
  error,
  isUploading = false,
  isRunning = false,
  resetSignal = 0,
}: RecordControlsProps) {
  return (
    <section className="paper-panel rounded-[var(--radius-panel)] p-6">
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr] lg:items-center">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-accent-strong">
            Input suara
          </p>
          <h2 className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-ink">
            Kontrol mikrofon demonstrasi tesis
          </h2>
          <p
            data-testid="recording-status"
            className="mt-3 text-sm leading-6 text-ink-muted"
          >
            Status: {status}
          </p>
          <p
            data-testid="audio-error"
            className={`mt-2 text-sm font-medium ${error ? "text-error" : "text-success"}`}
          >
            {error || "Tidak ada galat audio."}
          </p>
        </div>
        <div
          className={`transition-opacity duration-300 ${
            isUploading || isRunning ? "pointer-events-none opacity-70" : ""
          }`}
        >
          <AudioRecorder
            onRecordingComplete={onRecordingCompleteAction}
            disabled={isUploading || isRunning}
            resetSignal={resetSignal}
          />
        </div>
      </div>
    </section>
  );
}
