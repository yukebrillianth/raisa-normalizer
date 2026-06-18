"use client";

import { useCallback, useMemo, useState } from "react";
import { MicButton } from "@/components/MicButton";
import { TranscriptBlock } from "@/components/TranscriptBlock";
import { AnswerBlock } from "@/components/AnswerBlock";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import type { PipelinePhase } from "@/hooks/usePipelineStream";

type VoicePanelProps = {
  phase: PipelinePhase;
  transcript: string;
  normalizedQuery: string;
  finalAnswer: string;
  spokenAnswer: string;
  audioUrl: string;
  onSubmitAudio: (blob: Blob, mimeType: string) => Promise<{ error: string | null }>;
  onReset: () => void;
};

export function VoicePanel({
  phase,
  transcript,
  normalizedQuery,
  finalAnswer,
  spokenAnswer,
  audioUrl,
  onSubmitAudio,
  onReset,
}: VoicePanelProps) {
  const {
    isRecording,
    isProcessing,
    durationSeconds,
    error: recorderError,
    startRecording,
    stopRecording,
    audioBlob,
    mimeType,
    resetBlob,
    stream,
  } = useAudioRecorder();

  const [submitted, setSubmitted] = useState(false);
  const isRunning = phase === "uploading" || phase === "streaming";
  const hasTranscript = Boolean(transcript || normalizedQuery);
  const hasAnswer = Boolean(finalAnswer || spokenAnswer);

  // Auto-submit when recording stops and blob is ready
  const handleStop = useCallback(() => {
    stopRecording();
  }, [stopRecording]);

  // Watch for audioBlob changes to auto-submit
  const handleStart = useCallback(() => {
    setSubmitted(false);
    onReset();
    resetBlob();
    startRecording();
  }, [onReset, resetBlob, startRecording]);

  // Submit when blob becomes available after recording
  useMemo(() => {
    if (audioBlob && !submitted && !isRecording) {
      setSubmitted(true);
      onSubmitAudio(audioBlob, mimeType);
    }
  }, [audioBlob, submitted, isRecording, onSubmitAudio, mimeType]);

  return (
    <div className="flex flex-col items-center gap-6 py-8 px-4">
      {/* Robot greeting — shown only in idle state */}
      {phase === "idle" && !hasTranscript && (
        <div className="text-center animate-fade-in mb-2">
          <h2 className="font-display text-2xl text-its-cover tracking-tight">
            Halo, ada yang bisa saya bantu?
          </h2>
          <p className="text-sm text-text-muted mt-2">
            Tekan tombol mikrofon dan ajukan pertanyaan Anda
          </p>
        </div>
      )}

      {/* Mic button */}
      <MicButton
        isRecording={isRecording}
        isProcessing={isProcessing}
        isRunning={isRunning}
        durationSeconds={durationSeconds}
        onStart={handleStart}
        onStop={handleStop}
        stream={stream}
      />

      {/* Recorder error */}
      {recorderError && (
        <div
          className="w-full max-w-md text-center text-sm text-error bg-error-soft rounded-lg px-4 py-2"
          data-testid="audio-error"
        >
          {recorderError.message}
        </div>
      )}

      {/* Transcript */}
      <div className="w-full max-w-xl">
        <TranscriptBlock
          transcript={transcript}
          normalizedQuery={normalizedQuery}
          isVisible={hasTranscript}
        />
      </div>

      {/* Answer */}
      <div className="w-full max-w-xl">
        <AnswerBlock
          answer={finalAnswer}
          spokenAnswer={spokenAnswer}
          audioUrl={audioUrl}
          isVisible={hasAnswer}
        />
      </div>

      {/* Done state — prompt for next question */}
      {phase === "done" && (
        <p className="text-xs text-text-muted animate-fade-in mt-2">
          Tekan tombol mikrofon untuk pertanyaan baru
        </p>
      )}
    </div>
  );
}
