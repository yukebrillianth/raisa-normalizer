import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderError =
  | "unsupported"
  | "no_microphone"
  | "permission_denied"
  | "unknown";

export interface UseAudioRecorderReturn {
  audioBlob: Blob | null;
  mimeType: string;
  durationSeconds: number;
  isRecording: boolean;
  isProcessing: boolean;
  error: { type: RecorderError; message: string } | null;
  isSupported: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  resetBlob: () => void;
}

const PREFERRED_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg",
];

function getMaxDuration(): number {
  const raw = process.env.NEXT_PUBLIC_MAX_RECORDING_SECONDS;
  if (raw) {
    const parsed = parseInt(raw, 10);
    if (!isNaN(parsed) && parsed > 0) return parsed;
  }
  return 30; // default
}

function getMimeType(): string {
  for (const mt of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(mt)) return mt;
  }
  return ""; // fallback to browser default
}

export function useAudioRecorder(): UseAudioRecorderReturn {
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [mimeType, setMimeType] = useState<string>("");
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<UseAudioRecorderReturn["error"]>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedRef = useRef(0);

  const isSupported =
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices &&
    !!navigator.mediaDevices.getUserMedia &&
    typeof MediaRecorder !== "undefined";

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const stopRecording = useCallback(() => {
    clearTimer();
    setDurationSeconds(elapsedRef.current);
    if (recorderRef.current?.state === "recording") {
      setIsProcessing(true);
      recorderRef.current.stop();
    }
    setIsRecording(false);
  }, [clearTimer]);

  const startRecording = useCallback(async () => {
    setError(null);
    setAudioBlob(null);
    setIsProcessing(false);
    setDurationSeconds(0);
    chunksRef.current = [];
    elapsedRef.current = 0;

    // Browser compatibility check
    if (!isSupported) {
      setError({ type: "unsupported", message: "Your browser does not support audio recording." });
      return;
    }

    // MIME detection
    const selectedMime = getMimeType();
    setMimeType(selectedMime);

    // Request microphone
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
    } catch (err: unknown) {
      const e = err as DOMException | null;
      if (e?.name === "NotAllowedError" || e?.name === "PermissionDeniedError") {
        setError({ type: "permission_denied", message: "Microphone access was denied. Please allow microphone permission in your browser settings." });
      } else if (e?.name === "NotFoundError" || e?.name === "DevicesNotFoundError") {
        setError({ type: "no_microphone", message: "No microphone detected. Please connect a microphone and try again." });
      } else {
        setError({ type: "unknown", message: `Could not access microphone: ${e?.message ?? "unknown error"}` });
      }
      return;
    }

    // Create MediaRecorder
    const recorder = selectedMime
      ? new MediaRecorder(streamRef.current, { mimeType: selectedMime })
      : new MediaRecorder(streamRef.current);
    recorderRef.current = recorder;

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    const maxSeconds = getMaxDuration();

    recorder.onstop = () => {
      cleanupStream();
      clearTimer();
      const blob = new Blob(chunksRef.current, { type: selectedMime || chunksRef.current[0]?.type || "audio/webm" });
      setAudioBlob(blob);
      setDurationSeconds(elapsedRef.current);
      setIsRecording(false);
      setIsProcessing(false);
    };

    recorder.onerror = () => {
      cleanupStream();
      clearTimer();
      setError({ type: "unknown", message: "An error occurred during recording." });
      setIsRecording(false);
      setIsProcessing(false);
    };

    // Auto-stop at max duration
    timerRef.current = setInterval(() => {
      elapsedRef.current += 1;
      setDurationSeconds(elapsedRef.current);
      if (elapsedRef.current >= maxSeconds) {
        stopRecording();
      }
    }, 1000);

    recorder.start();
    setIsRecording(true);
  }, [isSupported, stopRecording, cleanupStream, clearTimer]);

  const resetBlob = useCallback(() => {
    setAudioBlob(null);
    setIsProcessing(false);
    setDurationSeconds(0);
    chunksRef.current = [];
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimer();
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }
      cleanupStream();
    };
  }, [clearTimer, cleanupStream]);

  return {
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
  };
}
