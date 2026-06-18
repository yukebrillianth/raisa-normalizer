"use client";

import { useEffect, useRef } from "react";

type AnswerBlockProps = {
  answer: string;
  spokenAnswer: string;
  audioUrl: string;
  isVisible: boolean;
};

export function AnswerBlock({
  answer,
  spokenAnswer,
  audioUrl,
  isVisible,
}: AnswerBlockProps) {
  const audioRef = useRef<HTMLAudioElement>(null);

  // Auto-play when audio URL becomes available
  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.load();
      audioRef.current.play().catch(() => {
        // Autoplay blocked — user can press play manually
      });
    }
  }, [audioUrl]);

  if (!isVisible) return null;

  const displayAnswer = spokenAnswer || answer;

  return (
    <div className="animate-fade-in-up">
      <div className="raisa-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-its-blue">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </span>
          <span className="text-sm font-display text-its-cover">
            Jawaban RAISA
          </span>
        </div>
        <p
          className="text-lg text-text-primary leading-relaxed"
          data-testid="final-answer"
        >
          {displayAnswer}
        </p>

        {/* Audio player */}
        {audioUrl && (
          <div className="audio-player mt-4 pt-3 border-t border-surface-3">
            <audio
              ref={audioRef}
              controls
              src={audioUrl}
              data-testid="tts-audio-player"
            >
              Browser Anda tidak mendukung pemutar audio.
            </audio>
          </div>
        )}
      </div>
    </div>
  );
}
