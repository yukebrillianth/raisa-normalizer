"use client";

type TranscriptBlockProps = {
  transcript: string;
  normalizedQuery: string;
  isVisible: boolean;
};

export function TranscriptBlock({
  transcript,
  normalizedQuery,
  isVisible,
}: TranscriptBlockProps) {
  if (!isVisible) return null;

  const hasNormalized = normalizedQuery && normalizedQuery !== transcript;

  return (
    <div className="animate-fade-in-up space-y-3">
      {/* User transcript */}
      <div className="raisa-card p-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-surface-2">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="2" width="6" height="11" rx="3" />
              <path d="M5 10a7 7 0 0 0 14 0" />
            </svg>
          </span>
          <span className="text-[11px] font-mono uppercase tracking-widest text-text-muted">
            Transkrip
          </span>
        </div>
        <p className="text-base text-text-primary leading-relaxed" data-testid="transcript">
          {transcript || "—"}
        </p>
      </div>

      {/* Normalized query */}
      {hasNormalized && (
        <div className="raisa-card p-4 border-l-2 border-l-its-blue animate-fade-in-up">
          <div className="flex items-center gap-2 mb-2">
            <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-accent-soft">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--its-blue)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9" />
                <path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.855z" />
              </svg>
            </span>
            <span className="text-[11px] font-mono uppercase tracking-widest text-its-blue">
              Normalisasi
            </span>
          </div>
          <p className="text-base text-text-primary leading-relaxed font-medium" data-testid="normalized-query">
            {normalizedQuery}
          </p>
        </div>
      )}
    </div>
  );
}
