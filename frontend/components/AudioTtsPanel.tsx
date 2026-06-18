import { SectionCard } from "@/components/SectionCard";

export function AudioTtsPanel() {
  return (
    <SectionCard title="Output Suara" eyebrow="TTS">
      <div className="space-y-4">
        <p
          data-testid="tts-status"
          className="rounded-2xl border border-success bg-success-soft p-4 font-mono text-sm text-success"
        >
          TTS siap: audio jawaban placeholder tersedia untuk validasi UI.
        </p>
        <audio
          data-testid="audio-player"
          controls
          className="w-full rounded-2xl border border-line bg-surface-strong p-3"
        >
          <track kind="captions" label="Bahasa Indonesia" srcLang="id" />
        </audio>
      </div>
    </SectionCard>
  );
}
