import { SectionCard } from "@/components/SectionCard";

type AudioTtsPanelProps = {
  audioUrl?: string;
  provider?: string;
  fallbackUsed?: boolean;
  latencyMs?: number;
  status?: string;
};

export function AudioTtsPanel({
  audioUrl = "",
  provider = "-",
  fallbackUsed = false,
  latencyMs,
  status = "Menunggu hasil TTS dari pipeline.",
}: AudioTtsPanelProps) {
  return (
    <SectionCard title="Output Suara" eyebrow="TTS">
      <div className="space-y-4">
        <p
          data-testid="tts-status"
          className="rounded-2xl border border-info bg-info-soft p-4 font-mono text-sm text-info"
        >
          {status} provider={provider} fallback={String(fallbackUsed)}
          {typeof latencyMs === "number" ? ` latency=${latencyMs.toFixed(0)}ms` : ""}
        </p>
        <audio
          data-testid="audio-player"
          controls
          className="w-full rounded-2xl border border-line bg-surface-strong p-3"
          src={audioUrl || undefined}
        >
          <track kind="captions" label="Bahasa Indonesia" srcLang="id" />
        </audio>
        {!audioUrl ? (
          <p className="text-sm leading-6 text-ink-muted">
            Audio belum tersedia atau TTS gagal; jawaban teks tetap ditampilkan.
          </p>
        ) : null}
      </div>
    </SectionCard>
  );
}
