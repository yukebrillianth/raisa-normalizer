import { PipelineConsole } from "@/components/PipelineConsole";

export default function Home() {
  return (
    <main className="min-h-screen px-4 py-6 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="paper-panel overflow-hidden rounded-[var(--radius-panel)] p-6 sm:p-8">
          <div className="relative grid gap-6 lg:grid-cols-[1.25fr_0.75fr] lg:items-end">
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.32em] text-accent-strong">
                IRIS thesis observability console
              </p>
              <h1 className="mt-4 max-w-4xl text-4xl font-black tracking-[-0.07em] text-ink text-balance sm:text-6xl">
                Voice Assistant - Thesis Debug
              </h1>
              <p className="mt-5 max-w-3xl text-base leading-8 text-ink-muted sm:text-lg">
                Dashboard ini selalu menampilkan internal pipeline: audio masuk, STT, normalisasi,
                retrieval, reranking, seleksi LLM, TTS, latensi, dan panel galat untuk kebutuhan
                demonstrasi tesis berbahasa Indonesia.
              </p>
            </div>
            <div className="rounded-[var(--radius-card)] border border-line bg-surface-strong p-5">
              <p className="font-mono text-xs uppercase tracking-[0.22em] text-ink-muted">
                Mode tampilan
              </p>
              <p className="mt-3 text-2xl font-black tracking-[-0.05em] text-accent-strong">
                DEBUG ON
              </p>
              <p className="mt-2 text-sm leading-6 text-ink-muted">
                Semua bagian inti pipeline terlihat saat runtime, termasuk aliran SSE dari backend.
              </p>
            </div>
          </div>
        </header>

        <PipelineConsole />
      </div>
    </main>
  );
}
