export function RecordControls() {
  return (
    <section className="paper-panel rounded-[var(--radius-panel)] p-6">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent-strong">
            Input suara
          </p>
          <h2 className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-ink">
            Kontrol mikrofon demonstrasi tesis
          </h2>
          <p
            data-testid="recording-status"
            className="mt-3 text-sm leading-6 text-ink-muted"
          >
            Status: siap merekam (placeholder UI; integrasi audio dilakukan pada tahap berikutnya).
          </p>
          <p data-testid="audio-error" className="mt-2 text-sm font-medium text-success">
            Tidak ada galat audio.
          </p>
        </div>
        <button
          data-testid="record-button"
          type="button"
          className="group relative inline-flex min-h-16 items-center justify-center overflow-hidden rounded-full border border-accent-strong bg-accent px-8 text-base font-bold text-surface shadow-[0_14px_35px_rgb(159_46_23_/_0.25)] transition-transform duration-300 hover:-translate-y-1 focus:outline-none focus:ring-4 focus:ring-accent-soft"
        >
          <span className="absolute inset-0 translate-y-full bg-accent-strong transition-transform duration-300 group-hover:translate-y-0" />
          <span className="relative">Mulai Rekam</span>
        </button>
      </div>
    </section>
  );
}
