export function ErrorPanel() {
  return (
    <aside data-testid="error-panel" className="paper-panel rounded-[var(--radius-panel)] p-5">
      <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent-strong">
        Panel galat
      </p>
      <h2 className="mt-2 text-xl font-semibold tracking-[-0.03em] text-ink">
        Diagnostik sistem
      </h2>
      <div className="mt-5 rounded-2xl border border-success bg-success-soft p-4 text-sm leading-6 text-success">
        Tidak ada error aktif. Bagian ini tetap terlihat dalam debug mode untuk mendukung demonstrasi dan pengujian otomatis.
      </div>
    </aside>
  );
}
