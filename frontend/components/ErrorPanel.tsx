import type { PipelineError } from "@/components/types";

type ErrorPanelProps = {
  errors?: PipelineError[];
};

export function ErrorPanel({ errors = [] }: ErrorPanelProps) {
  return (
    <aside data-testid="error-panel" className="paper-panel rounded-[var(--radius-panel)] p-5">
      <p className="text-xs uppercase tracking-[0.28em] text-accent-strong">
        Panel galat
      </p>
      <h2 className="mt-2 text-xl font-semibold tracking-[-0.03em] text-ink">
        Diagnostik sistem
      </h2>
      <div className="mt-5 space-y-3">
        {errors.length === 0 ? (
          <div className="rounded-2xl border border-success bg-success-soft p-4 text-sm leading-6 text-success">
            Tidak ada error aktif. Bagian ini tetap terlihat dalam debug mode untuk mendukung demonstrasi dan pengujian otomatis.
          </div>
        ) : (
          errors.map((error, index) => (
            <div
              key={`${error.stage}-${index}`}
              className="rounded-2xl border border-error bg-error-soft p-4 text-sm leading-6 text-error"
            >
              <p className="text-xs uppercase tracking-[0.18em]">
                {error.stage} / {error.recoverable === false ? "fatal" : "recoverable"}
              </p>
              <p className="mt-2 font-semibold">{error.message}</p>
              {error.detail ? <p className="mt-1 break-words">{error.detail}</p> : null}
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
