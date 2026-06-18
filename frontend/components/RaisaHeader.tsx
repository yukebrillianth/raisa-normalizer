"use client";

export function RaisaHeader() {
  return (
    <header className="flex items-center gap-4 px-6 py-4 bg-surface-0">
      <div className="flex items-center gap-3">
        {/* ITS accent bar */}
        <div className="w-1 h-8 rounded-full bg-its-blue" />
        <div>
          <h1 className="font-display text-xl tracking-tight text-its-cover leading-tight">
            RAISA
          </h1>
          <p className="text-xs text-text-muted font-body leading-tight">
            Robot Asisten Informasi ITS
          </p>
        </div>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-surface-3 bg-surface-1 px-3 py-1 text-[11px] text-text-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
          Online
        </span>
      </div>
    </header>
  );
}
