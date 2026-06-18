"use client";

import { useRef, useState } from "react";
import { SectionCard } from "@/components/SectionCard";

type ImportError = {
  row?: number;
  error: string;
};

type ImportResult = {
  total: number;
  imported: number;
  errors: ImportError[];
};

type AdminCSVImportProps = {
  token: string;
  onImportSuccess: () => void;
};

const API_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export function AdminCSVImport({ token, onImportSuccess }: AdminCSVImportProps) {
  const [file, setFile] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleImport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Pilih file CSV terlebih dahulu.");
      return;
    }

    setImporting(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/api/admin/qa/import`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }

      const data: ImportResult = await res.json();
      setResult(data);
      onImportSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal import CSV.");
    } finally {
      setImporting(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <SectionCard title="CSV Import" eyebrow="Bulk QA loader" testId="admin-csv-import">
      <form onSubmit={handleImport} className="space-y-4">
        <div className="rounded-lg border border-dashed border-surface-3-light bg-surface-0 p-5">
          <label
            htmlFor="csv-file"
            className="mb-2 block font-mono text-xs uppercase tracking-[0.18em] text-text-muted"
          >
            Upload CSV
          </label>
          <p className="mb-4 text-sm leading-6 text-text-secondary">
            File harus memiliki header <span className="font-mono text-text-primary font-medium">question</span> dan{" "}
            <span className="font-mono text-text-primary font-medium">answer</span>. Setiap baris akan dibuat dengan
            embedding baru dari backend.
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <input
              id="csv-file"
              ref={inputRef}
              data-testid="admin-csv-input"
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block flex-1 text-sm text-text-secondary file:mr-4 file:rounded-md file:border file:border-its-black file:bg-surface-2 file:px-4 file:py-2 file:font-mono file:text-xs file:font-semibold file:text-text-primary hover:file:bg-surface-3-light cursor-pointer"
            />
            <button
              type="submit"
              disabled={importing || !file}
              className="rounded-md bg-its-blue px-5 py-2 font-mono text-xs font-semibold text-white transition-colors hover:bg-its-blue-dark disabled:cursor-not-allowed disabled:opacity-50 border border-its-black"
            >
              {importing ? "Mengimport..." : "Import CSV"}
            </button>
            {file && (
              <button
                type="button"
                onClick={clearFile}
                className="rounded-md border border-its-black px-4 py-2 font-mono text-xs text-text-primary bg-surface-0 hover:bg-surface-2 transition-colors"
              >
                Bersihkan
              </button>
            )}
          </div>
        </div>
      </form>

      {error && (
        <div className="mt-4 rounded-md border border-error bg-error-soft px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-5 space-y-4" data-testid="admin-csv-result">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-its-black bg-surface-0 p-4">
              <p className="font-mono text-xs uppercase tracking-[0.15em] text-text-muted">
                Total rows
              </p>
              <p className="mt-2 text-2xl font-bold tracking-[-0.05em] text-text-primary">
                {result.total}
              </p>
            </div>
            <div className="rounded-lg border border-success bg-success-soft p-4">
              <p className="font-mono text-xs uppercase tracking-[0.15em] text-success">
                Imported
              </p>
              <p className="mt-2 text-2xl font-bold tracking-[-0.05em] text-success">
                {result.imported}
              </p>
            </div>
            <div className="rounded-lg border border-error bg-error-soft p-4">
              <p className="font-mono text-xs uppercase tracking-[0.15em] text-error">
                Failed
              </p>
              <p className="mt-2 text-2xl font-bold tracking-[-0.05em] text-error">
                {result.errors.length}
              </p>
            </div>
          </div>

          {result.errors.length > 0 ? (
            <div className="overflow-hidden rounded-lg border border-error bg-error-soft">
              <div className="border-b border-error/30 px-4 py-3">
                <h3 className="font-mono text-xs uppercase tracking-[0.18em] text-error">
                  Row-level errors
                </h3>
              </div>
              <div className="max-h-64 overflow-auto">
                {result.errors.map((item, idx) => (
                  <div
                    key={`${item.row ?? "unknown"}-${idx}`}
                    className="grid gap-2 border-b border-error/20 px-4 py-3 text-sm last:border-b-0 sm:grid-cols-[7rem_1fr]"
                  >
                    <span className="font-mono text-error">
                      Row {item.row ?? "?"}
                    </span>
                    <span className="text-error">{item.error}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-success bg-success-soft px-4 py-3 text-sm text-success">
              Semua baris berhasil diimport.
            </div>
          )}
        </div>
      )}
    </SectionCard>
  );
}
