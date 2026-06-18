"use client";

import { useState } from "react";

type QARow = {
  id: string;
  question: string;
  answer: string;
  embedding_generated: boolean;
};

type AdminQAFormProps = {
  token: string;
  mode: "create" | "edit";
  editRow?: QARow;
  onSuccess: () => void;
  onCancel: () => void;
};

const API_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export function AdminQAForm({
  token,
  mode,
  editRow,
  onSuccess,
  onCancel,
}: AdminQAFormProps) {
  const [question, setQuestion] = useState(editRow?.question ?? "");
  const [answer, setAnswer] = useState(editRow?.answer ?? "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [resultEmbedding, setResultEmbedding] = useState<boolean | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || !answer.trim()) {
      setError("Pertanyaan dan jawaban wajib diisi.");
      return;
    }

    setSubmitting(true);
    setError("");

    try {
      if (mode === "create") {
        const res = await fetch(`${API_BASE}/api/admin/qa`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ question: question.trim(), answer: answer.trim() }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        setResultEmbedding(data.embedding_generated);
      } else if (editRow) {
        const res = await fetch(
          `${API_BASE}/api/admin/qa/${encodeURIComponent(editRow.id)}`,
          {
            method: "PUT",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              question: question.trim(),
              answer: answer.trim(),
            }),
          }
        );
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        setResultEmbedding(data.embedding_generated);
      }

      setTimeout(() => onSuccess(), 1200);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal menyimpan.");
    } finally {
      setSubmitting(false);
    }
  };

  const isEdit = mode === "edit";

  return (
    <form
      data-testid={isEdit ? "admin-edit-form" : "admin-add-form"}
      onSubmit={handleSubmit}
      className="rounded-lg border border-its-black bg-surface-0 p-5"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.22em] text-its-blue">
          {isEdit ? "Edit QA" : "Tambah QA Baru"}
        </h3>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md border border-its-black px-3 py-1.5 text-xs text-text-primary bg-surface-0 hover:bg-surface-2 transition-colors"
        >
          Batal
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label
            htmlFor={`question-${mode}`}
            className="mb-1.5 block text-xs uppercase tracking-[0.15em] text-text-muted"
          >
            Pertanyaan
          </label>
          <textarea
            id={`question-${mode}`}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            placeholder="Masukkan pertanyaan..."
            className="w-full rounded-md border border-its-black bg-surface-0 px-4 py-3 text-sm text-text-primary placeholder:text-text-muted/50 focus:border-its-blue focus:outline-none"
          />
        </div>

        <div>
          <label
            htmlFor={`answer-${mode}`}
            className="mb-1.5 block text-xs uppercase tracking-[0.15em] text-text-muted"
          >
            Jawaban
          </label>
          <textarea
            id={`answer-${mode}`}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            rows={4}
            placeholder="Masukkan jawaban..."
            className="w-full rounded-md border border-its-black bg-surface-0 px-4 py-3 text-sm text-text-primary placeholder:text-text-muted/50 focus:border-its-blue focus:outline-none"
          />
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-md border border-error bg-error-soft px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}

      {resultEmbedding !== null && (
        <div className="mt-4 rounded-md border border-success bg-success-soft px-4 py-3 text-sm text-success">
          Berhasil disimpan! Embedding:{" "}
          <span className="font-semibold">
            {resultEmbedding ? "digenerate" : "tidak digenerate"}
          </span>
        </div>
      )}

      <div className="mt-5 flex justify-end">
        <button
          type="submit"
          disabled={submitting}
          className="rounded-md bg-its-blue px-6 py-2 text-xs font-semibold text-white transition-colors hover:bg-its-blue-dark disabled:cursor-not-allowed disabled:opacity-50 border border-its-black"
        >
          {submitting ? "Menyimpan..." : isEdit ? "Perbarui" : "Simpan"}
        </button>
      </div>
    </form>
  );
}
