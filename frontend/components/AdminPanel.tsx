"use client";

import { AdminCSVImport } from "@/components/AdminCSVImport";
import { AdminQAForm } from "@/components/AdminQAForm";
import { SectionCard } from "@/components/SectionCard";
import { useCallback, useEffect, useRef, useState } from "react";

type QARow = {
  id: string;
  question: string;
  answer: string;
  embedding_generated: boolean;
};

const API_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export function AdminPanel() {
  const [token, setToken] = useState("");
  const [authed, setAuthed] = useState(false);
  const [qaList, setQaList] = useState<QARow[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState("");
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingRow, setEditingRow] = useState<QARow | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const searchTimer = useRef<ReturnType<typeof setTimeout>>(null);

  const headers = useCallback(
    () => ({
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    }),
    [token]
  );

  const fetchQA = useCallback(
    async (query: string) => {
      if (!token) return;
      setLoading(true);
      setListError("");
      try {
        const params = query ? `?search=${encodeURIComponent(query)}` : "";
        const res = await fetch(`${API_BASE}/api/admin/qa${params}`, {
          headers: headers(),
        });
        if (res.status === 401) {
          setAuthed(false);
          setListError("Token tidak valid. Silakan masukkan ulang.");
          return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: QARow[] = await res.json();
        setQaList(data);
        if (!authed) setAuthed(true);
      } catch (err) {
        setListError(err instanceof Error ? err.message : "Gagal memuat data");
      } finally {
        setLoading(false);
      }
    },
    [token, headers, authed]
  );

  const handleTokenSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (token.trim()) {
      setAuthed(true);
      fetchQA("");
    }
  };

  const handleSearchChange = (value: string) => {
    setSearch(value);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => fetchQA(value), 350);
  };

  const handleDelete = async (id: string) => {
    setDeleting(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/admin/qa/${encodeURIComponent(id)}?confirm=true`,
        { method: "DELETE", headers: headers() }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setQaList((prev) => prev.filter((r) => r.id !== id));
    } catch {
      setListError("Gagal menghapus baris.");
    } finally {
      setDeleting(false);
      setDeleteConfirmId(null);
    }
  };

  const handleFormSuccess = () => {
    setShowAddForm(false);
    setEditingRow(null);
    fetchQA(search);
  };

  const handleLogout = () => {
    setToken("");
    setAuthed(false);
    setQaList([]);
    setSearch("");
  };

  useEffect(() => {
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, []);

  if (!authed) {
    return (
      <SectionCard title="Admin Panel" eyebrow="QA Management" testId="admin-panel">
        <form onSubmit={handleTokenSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="admin-token"
              className="mb-2 block text-xs uppercase tracking-[0.18em] text-text-muted"
            >
              Admin Token
            </label>
            <div className="flex gap-3">
              <input
                id="admin-token"
                data-testid="admin-token-input"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Masukkan admin token..."
                className="flex-1 rounded-md border border-its-black bg-surface-0 px-4 py-2 text-sm text-text-primary placeholder:text-text-muted/50 focus:border-its-blue focus:outline-none"
              />
              <button
                type="submit"
                disabled={!token.trim()}
                className="rounded-md bg-its-blue px-6 py-2 text-sm font-semibold text-white transition-colors hover:bg-its-blue-dark disabled:cursor-not-allowed disabled:opacity-40 border border-its-black"
              >
                Masuk
              </button>
            </div>
          </div>
          <p className="text-sm text-text-secondary">
            Masukkan admin token untuk mengakses panel manajemen QA.
          </p>
        </form>
      </SectionCard>
    );
  }

  return (
    <div data-testid="admin-panel" className="space-y-6">
      <div className="raisa-card flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="flex items-center gap-3">
          <span className="h-2.5 w-2.5 rounded-full bg-success" />
          <span className="text-xs text-text-muted">
            Terhubung sebagai admin
          </span>
        </div>
        <button
          onClick={handleLogout}
          className="rounded-md border border-its-black px-4 py-1.5 text-xs text-text-primary bg-surface-0 hover:bg-surface-2 transition-colors"
        >
          Keluar
        </button>
      </div>

      <SectionCard title="Daftar QA" eyebrow="Kelola data QA" testId="admin-qa-list">
        <div className="flex flex-wrap items-center gap-3">
          <input
            type="text"
            data-testid="admin-search-input"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Cari pertanyaan atau jawaban..."
            className="min-w-48 flex-1 rounded-md border border-its-black bg-surface-0 px-4 py-2 text-sm text-text-primary placeholder:text-text-muted/50 focus:border-its-blue focus:outline-none"
          />
          <button
            onClick={() => setShowAddForm(true)}
            className="rounded-md bg-its-blue px-5 py-2 text-xs font-semibold text-white transition-colors hover:bg-its-blue-dark border border-its-black"
          >
            + Tambah QA
          </button>
        </div>

        {listError && (
          <div className="mt-4 rounded-md border border-error bg-error-soft px-4 py-3 text-sm text-error">
            {listError}
          </div>
        )}

        {loading && (
          <div className="mt-4 py-8 text-center text-sm text-text-muted">
            Memuat data...
          </div>
        )}

        {showAddForm && (
          <div className="mt-5">
            <AdminQAForm
              token={token}
              mode="create"
              onSuccess={handleFormSuccess}
              onCancel={() => setShowAddForm(false)}
            />
          </div>
        )}

        {editingRow && (
          <div className="mt-5">
            <AdminQAForm
              token={token}
              mode="edit"
              editRow={editingRow}
              onSuccess={handleFormSuccess}
              onCancel={() => setEditingRow(null)}
            />
          </div>
        )}

        {!loading && qaList.length === 0 && (
          <div className="mt-6 py-10 text-center">
            <p className="text-sm text-text-muted">
              Tidak ada data QA ditemukan.
            </p>
          </div>
        )}

        {!loading && qaList.length > 0 && (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-surface-3-light">
                  <th className="py-3 pr-4 text-xs uppercase tracking-[0.15em] text-text-muted">
                    ID
                  </th>
                  <th className="py-3 pr-4 text-xs uppercase tracking-[0.15em] text-text-muted">
                    Pertanyaan
                  </th>
                  <th className="py-3 pr-4 text-xs uppercase tracking-[0.15em] text-text-muted">
                    Jawaban
                  </th>
                  <th className="py-3 pr-4 text-xs uppercase tracking-[0.15em] text-text-muted">
                    Embedding
                  </th>
                  <th className="py-3 text-right text-xs uppercase tracking-[0.15em] text-text-muted">
                    Aksi
                  </th>
                </tr>
              </thead>
              <tbody>
                {qaList.map((row) => (
                  <tr
                    key={row.id}
                    data-testid="admin-qa-row"
                    className="border-b border-surface-3-light/50 hover:bg-surface-1"
                  >
                    <td className="max-w-24 truncate py-3 pr-4 text-xs text-text-muted">
                      {row.id}
                    </td>
                    <td className="max-w-xs truncate py-3 pr-4 text-text-primary">
                      {row.question}
                    </td>
                    <td className="max-w-xs truncate py-3 pr-4 text-text-secondary">
                      {row.answer}
                    </td>
                    <td className="py-3 pr-4">
                      {row.embedding_generated ? (
                        <span className="inline-flex items-center gap-1 rounded bg-success-soft px-2 py-0.5 text-xs text-success border border-success/20">
                          <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                          Ya
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded bg-warning-soft px-2 py-0.5 text-xs text-warning border border-warning/20">
                          <span className="h-1.5 w-1.5 rounded-full bg-warning" />
                          Tidak
                        </span>
                      )}
                    </td>
                    <td className="py-3 text-right">
                      <div className="flex justify-end gap-2">
                        <button
                          onClick={() => setEditingRow(row)}
                          className="rounded-md border border-its-black px-3 py-1 text-xs text-text-primary bg-surface-0 hover:bg-surface-2 transition-colors"
                        >
                          Edit
                        </button>
                        {deleteConfirmId === row.id ? (
                          <div className="flex items-center gap-1.5">
                            <button
                              onClick={() => handleDelete(row.id)}
                              disabled={deleting}
                              className="rounded-md bg-error px-3 py-1 text-xs font-semibold text-white transition-colors hover:bg-red-700 disabled:opacity-50 border border-its-black"
                            >
                              {deleting ? "..." : "Ya"}
                            </button>
                            <button
                              onClick={() => setDeleteConfirmId(null)}
                              className="rounded-md border border-its-black px-3 py-1 text-xs text-text-primary bg-surface-0 hover:bg-surface-2 transition-colors"
                            >
                              Batal
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setDeleteConfirmId(row.id)}
                            className="rounded-md border border-error px-3 py-1 text-xs text-error bg-surface-0 hover:bg-error-soft transition-colors"
                          >
                            Hapus
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>

      <AdminCSVImport token={token} onImportSuccess={() => fetchQA(search)} />
    </div>
  );
}
