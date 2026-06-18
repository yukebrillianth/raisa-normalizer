import { AdminPanel } from "@/components/AdminPanel";

export const metadata = {
  title: "Admin Panel — IRIS QA Management",
  description:
    "Admin panel untuk mengelola basis data QA, import CSV, dan regenerasi embedding.",
};

export default function AdminPage() {
  return (
    <main className="min-h-screen px-4 py-6 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="paper-panel overflow-hidden rounded-[var(--radius-panel)] p-6 sm:p-8">
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-accent-strong">
            IRIS admin console
          </p>
          <h1 className="mt-4 max-w-4xl text-4xl font-black tracking-[-0.07em] text-ink text-balance sm:text-6xl">
            Admin Panel
          </h1>
          <p className="mt-5 max-w-3xl text-base leading-8 text-ink-muted sm:text-lg">
            Kelola basis data QA untuk tesis. Tambah, edit, hapus pertanyaan-jawaban,
            import CSV, dan pantau status embedding.
          </p>
        </header>

        <AdminPanel />
      </div>
    </main>
  );
}
