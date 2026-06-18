import { AudioTtsPanel } from "@/components/AudioTtsPanel";
import { ErrorPanel } from "@/components/ErrorPanel";
import { LatencyTimeline } from "@/components/LatencyTimeline";
import { LLMSelection } from "@/components/LLMSelection";
import { PipelineStage } from "@/components/PipelineStage";
import { RecordControls } from "@/components/RecordControls";
import { RetrievalCandidates } from "@/components/RetrievalCandidates";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import type { PipelineStageData, RetrievalCandidate } from "@/components/types";

const pipelineStages: PipelineStageData[] = [
  {
    id: "stt",
    name: "1. Speech-to-Text",
    description: "Mengubah ujaran pengguna menjadi teks mentah untuk analisis berikutnya.",
    status: "complete",
    latencyMs: 820,
    testId: "stage-stt",
    detail:
      "Transkrip: Bagaimana prosedur pengajuan cuti akademik jika mahasiswa sedang menjalani penelitian lapangan yang jadwalnya berubah-ubah?",
  },
  {
    id: "normalize",
    name: "2. Normalisasi Query",
    description: "Membersihkan filler, memperbaiki istilah, dan menyusun query formal bahasa Indonesia.",
    status: "complete",
    latencyMs: 132,
    testId: "stage-normalize",
    detail:
      "Query: prosedur pengajuan cuti akademik untuk mahasiswa yang sedang menjalani penelitian lapangan dengan jadwal berubah",
  },
  {
    id: "retrieve",
    name: "3. Retrieval Kandidat",
    description: "Mengambil kandidat FAQ/dokumen menggunakan skor similarity dan pencocokan kata kunci.",
    status: "complete",
    latencyMs: 244,
    testId: "stage-retrieve",
    detail: "Top-3 kandidat ditemukan dari basis pengetahuan akademik dengan ambang similarity > 0.72.",
  },
  {
    id: "rerank",
    name: "4. Reranking",
    description: "Mengurutkan ulang kandidat berdasarkan relevansi semantik dan konteks pertanyaan.",
    status: "complete",
    latencyMs: 391,
    testId: "stage-rerank",
    detail: "Kandidat #1 dipilih karena mencakup syarat, alur administrasi, dan batas waktu pengajuan.",
  },
  {
    id: "select-verbalize",
    name: "5. Seleksi & Verbalization",
    description: "Memilih jawaban final lalu mengubahnya menjadi tuturan yang ringkas dan natural.",
    status: "active",
    latencyMs: 645,
    testId: "stage-select-verbalize",
    detail: "LLM menyusun jawaban lisan dengan alasan pemilihan kandidat dan gaya bahasa demonstrasi tesis.",
  },
  {
    id: "tts",
    name: "6. Text-to-Speech",
    description: "Menghasilkan audio jawaban akhir untuk diputar ke pengguna.",
    status: "pending",
    testId: "stage-tts",
    detail: "Menunggu final spoken_answer sebelum sintesis suara dijalankan.",
  },
];

const retrievalCandidates: RetrievalCandidate[] = [
  {
    rank: 1,
    question:
      "Apa prosedur pengajuan cuti akademik untuk mahasiswa yang sedang menghadapi kondisi akademik atau non-akademik khusus?",
    answer:
      "Mahasiswa mengajukan permohonan melalui sistem akademik, melampirkan alasan dan dokumen pendukung, meminta persetujuan dosen wali, lalu menunggu validasi program studi dan bagian akademik sesuai kalender akademik.",
    similarity: 0.912,
    keyword_score: 0.884,
    rerank_score: 0.946,
  },
  {
    rank: 2,
    question: "Kapan batas waktu pengajuan cuti akademik pada semester berjalan?",
    answer:
      "Pengajuan cuti akademik umumnya dilakukan sebelum batas perubahan rencana studi atau sesuai tanggal pada kalender akademik. Keterlambatan perlu disertai alasan khusus dan persetujuan tambahan.",
    similarity: 0.841,
    keyword_score: 0.742,
    rerank_score: 0.812,
  },
  {
    rank: 3,
    question:
      "Dokumen apa saja yang diperlukan mahasiswa untuk mengurus perubahan status akademik?",
    answer:
      "Dokumen pendukung dapat berupa surat permohonan, bukti kondisi yang relevan, rekomendasi dosen wali, dan formulir administrasi dari fakultas atau program studi.",
    similarity: 0.774,
    keyword_score: 0.701,
    rerank_score: 0.768,
  },
];

const latencyItems = [
  { label: "STT", ms: 820 },
  { label: "Normalize", ms: 132 },
  { label: "Retrieve", ms: 244 },
  { label: "Rerank", ms: 391 },
  { label: "Select", ms: 645 },
  { label: "TTS", ms: 0 },
];

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
                Semua bagian inti pipeline terlihat tanpa membutuhkan backend atau API call.
              </p>
            </div>
          </div>
        </header>

        <RecordControls />

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
          <div className="space-y-6">
            <section className="space-y-4">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent-strong">
                    Vertical pipeline flow
                  </p>
                  <h2 className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-ink">
                    Tahapan pemrosesan suara
                  </h2>
                </div>
                <span className="rounded-full border border-line bg-surface px-4 py-2 font-mono text-xs text-ink-muted">
                  placeholder data / no backend
                </span>
              </div>
              <div className="space-y-4">
                {pipelineStages.map((stage) => (
                  <PipelineStage key={stage.id} stage={stage} />
                ))}
              </div>
            </section>

            <TranscriptPanel
              transcript="Bagaimana prosedur pengajuan cuti akademik jika mahasiswa sedang menjalani penelitian lapangan yang jadwalnya berubah-ubah dan membutuhkan fleksibilitas administrasi?"
              normalizedQuery="prosedur pengajuan cuti akademik mahasiswa penelitian lapangan jadwal berubah fleksibilitas administrasi"
              providerInfo="provider=openai | model=gpt-4.1-mini | embedding=text-embedding-3-small"
            />

            <RetrievalCandidates candidates={retrievalCandidates} />

            <LLMSelection
              selectedRank={1}
              reason="Kandidat pertama paling sesuai karena menjelaskan alur pengajuan, dokumen pendukung, persetujuan dosen wali, dan validasi akademik. Kandidat lain hanya menjelaskan batas waktu atau dokumen secara parsial."
              spokenAnswer="Untuk mengajukan cuti akademik, mahasiswa perlu membuat permohonan melalui sistem akademik, melampirkan alasan serta dokumen pendukung, lalu meminta persetujuan dosen wali sebelum divalidasi oleh program studi dan bagian akademik."
              finalAnswer="Mahasiswa dapat mengajukan cuti akademik melalui sistem akademik dengan melampirkan alasan dan dokumen pendukung, meminta persetujuan dosen wali, kemudian menunggu validasi program studi serta bagian akademik sesuai kalender akademik."
            />

            <AudioTtsPanel />
          </div>

          <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
            <LatencyTimeline items={latencyItems} />
            <ErrorPanel />
          </div>
        </div>
      </div>
    </main>
  );
}
