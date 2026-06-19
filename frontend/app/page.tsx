"use client";

import { PipelineTimeline } from "@/components/PipelineTimeline";
import { RaisaHeader } from "@/components/RaisaHeader";
import { VoicePanel } from "@/components/VoicePanel";
import { usePipelineStream } from "@/hooks/usePipelineStream";
import { useState } from "react";

export default function Home() {
  const { state, submitAudio, reset } = usePipelineStream();
  const [sidebarHidden, setSidebarHidden] = useState(false);

  return (
    <main className="flex flex-col h-[100dvh] bg-surface-1">
      <RaisaHeader />

      <div className="flex-1 flex min-h-0">
        {/* LEFT — Voice interaction panel (~60%) */}
        <section className="flex-1 overflow-y-auto scrollable flex flex-col relative">
          <VoicePanel
            phase={state.phase}
            transcript={state.transcript}
            normalizedQuery={state.normalizedQuery}
            finalAnswer={state.finalAnswer}
            spokenAnswer={state.spokenAnswer}
            audioUrl={state.audioUrl}
            onSubmitAudio={submitAudio}
            onReset={reset}
          />

          {sidebarHidden && (
            <button
              onClick={() => setSidebarHidden(false)}
              className="hidden md:flex absolute top-3 right-3 z-30 items-center gap-1.5 rounded-lg bg-surface-0 border border-surface-3 px-3 py-1.5 text-xs text-text-muted hover:text-its-blue hover:border-its-blue transition-colors shadow-sm"
              title="Show pipeline sidebar"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="15" y1="3" x2="15" y2="21" />
              </svg>
              Pipeline
            </button>
          )}
        </section>

        {/* RIGHT — Pipeline timeline (~40%, max 400px) */}
        {!sidebarHidden && (
          <section className="w-[380px] shrink-0 bg-surface-0 hidden md:flex md:flex-col">
            <PipelineTimeline
              stages={state.stages}
              latencyItems={state.latencyItems}
              errors={state.errors}
              candidates={state.candidates}
              llmSelection={state.llmSelection}
              phase={state.phase}
              requestId={state.requestId}
              onHide={() => setSidebarHidden(true)}
            />
          </section>
        )}
      </div>

      {/* Mobile: Pipeline accessible via bottom sheet (simplified) */}
      <MobilePipelineToggle state={state} />
    </main>
  );
}

/* ── Mobile pipeline toggle ─────────────────────── */

import type { PipelineState } from "@/hooks/usePipelineStream";

function MobilePipelineToggle({ state }: { state: PipelineState }) {
  const [open, setOpen] = useState(false);
  const activeStages = state.stages.filter(
    (s) => s.status !== "pending",
  ).length;
  const totalStages = state.stages.length;

  return (
    <>
      {/* Toggle button — mobile only */}
      <button
        className="md:hidden fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-full bg-its-blue text-white px-4 py-2.5 shadow-md text-sm font-medium"
        onClick={() => setOpen(!open)}
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
        Pipeline {activeStages}/{totalStages}
      </button>

      {/* Bottom sheet */}
      {open && (
        <div className="md:hidden fixed inset-0 z-40">
          <div
            className="absolute inset-0 bg-black/30"
            onClick={() => setOpen(false)}
          />
          <div className="absolute bottom-0 left-0 right-0 max-h-[70vh] bg-surface-0 rounded-t-xl border-t border-surface-3 overflow-hidden flex flex-col animate-fade-in-up">
            <PipelineTimeline
              stages={state.stages}
              latencyItems={state.latencyItems}
              errors={state.errors}
              candidates={state.candidates}
              llmSelection={state.llmSelection}
              phase={state.phase}
              requestId={state.requestId}
            />
          </div>
        </div>
      )}
    </>
  );
}
