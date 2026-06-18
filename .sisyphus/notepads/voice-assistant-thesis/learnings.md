T1 learnings - 2026-06-18

- Next.js scaffold command worked: npx create-next-app@latest frontend --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*".
- Backend config uses pydantic-settings BaseSettings with SettingsConfigDict and Field aliases for uppercase env vars.
- Required env vars are DATABASE_URL, OPENAI_API_KEY, and ADMIN_TOKEN; missing vars are re-raised with a clear RuntimeError message.
- FastAPI app exposes GET / returning {"status": "ok"} and enables CORS from FRONTEND_ORIGIN.
- LSP import errors are expected before backend Python dependencies are installed.

T2 learnings - 2026-06-18

- psycopg2-binary installed into project root .venv (psycopg2-binary==2.9.12 on arm64 macOS).
- db.py uses psycopg2.connect(DATABASE_URL) with autocommit=True (matching notebook pattern at `_MConverter.eu_test_retrieval.md:107`).
- Graceful degradation: if DB is unreachable, `Database.connect()` logs a warning and sets `_connected=False` — app never crashes on DB failure.
- Schema inspection handles both regular arrays (`array_length(embedding::real[], 1)`) and pgvector (`vector_dims(embedding)`) for dimension detection, with a fallback if the first fails.
- Row identifier strategy: `id` column → "id" strategy; otherwise → "question" (text-based).
- Pyright type-narrowing: `self._conn` is `Optional[connection]` — added `assert self._conn is not None` in internal inspection methods and `cur.fetchone()` result None checks to satisfy the type checker.
- The `/api/health/db` endpoint returns `{connected, table_exists, row_count, vector_dimension, id_strategy, pgvector_available}` from the cached singleton schema.
- `get_db()` singleton performs `Database.startup()` (connect + inspect) on first call, so health endpoint triggers inspection lazily.

T5 learnings - 2026-06-18

- Provider package is at `backend/app/providers/` with four modules: `__init__.py`, `base.py` (ABCs), `schemas.py` (Pydantic v2 models), `timing.py` (TimingContext).
- All six pipeline providers are abstract base classes using `abc.ABC` + `abc.abstractmethod`: STTProvider, NormalizerProvider, EmbeddingProvider, RetrievalProvider, SelectionVerbalizerProvider, TTSProvider.
- Each ABC has `async process(...)` and `async health() → dict`.
- Pydantic models use v2 `model_config`, `model_dump()`, `model_dump_json()`. The PipelineResponse schema exactly matches the plan's final JSON (lines 207-246).
- SSE event models carry `event` (frozen string literal), `request_id`, `stage` (where applicable), and `timestamp`. `StageCompleteEvent` has a generic `data: dict[str, Any]` payload.
- `StageErrorEvent` includes `recoverable: bool` for pipeline error handling decisions.
- `TimingContext` uses `time.perf_counter()` with dataclass-backed `StageTiming` entries and `to_dict()` mapping to flat `TimingResult` keys.
- `generate_request_id()` returns UUID4 hex (32 chars).
- Imports work from `backend/` directory using `/absolute/path/to/.venv/bin/python`.

T6 learnings - 2026-06-18

- README.md created at project root with 13 sections covering full setup guide.
- All 37 env vars from .env.example documented with defaults and descriptions.
- Text-based architecture diagram maps all 7 pipeline stages (STT, Normalize, Embed, Retrieve, Select+Verbalize, TTS, Stream).
- Troubleshooting covers 4 scenarios: mic HTTPS, GPU OOM, DB unavailable, OpenAI unavailable.
- Scope disclaimer at top: lab/demo v1, push-to-talk only, no training.
- Bash heredoc used for file creation due to patch tool issues with triple-backtick content.

T4 learnings - 2026-06-18

- Frontend uses Next.js 16.2.9 App Router, React 19.2.4, Tailwind v4 via @import "tailwindcss" in app/globals.css.
- Audio recorder hook lives in frontend/hooks/useAudioRecorder.ts and detects MIME support in order: audio/webm;codecs=opus, audio/webm, audio/ogg.
- Browser recording compatibility must guard navigator.mediaDevices.getUserMedia and MediaRecorder before use; SSR-safe checks are required in client hooks.
- Browser-facing env vars need NEXT_PUBLIC_ prefix; hook reads NEXT_PUBLIC_MAX_RECORDING_SECONDS and defaults to 30 if absent/invalid.
- Component supports both click-to-toggle and press-and-hold push-to-talk via pointer events with a short hold threshold.
- Clean build may require removing frontend/.next after type shape changes because stale build artifacts can report old TypeScript errors.

T3 learnings - 2026-06-18

- Tailwind v4 uses `@theme inline` in globals.css to register design tokens directly; custom colors defined here become Tailwind utility classes automatically (e.g., `--color-accent: var(--accent)` → `bg-accent`, `text-accent`).
- Pre-existing `AudioRecorder.tsx` had stray duplicate JSX lines outside the component closing brace (lines 205-210) causing TypeScript build errors; fixed by removing the orphaned code and adding it inside the component return statement.
- Design system uses CSS custom properties with `color-mix(in srgb, ...)` for translucency effects — avoids hardcoding hex alpha values.
- Layout follows a 2-column grid on xl screens: main content (pipeline stages + data panels) on left, sticky sidebar (latency timeline + error panel) on right.
- Mock data uses Indonesian thesis context (cuti akademik questions) for realistic demonstration.
- All data-testid attributes placed on the elements specified in the plan for T14 integration readiness.
- `text-balance` utility works with Next.js 16 + Tailwind v4 for balanced heading line breaks.
- Build verification: `rm -rf .next && npx next build` is needed after significant structural changes to avoid stale cache errors.
