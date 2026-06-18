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

T7 learnings - 2026-06-18

- Created `backend/app/providers/stt/openai.py` and `backend/app/providers/stt/__init__.py`.
- `STTProvider.process()` base signature changed from `-> str` to `-> dict[str, Any]` to match plan spec (returns `{"transcript", "language", "latency_ms"}`).
- Installed openai==2.43.0; the legacy `openai.Whisper.transcriptions.create()` path does not exist in v2.x. Used `openai.AsyncOpenAI().audio.transcriptions.create()` instead.
- `openai.AsyncOpenAI` triggers a `reportAttributeAccessIssue` Pyright false positive (the class exists at runtime but type stubs resolve it as a module). Suppressed with `# pyright: ignore`.
- Custom `STTProviderError(RuntimeError)` with `.code` attribute for all STT error variants: STT_INVALID_AUDIO (MIME/size/empty), STT_NO_KEY, STT_EMPTY_TRANSCRIPT, STT_API_ERROR.
- `health()` reports `api_key_configured: bool` without exposing key value.
- Added `openai` to `requirements.txt`.

T9 learnings - 2026-06-18

- Created `backend/app/providers/normalizer/vllm.py` with `VLLMNormalizerProvider` implementing `NormalizerProvider`.
- Uses httpx to call vLLM's OpenAI-compatible `/v1/chat/completions` endpoint with chat format (system + user messages).
- System prompt mirrors the Alpaca normalization instruction from `_MConverter.eu_test_retrieval.md:201-211`: same semantic content but adapted to chat format role.
- `process()` raises `RuntimeError` if `NORMALIZER_VLLM_BASE_URL` is empty (not configured), or on HTTP errors/timeouts, or if vLLM returns empty choices/content.
- Output cleaned via `_clean_output()` removing thinking blocks, quotes, and extra whitespace (matching the notebook's `clean_output`).
- `health()` returns `{provider, configured, base_url, model, reachable, status}`. When not configured, `status='unavailable'` (not an error — graceful degradation).
- When configured but vLLM unreachable, `status='unreachable'` with `reachable=False`. Health ping uses `GET /v1/models`.
- `configured` is derived from `bool(self._base_url)` — empty string means not configured.
- Added `httpx` to `requirements.txt` and `pyproject.toml` dependencies.
- Tested: imports resolve; `configured=False` + `status='unavailable'` when env var is empty; `configured=True` + `status='unreachable'` when vLLM URL set but server not running.
- LSP reports `httpx` import unresolved (same category as existing `torch`/`transformers` errors for alpaca.py — pre-existing project pyright config issue with root-level venv).

T8 learnings - 2026-06-18

- Alpaca normalizer at `backend/app/providers/normalizer/alpaca.py` implements `NormalizerProvider(ABC)` from `app.providers.base`.
- Uses `transformers.AutoTokenizer` + `AutoModelForCausalLM` for base model, `peft.PeftModel` for LoRA adapter — both optional, lazy-loaded on first `process()` call.
- Prompt format matches notebook exactly: `### Instruction:\n...### Input:\n{informal_text}\n### Response:\n` — verbatim ITS instructions.
- Generation is deterministic (greedy): do_sample=False, temperature=0.0, top_p=1.0, num_beams=1, repetition_penalty=1.05.
- Output cleaning: `re.sub(r" thinking.*? response", "", text, flags=re.DOTALL)` to strip thinking blocks, then strip quotes and collapse whitespace.
- GPU inference serialized via `threading.Lock()` to prevent OOM when multiple requests stack up.
- Singleton pattern: `get_instance()` classmethod returns shared instance (model loading is expensive).
- Fallback: if model loading fails, logs error and `process()` returns raw transcript; `health()` reports `model_loaded=False`.
- ABC's `process()` returns `str` (the normalized query text); provider/latency metadata is handled at the pipeline level via `ProviderMeta` schema.
- `torch`, `transformers`, and `peft` added to `requirements.txt`; they are not installed on dev machines without GPU — LSP import errors are expected per T1 learnings.
- Notebook used Unsloth's `FastLanguageModel` with `BitsAndBytesConfig` (8-bit), but the app uses standard `AutoModelForCausalLM` with `torch_dtype=torch.float16` + `device_map="auto"` for CUDA efficiency.
- Existing `vllm.py` already in normalizer package; updated `__init__.py` exports both `AlpacaNormalizerProvider` and `VLLMNormalizerProvider`.

T11 learnings - 2026-06-18

- Created `backend/app/providers/tts/` package with `__init__.py`, `supertonic.py`, and `openai.py`.
- Both providers implement `TTSProvider(ABC)` from `app.providers.base` returning `dict[str, Any]` (same pattern as STT — base says `-> str`, override returns `dict`).
- Supertonic-3: uses `supertonic==1.3.1` SDK, `TTS(auto_download=True)` lazy-loaded in `_ensure_loaded()`. `synthesize(text, voice_style, total_steps=...)` returns `tuple[np.ndarray, np.ndarray]` — wav is first element. Audio saved via `self._tts.save_audio(wav, path)`. Empty text validated before model load to avoid unnecessary downloads.
- OpenAI fallback: uses `openai.audio.speech.create(model=..., voice=..., input=...)` returning `response.content` (bytes). Gated behind `OPENAI_API_KEY` being set. Always sets `fallback_used=True`.
- Audio served via `FastAPI StaticFiles` at `/api/audio/` mounted on `backend/audio/` directory (auto-created).
- Auto-cleanup runs every 10 minutes via `asyncio.sleep(600)` in a daemon thread, deleting files older than `audio_retention_minutes * 60` seconds.
- LSP diagnostics show false positives: return type mismatch (same as STT pattern), `numpy`/`soundfile`/`supertonic` import resolution (venv not visible to Pyright), `openai.api_key`/`openai.audio` attribute access.
- Runtime verification passes: ABC subclass checks, health endpoints, empty text guards, config reading from settings.
- `_cleanup_loop` avoids `asyncio.run()` inside event loop by running in a daemon `threading.Thread`.

T10 learnings - 2026-06-18

- Retrieval package created at `backend/app/providers/retrieval/` with `embedding.py`, `retriever.py`, and `__init__.py`.
- BGE-M3 embedding provider lazy-loads `SentenceTransformer(embedding_model_name, device=embedding_device)` and calls `.eval()`, matching notebook lines 176-180; lazy import keeps module imports usable even before heavy ML deps are installed.
- `sentence-transformers` added to `backend/requirements.txt`; local install attempt timed out due package size, but dependency is declared for environment setup.
- pgvector retrieval intentionally uses raw psycopg2 via `get_db().execute_query(...)` and notebook-parity SQL shape with `%s::vector`, `1 - (embedding <=> %s::vector) AS similarity`, ordered by `<=>`, limited by `retrieval_top_k`.
- Vector literal formatting matches notebook requirement: `"[" + ",".join(map(str, embedding)) + "]"`; if the embedding comes from numpy upstream this is equivalent to required `vector.tolist()` formatting because provider returns `list[float]`.
- Keyword overlap formula exactly matches notebook: `len(q_words & c_words) / (len(q_words) + 1e-6)`. Example: overlap 2 of 4 words returns `0.49999987500003124`, not exactly `0.5` because of epsilon.
- Existing schemas capped `rerank_score <= 1.0`, but notebook formula `similarity + 0.2 * keyword_score` can exceed 1.0; schema bounds were relaxed to `ge=0.0` only for rerank_score.
- Baseline answered flag uses reranked top candidate's original `similarity >= retrieval_similarity_threshold`, not rerank_score, matching notebook lines 351-357.
- LSP diagnostics for `backend/app/providers/retrieval` are clean after using dynamic import for `sentence_transformers`.

T12 learnings - 2026-06-18

- Admin router at `backend/app/routers/admin.py` with prefix `/api/admin/qa`, registered via `app.include_router(admin.router)` in main.py.
- Auth dependency `_verify_admin` uses FastAPI `HTTPBearer` + `Depends`, compares `credentials.credentials` against `get_settings().admin_token`. Returns 401 on mismatch.
- Write operations use a separate psycopg2 connection from `_get_write_conn()` with explicit commit/rollback in try/except/finally blocks, avoiding conflicts with `Database` class's `autocommit=True` read pattern.
- Embedding model is lazily loaded as a singleton (`SentenceTransformer(get_settings().embedding_model_name)`), cached in module-level `_embedding_model`.
- ID strategy from `db.schema.id_strategy` ("id" or "question") — `_id_filter()` generates the correct WHERE clause for either strategy.
- `_dict_row()` maps column names to row values; `_id_value()` extracts the identifier based on strategy.
- CSV import: validates multipart/form-data, checks for "question" and "answer" columns in header, reports per-row errors in response with row numbers. On row-level DB failures, creates a new write connection to continue processing remaining rows.
- Delete endpoint requires `?confirm=true` query parameter.
- Update endpoint regenerates embedding only when question text actually changes.
- `sentence-transformers` added to requirements.txt (pulls torch, transformers, scikit-learn).
- LSP "reportMissingImports" for fastapi/pydantic/psycopg2 are environment issues (LSP doesn't find project venv), not code errors.

T13 learnings - 2026-06-18

- SSE pipeline router lives at `backend/app/routers/pipeline.py` and is registered in `main.py` via `app.include_router(pipeline.router)`.
- SSE formatting uses `event: {event_type}\ndata: {json}\n\n` with `StreamingResponse(media_type="text/event-stream")` and anti-buffering headers.
- Pipeline stage names for streamed events are `stt`, `normalize`, `embed`, `retrieve`, `baseline_rerank`, `select_and_verbalize`, and `tts`; timing keys still use existing `TimingContext` names (`normalization`, `embedding`, `retrieval`, `llm_selection`).
- LLM selection/verbalization provider reuses the Alpaca normalizer singleton/model/lock; the prompt includes raw transcript, normalized query, and top-3 question/answer/similarity/rerank_score candidates.
- To prevent answer invention, final `answer` and `llm_selection.selected_answer` are copied from the selected retrieved candidate by rank, not trusted from the LLM payload; the LLM only supplies rank/reason/spoken rephrase.
- Threshold gate skips LLM selection when no retrieved candidate passes similarity threshold and returns configured `FALLBACK_ANSWER`.
- Graceful degradation behavior: STT/embed/retrieve fatal stop with fallback/error details; normalizer falls back to transcript; LLM selection failure falls back to baseline; TTS failure returns text-only with a recoverable stage error.
- Verification used `python3 -m compileall app` plus source invariant checks for SSE format/events because local LSP still reports missing dependency imports (known project environment issue).

T13 review fixes - 2026-06-18

- T13 plan context expects `pipeline_start` to include the ordered stage list and `pipeline_complete` payload key to be `final_response`; schemas/router now match that SSE contract.
- Threshold gate must check whether ANY top-3 candidate passes similarity threshold, not `retrieval_result.answered` (which only reflects reranked top-1).
- LLM `selected_rank` output should be coerced from common JSON variants (`"1"`, `1.0`, `1`) into `int | None` before pipeline validation.
- `LLMSelectionResult` now carries `spoken_answer` so the `select_and_verbalize` stage_complete payload contains the same spoken rephrase later used for TTS.
- TTS degradation should try OpenAI fallback after Supertonic failure; text-only response is only after the selected provider and fallback path fail.
- Pipeline upload reading should be chunk-limited instead of reading the complete upload before size validation.
- Provider override form fields are security-sensitive; reject unknown normalizer/TTS providers with 422 instead of silently falling back.
- TTS should not speak unvalidated LLM text: use a lightweight divergence/instruction-marker check and fall back to selected candidate answer when suspicious.
- `psycopg2-binary` must be declared in both `backend/requirements.txt` and `backend/pyproject.toml` because db/admin modules import psycopg2.

T15 learnings - 2026-06-18

- Created 4 files: `AdminPanel.tsx`, `AdminQAForm.tsx`, `AdminCSVImport.tsx`, `app/admin/page.tsx`.
- All admin components use `'use client'` directive since they manage interactive state and event handlers.
- Token stored in React state (not localStorage) per spec; auth status toggles between empty state and full panel.
- Search uses debounced fetch with 350ms timeout via `useRef<ReturnType<typeof setTimeout>>` — real-time without hammering the API.
- Backend URL resolved from `NEXT_PUBLIC_BACKEND_URL` env var, falling back to `http://localhost:8000`.
- Authorization header: `Bearer ${token}` sent on every admin API call via a `headers()` useCallback.
- Delete confirmation uses inline state (`deleteConfirmId`) rather than a separate modal component — shows "Ya/Batal" buttons in the row.
- CSV import uses `FormData` with file input; response renders per-row errors with row numbers from backend.
- Design patterns reused: `SectionCard`, `paper-panel`, `rounded-[var(--radius-card)]`, `border-line`, `bg-background`, `font-mono` labels, `uppercase tracking-[0.18em]`, semantic color tokens (`success-soft`, `error-soft`, `warning-soft`).
- Build verification: `rm -rf .next && npx next build` passes with BUILD_ID generated.

T16 learnings - 2026-06-18

- Created `backend/app/routers/health.py` with aggregate `GET /api/health` endpoint using `APIRouter(prefix="/api/health")`.
- Each provider subsection wrapped in async helper with try/except — a single provider failure never blocks the full health check.
- DB health reports: connected, table_exists, row_count, pgvector_available from the cached singleton.
- Embedding health: delegates to `BGEEmbeddingProvider().health()` — returns provider/model_name/device/loaded.
- STT health: delegates to `OpenAIWhisperSTTProvider().health()` + re-enforces `api_key_configured` as bool-only.
- Normalizer health: aggregates Alpaca (required, via `get_instance()`) and vLLM (optional, returns `unavailable` if unreachable).
- TTS health: aggregates Supertonic (optional, returns `unavailable` on failure) and OpenAI (reports `api_key_configured` as bool).
- Admin health: reports `token_configured: bool` from `settings.admin_token`, never exposes token value.
- Version reported as `"1.0.0"` in aggregate response via module-level `__version__`.
- Removed old inline `health()` and `db_health()` endpoints from main.py (collision with `from app.routers import health` import).
- Also removed unused `get_db` import from main.py (only used by old db_health endpoint).
- Verdict: All LSP errors are pre-existing `fastapi` import resolution issues (venv not in LSP path).

T14 learnings - 2026-06-18

- Frontend SSE integration uses `fetch()` + `ReadableStream` instead of `EventSource` because backend stream requires `POST multipart/form-data`; upload sends `audio` blob and `mime_type` field to `/api/pipeline/audio-query/stream`.
- Backend URL resolution follows spec: `NEXT_PUBLIC_BACKEND_URL` with fallback `http://localhost:8000`, trimming trailing slash before composing API/audio URLs.
- Pipeline stages are rendered progressively from SSE events: `pipeline_start`, `stage_start`, `stage_complete`, `stage_error`, and `pipeline_complete`; stage ids match backend (`stt`, `normalize`, `embed`, `retrieve`, `baseline_rerank`, `select_and_verbalize`, `tts`).
- `pipeline_complete` client parser accepts both `data.response` and `data.final_response` because backend router currently emits `response=` while schema field is `final_response`.
- Existing visual system uses warm paper CSS variables/Tailwind v4 tokens; updated recorder styles away from default zinc/blue/red utility palette into semantic tokens (`accent`, `error`, `info`, `surface`, `line`).
- LSP diagnostics are clean for changed frontend files; full frontend directory still has pre-existing admin client-prop serialization warnings from T15.
- `npx next build` passes from `frontend/`.
## T17 README GPU lab deployment docs

- README keeps one deployment narrative: prerequisites, environment variables, model paths, database, backend setup, frontend setup, running, admin, troubleshooting.
- Backend dependency list in docs should match `backend/requirements.txt`: `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `python-multipart`, `python-dotenv`, `sentence-transformers`, `httpx`, `openai`, `supertonic`, `psycopg2-binary`, `soundfile`, `numpy`, `torch`, `transformers`, `peft`.
- GPU lab docs now call out local model path setup for the base HuggingFace model, LoRA adapter directory, Supertonic cache warmup, `BAAI/bge-m3` cache behavior, pgvector verification SQL, ffmpeg install, remote microphone HTTPS caveat, and expanded CUDA OOM mitigations.


## T19 QA smoke scenarios
- Created `.sisyphus/qa-scenarios.md` with 15 executable smoke scenarios covering API, frontend, admin, provider fallbacks, and failure paths.
- Confirmed actual endpoints used by implementation: `POST /api/pipeline/audio-query/stream`, `GET/POST/PUT/DELETE /api/admin/qa`, `POST /api/admin/qa/import`, `POST /api/admin/qa/{identifier}/regenerate-embedding`, `GET /api/health`, and `GET /api/audio/{filename}`.

## 2026-06-19 — OpenAI selection verbalizer
- Replaced pipeline selection+verbalization provider export/import with `OpenAISelectionVerbalizerProvider`; Alpaca selection file remains in place for reference and normalizer stays Alpaca/VLLM-controlled.
- New provider uses `AsyncOpenAI` chat completions with model `gpt-4.1-mini`, temperature 0.3, JSON-object response format, ITS system prompt, raw transcript, normalized query, and top-3 candidate question/answer/similarity/rerank scores.
- Provider always copies selected question/answer from retrieved candidates after parsing so only `spoken_answer` comes from GPT rephrasing; API/key/JSON failures fall back to top-1 candidate without breaking the pipeline.
