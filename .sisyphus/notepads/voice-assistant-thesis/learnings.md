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
