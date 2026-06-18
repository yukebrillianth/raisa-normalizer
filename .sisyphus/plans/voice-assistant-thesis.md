# Voice Assistant Thesis Web App Work Plan

## TL;DR

> **Quick Summary**: Build a thesis/debug-mode web app for an Indonesian campus information voice assistant: push-to-talk audio → OpenAI Whisper STT → fine-tuned LLM query normalization → PostgreSQL/pgvector top-3 retrieval + notebook-parity reranking for baseline/debug → LLM answer selection + verbalization from top-3 candidates → Supertonic-3 TTS with OpenAI TTS fallback, while streaming every stage to the UI via SSE.
>
> **Deliverables**:
> - Next.js thesis/debug frontend with microphone recording and agentic SSE pipeline visualization
> - FastAPI backend orchestrating STT, normalization providers, retrieval, baseline reranking, LLM answer selection + verbalization, TTS, SSE events, and admin operations
> - Configurable normalizer providers: Alpaca-format model as default, vLLM chat-template provider as optional/comparison mode
> - PostgreSQL/pgvector integration against existing `qa_ground_truth` data with `BAAI/bge-m3` embeddings
> - Admin endpoints/panel for listing/searching QA rows, add/update/delete, CSV import, and embedding regeneration
> - Lab GPU server setup documentation and environment contract
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 4 implementation waves + final review wave
> **Critical Path**: T1 config/schema → T5 provider/SSE contracts → T8 normalizer/verbalizer → T13 SSE orchestrator → T14 frontend integration → Final QA

---

## Context

### Original Request
User wants a web app for a senior thesis project. The app will be used like a robot waiter interface: user presses a button, speaks, the audio becomes text, the text is normalized using fine-tuned Qwen/Mistral models trained for informal-to-formal Indonesian query normalization, retrieval runs against PostgreSQL data, and the final response is spoken using TTS. STT should use OpenAI Whisper. TTS should use Supertonic-3, with OpenAI TTS fallback if Indonesian quality is poor. Project is currently empty except exported notebook/markdown references.

### Interview Summary
**Key Discussions**:
- Production priority is the best Alpaca-format model, served through custom Python/Transformers/PEFT/Unsloth-style inference.
- A chat-template/vLLM-compatible model should also be supported for comparison, even if quality is worse.
- Deployment target is a GPU lab server.
- UI should be thesis/debug mode: verbose, transparent, and showing every pipeline step.
- Existing database already contains QA rows and `BAAI/bge-m3` embeddings.
- Add admin endpoints/panel for QA data management and embedding generation.
- No automated test suite is requested; all verification will be agent-executed QA scenarios and smoke checks.
- Tech stack selected: Next.js frontend + FastAPI backend/services.
- Audio input should use the fastest/simple approach: Browser MediaRecorder push-to-talk upload.

**Research Findings**:
- `_MConverter.eu_test_retrieval.md` is the notebook source of truth for retrieval behavior.
- Notebook defaults: `TOP_K=3`, `SIMILARITY_THRESHOLD=0.75`, `RERANK_KEYWORD_WEIGHT=0.2`, `BAAI/bge-m3`, table `qa_ground_truth(question, answer, embedding)`.
- Notebook SQL computes `1 - (embedding <=> query_vector) AS similarity`, orders by pgvector distance, then applies a hybrid keyword rerank.
- Notebook applies threshold to selected candidate's original `similarity`, not `rerank_score`.
- Alpaca prompt format must preserve `### Instruction`, `### Input`, `### Response` and the instruction “Jangan menambahkan informasi baru.”
- Supertonic-3 is an ONNX-based 99M TTS model, CPU-friendly, fast enough for voice assistant use; `total_steps=5` is recommended.
- Supertonic-3 has no native streaming; v1 should synthesize complete answer audio and optionally sentence-chunk later.

### Metis Review
**Identified Gaps** (addressed):
- Exact GPU server/model paths are unknown → handled through environment variables and startup diagnostics.
- Existing DB schema beyond notebook columns is unknown → plan includes schema inspection and safe admin design.
- Admin endpoints could mutate thesis data unsafely → plan requires simple token auth and explicit destructive-operation guardrails.
- OpenAI Whisper should be explicit API mode → plan defaults to OpenAI API STT, not local Whisper.
- Need non-goals to prevent scope creep → plan explicitly excludes training, generated RAG, streaming v1, Kubernetes, and advanced rerankers.
- No automated tests but verification still needed → every task has agent-executed QA scenarios.

---

## Work Objectives

### Core Objective
Create a complete thesis/demo web app that runs the full voice assistant pipeline and streams every intermediate stage to the UI via SSE for agentic-style visualization: transcript, normalized query, top-3 retrieval candidates, baseline hybrid rerank scores, LLM answer selection from top-3 candidates, spoken-answer verbalization, TTS provider, audio output, and per-stage latency.

### Concrete Deliverables
- Monorepo/project scaffold with `frontend/` Next.js and `backend/` FastAPI.
- Environment-driven configuration for DB, OpenAI, embedding model, normalizer providers, retrieval thresholds, and TTS providers.
- FastAPI SSE orchestration endpoint for end-to-end audio query processing with stage streaming.
- Provider abstraction for STT, normalizer, embedding, retrieval/reranking, LLM answer selection + verbalization, and TTS.
- Thesis/debug frontend for push-to-talk recording and SSE-driven agentic stage-by-stage display.
- Admin QA management endpoints and UI.
- README with GPU lab setup and run instructions.

### Definition of Done
- [ ] User can open the web app, record audio, submit it, see pipeline stages, receive answer text, and play answer audio.
- [ ] Alpaca-format normalizer is default provider.
- [ ] vLLM chat-template provider is configurable/selectable when available.
- [ ] Retrieval behavior matches notebook defaults unless overridden by env vars.
- [ ] Admin can add/update QA data and regenerate embeddings.
- [ ] Supertonic-3 TTS is attempted first; OpenAI TTS fallback works when configured.
- [ ] All secrets and local paths are read from environment variables only.
- [ ] Final QA evidence exists in `.sisyphus/evidence/`.

### Must Have
- OpenAI Whisper API STT.
- Alpaca-format model support as priority/default.
- vLLM-compatible model support as optional/comparison provider.
- PostgreSQL + pgvector retrieval from existing QA data.
- `BAAI/bge-m3` embedding compatibility.
- Thesis/debug UI exposing intermediate pipeline outputs.
- Admin token/auth guard for mutating endpoints.
- Graceful failure behavior for every pipeline stage.
- SSE streaming of pipeline stage events to the frontend.
- LLM answer selection + verbalization stage using the same Alpaca model, selecting from top-3 candidates before TTS.

### Must NOT Have (Guardrails)
- No model training/fine-tuning inside the web app.
- No generated answer synthesis over retrieved context in v1; final answer must be selected from top-3 candidates or fallback/refusal.
- No live streaming STT/TTS in v1; use push-to-talk upload and request/response processing.
- No Kubernetes/autoscaling/multi-tenant production platform.
- No advanced rerankers beyond notebook hybrid keyword overlap unless explicitly added later.
- No hardcoded secrets, DB credentials, OpenAI keys, or local model paths.
- Normalizer must never answer the user or add new facts; it only rewrites the query.
- LLM selection/verbalization must not add new information; it only selects from top-3 candidates and rephrases for natural spoken output.

---

## Assumptions and Configuration Contract

### Assumptions
- GPU lab server has enough VRAM to serve one 14B normalizer at a time; initial implementation may serialize GPU inference.
- Existing table is named `qa_ground_truth`, but schema inspection will verify whether `id`, timestamps, and indexes exist.
- Existing embeddings were generated with `BAAI/bge-m3` and are dimension-compatible with freshly generated embeddings.
- OpenAI API key and outbound internet are available from the lab server.
- Browser target is current Chrome/Edge on desktop/laptop; microphone access requires HTTPS for non-localhost deployment.

### Environment Variables
```env
DATABASE_URL=
QA_TABLE=qa_ground_truth
FRONTEND_ORIGIN=http://localhost:3000

OPENAI_API_KEY=
STT_PROVIDER=openai_whisper
OPENAI_WHISPER_MODEL=whisper-1

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cuda

NORMALIZER_PROVIDER=alpaca
NORMALIZER_BASE_MODEL_PATH=
NORMALIZER_LORA_PATH=
NORMALIZER_VLLM_BASE_URL=
NORMALIZER_VLLM_MODEL=
NORMALIZER_MAX_INPUT_TOKENS=128
NORMALIZER_MAX_NEW_TOKENS=128
VERBALIZER_PROVIDER=alpaca
VERBALIZER_MAX_NEW_TOKENS=192
VERBALIZER_ENABLED=true

RETRIEVAL_TOP_K=3
RETRIEVAL_SIMILARITY_THRESHOLD=0.75
RERANK_KEYWORD_WEIGHT=0.2
FALLBACK_ANSWER=Maaf, saya belum dapat menemukan jawaban yang sesuai untuk pertanyaan Anda.

TTS_PROVIDER=supertonic
SUPERTONIC_TOTAL_STEPS=5
SUPERTONIC_VOICE=M1
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy

ADMIN_TOKEN=
AUDIO_RETENTION_MINUTES=60
MAX_RECORDING_SECONDS=30
MAX_AUDIO_UPLOAD_MB=25
PIPELINE_STREAM_MODE=sse
```

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No acceptance criterion may require “user manually confirms.” Since user chose no automated tests, verification relies on reproducible smoke commands, browser QA, API calls, and captured evidence.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None, per user request
- **Framework**: none
- **Agent-Executed QA**: ALWAYS mandatory

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Frontend/UI**: Use Playwright - navigate, interact, assert DOM, screenshot.
- **API/Backend**: Use Bash/curl - send requests, assert status and response fields.
- **CLI/Server**: Use Bash - run health commands and validate output.
- **Audio**: Capture response metadata and saved audio files; verify playable MIME/size.

---

## Pipeline Response Schema

Backend end-to-end endpoint should support SSE stage streaming and also produce a final debug-friendly response.

### SSE Stage Events

The primary UI path should use Server-Sent Events so the frontend can show agentic “thinking/progress” as each stage starts and completes.

```text
event: pipeline_start
data: {"request_id":"req_123","stages":["stt","normalize","embed","retrieve","baseline_rerank","select_and_verbalize","tts"]}

event: stage_start
data: {"request_id":"req_123","stage":"stt","message":"Mentranskrip audio dengan Whisper..."}

event: stage_complete
data: {"request_id":"req_123","stage":"stt","transcript":"...","latency_ms":1240}

event: stage_start
data: {"request_id":"req_123","stage":"select_and_verbalize","message":"Memilih jawaban terbaik dari top-3 dan membuat lebih natural untuk diucapkan..."}

event: stage_complete
data: {"request_id":"req_123","stage":"select_and_verbalize","selected_rank":1,"spoken_answer":"...","latency_ms":1800}

event: pipeline_complete
data: {"request_id":"req_123","final_response":{...}}
```

### Final Response JSON

The final `pipeline_complete` event should include a response shaped like:

```json
{
  "request_id": "string",
  "transcript": "string",
  "normalized_query": "string",
  "normalizer": {"provider": "alpaca", "latency_ms": 1234, "fallback_used": false},
  "retrieval": {
    "top_k": 3,
    "similarity_threshold": 0.75,
    "rerank_keyword_weight": 0.2,
    "candidates": [
      {"question": "string", "answer": "string", "similarity": 0.91, "keyword_score": 0.4, "rerank_score": 0.99},
      {"question": "string", "answer": "string", "similarity": 0.85, "keyword_score": 0.3, "rerank_score": 0.91},
      {"question": "string", "answer": "string", "similarity": 0.78, "keyword_score": 0.2, "rerank_score": 0.82}
    ],
    "baseline_rerank_selected": {"question": "string", "answer": "string", "similarity": 0.91, "rerank_score": 0.99},
    "answered": true
  },
  "answer": "string",
  "spoken_answer": "string",
  "llm_selection": {
    "provider": "alpaca",
    "selected_rank": 1,
    "selected_question": "string",
    "selected_answer": "string",
    "reason": "Candidate 1 paling relevan dengan query user",
    "latency_ms": 1800,
    "fallback_used": false,
    "refused": false,
    "refusal_reason": "string"
  },
  "tts": {"provider": "supertonic", "fallback_used": false, "audio_url": "/audio/...", "latency_ms": 900},
  "timing": {"stt_ms": 1000, "normalization_ms": 1200, "embedding_ms": 100, "retrieval_ms": 50, "llm_selection_ms": 1800, "tts_ms": 900, "total_ms": 5050},
  "errors": []
}
```

---

## Execution Strategy

### Parallel Execution Waves

```text
Wave 1 (Foundation, can start immediately):
├── T1: Project scaffold + configuration contract [quick]
├── T2: DB schema inspection + safe data-access layer [unspecified-high]
├── T3: Frontend shell + thesis/debug UI layout [visual-engineering]
├── T4: Audio recording/upload design [visual-engineering]
├── T5: Backend provider interfaces + response schema [unspecified-high]
└── T6: Deployment/README skeleton for GPU lab [writing]

Wave 2 (Core providers and services):
├── T7: OpenAI Whisper STT provider [quick]
├── T8: Alpaca normalizer provider [deep]
├── T9: vLLM chat-template normalizer provider [unspecified-high]
├── T10: Embedding + retrieval + baseline reranking service [deep]
├── T11: TTS providers: Supertonic-3 + OpenAI fallback [unspecified-high]
└── T12: Admin API for QA CRUD/import/embedding regeneration [unspecified-high]

Wave 3 (Integration):
├── T13: SSE end-to-end orchestrator + LLM selection/verbalization [deep]
├── T14: Frontend SSE pipeline visualization integration [visual-engineering]
├── T15: Admin panel integration [visual-engineering]
└── T16: Health checks + graceful degradation [unspecified-high]

Wave 4 (Hardening and deployment):
├── T17: GPU lab deployment scripts/docs [writing]
├── T18: Security/secret handling/admin guardrails [unspecified-high]
├── T19: Manual QA smoke assets and sample utterances [quick]
└── T20: UX polish for thesis demo flow [visual-engineering]

Wave FINAL:
├── F1: Plan compliance audit
├── F2: Code quality review
├── F3: Real manual QA
└── F4: Scope fidelity check
```

### Dependency Matrix
- **T1**: blocks T7-T20; blocked by none.
- **T2**: blocks T10, T12, T13, T16; blocked by none.
- **T3**: blocks T14, T15, T20; blocked by none.
- **T4**: blocks T14, T19; blocked by none.
- **T5**: blocks T7-T13, T16; blocked by none.
- **T6**: blocks T17; blocked by none.
- **T7**: blocks T13; blocked by T1, T5.
- **T8**: blocks T13, T19; blocked by T1, T5.
- **T9**: blocks T13, T19; blocked by T1, T5.
- **T10**: blocks T12, T13, T19; blocked by T1, T2, T5.
- **T11**: blocks T13; blocked by T1, T5.
- **T12**: blocks T15, T18, T19; blocked by T1, T2, T5, T10.
- **T13**: blocks T14, T16, T19, T20; blocked by T7-T11.
- **T14**: blocks T20; blocked by T3, T4, T13.
- **T15**: blocks T20; blocked by T3, T12.
- **T16**: blocks T17, T19; blocked by T2, T5, T13.
- **T17**: blocks final verification; blocked by T6, T16.
- **T18**: blocks final verification; blocked by T12.
- **T19**: blocks final verification; blocked by T4, T8-T10, T12, T13, T16.
- **T20**: blocks final verification; blocked by T14, T15.

### Agent Dispatch Summary
- **Wave 1**: 6 agents — T1 quick, T2 unspecified-high, T3 visual-engineering, T4 visual-engineering, T5 unspecified-high, T6 writing.
- **Wave 2**: 6 agents — T7 quick, T8 deep, T9 unspecified-high, T10 deep, T11 unspecified-high, T12 unspecified-high.
- **Wave 3**: 4 agents — T13 deep, T14 visual-engineering, T15 visual-engineering, T16 unspecified-high.
- **Wave 4**: 4 agents — T17 writing, T18 unspecified-high, T19 quick, T20 visual-engineering.

---

## TODOs

> Implementation + verification = ONE task. Every task below includes mandatory QA scenarios. No automated unit tests are required unless the executor chooses a tiny smoke script; do not create a full test suite.

- [x] 1. Project scaffold + configuration contract

  **What to do**:
  - Create project structure for `frontend/` Next.js and `backend/` FastAPI.
  - Add `.env.example` with all variables from the configuration contract.
  - Add backend settings loader that validates required env vars without exposing secrets.
  - Add frontend environment config for backend base URL.

  **Must NOT do**:
  - Do not hardcode DB password, OpenAI key, local model paths, or server URLs.
  - Do not add automated test framework unless needed for minimal scaffold validation.

  **Recommended Agent Profile**:
  - **Category**: `quick` — scaffolding/config task.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7-T20
  - **Blocked By**: None

  **References**:
  - `_MConverter.eu_test_retrieval.md:69-89` — notebook constants that become env defaults.
  - `_MConverter.eu_test_retrieval.md:98-104` — DB config example; use only to know required fields, never copy credentials.

  **Acceptance Criteria**:
  - [ ] `frontend/` and `backend/` directories exist with runnable baseline apps.
  - [ ] `.env.example` documents DB, OpenAI, normalizer, retrieval, TTS, admin, and frontend variables.
  - [ ] Backend settings loader starts with sample env and reports missing required values clearly.

  **QA Scenarios**:
  ```text
  Scenario: Backend config validation succeeds with sample env
    Tool: Bash
    Preconditions: Fill non-secret dummy values from .env.example in a temporary env file.
    Steps:
      1. Run backend settings/health startup command.
      2. Assert output contains config keys but not secret values.
    Expected Result: Command exits 0 and does not print secret-like values.
    Evidence: .sisyphus/evidence/task-1-config-validation.txt

  Scenario: Missing required DATABASE_URL fails clearly
    Tool: Bash
    Preconditions: Remove DATABASE_URL from temporary env.
    Steps:
      1. Run backend settings validation.
      2. Assert nonzero exit or structured error mentioning DATABASE_URL.
    Expected Result: Clear validation error, no stacktrace-only failure.
    Evidence: .sisyphus/evidence/task-1-missing-db-url.txt
  ```

  **Commit**: YES
  - Message: `chore(scaffold): initialize thesis voice assistant app`

- [ ] 2. DB schema inspection + safe data-access layer

  **What to do**:
  - Implement DB connection layer using `DATABASE_URL`.
  - Add startup/schema inspection for `QA_TABLE`.
  - Verify columns for question, answer, embedding, optional id/timestamps.
  - Verify pgvector extension and embedding dimension.
  - Define safe row identifier strategy for admin operations; if no id exists, use a guarded compatibility mode and document migration recommendation.

  **Must NOT do**:
  - Do not destructively migrate existing thesis data without explicit backup/migration step.
  - Do not update/delete by ambiguous question text unless no safer key exists and endpoint requires explicit confirmation.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — DB safety and compatibility work.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T10, T12, T13, T16
  - **Blocked By**: None

  **References**:
  - `_MConverter.eu_test_retrieval.md:98-108` — psycopg2 connection/autocommit pattern.
  - `_MConverter.eu_test_retrieval.md:326-332` — source SQL query and assumed columns.

  **Acceptance Criteria**:
  - [ ] Health/schema endpoint reports DB connectivity, table existence, pgvector availability, embedding dimension, row count, and id strategy.
  - [ ] No credentials are logged.
  - [ ] DB errors return structured errors.

  **QA Scenarios**:
  ```text
  Scenario: DB health reports existing QA table
    Tool: Bash (curl)
    Preconditions: Backend running with real lab DB env.
    Steps:
      1. GET /api/health/db.
      2. Assert JSON has connected=true, table_exists=true, row_count >= 0, vector_dimension present.
    Expected Result: HTTP 200 with schema diagnostics.
    Evidence: .sisyphus/evidence/task-2-db-health.json

  Scenario: Invalid DB URL returns controlled error
    Tool: Bash (curl)
    Preconditions: Run backend with invalid DATABASE_URL.
    Steps:
      1. GET /api/health/db.
      2. Assert HTTP 503 or clear error body.
    Expected Result: Structured DB unavailable response, app does not crash.
    Evidence: .sisyphus/evidence/task-2-db-invalid.json
  ```

  **Commit**: YES
  - Message: `feat(db): inspect qa table and pgvector health`

- [ ] 3. Frontend shell + thesis/debug UI layout

  **What to do**:
  - Build Next.js shell with thesis-oriented layout.
  - Include sections for microphone controls, transcript, normalized query, model provider, retrieval candidates table, reranking scores, answer, TTS status, audio player, and latency timeline.
  - Include demo/debug toggle if useful, but default to debug mode.

  **Must NOT do**:
  - Do not hide core pipeline internals in default thesis mode.
  - Do not require backend availability to render the shell.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering` — UI/UX implementation.
  - **Skills**: `frontend-ui-ux` if available to executor.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T14, T15, T20
  - **Blocked By**: None

  **References**:
  - Pipeline response schema in this plan — determines UI sections.
  - `_MConverter.eu_test_retrieval.md:461-482` — evaluation/debug columns that inspire UI labels.

  **Acceptance Criteria**:
  - [ ] Frontend loads and shows all required pipeline sections with placeholder data.
  - [ ] Layout handles long question/answer text without breaking.
  - [ ] UI labels are understandable for thesis demonstration.

  **QA Scenarios**:
  ```text
  Scenario: Debug layout renders all pipeline sections
    Tool: Playwright
    Preconditions: Frontend running with mock/placeholder state.
    Steps:
      1. Navigate to http://localhost:3000.
      2. Assert selectors exist: [data-testid="record-button"], [data-testid="transcript"], [data-testid="normalized-query"], [data-testid="retrieval-candidates"], [data-testid="final-answer"], [data-testid="tts-status"].
      3. Capture screenshot.
    Expected Result: All sections visible.
    Evidence: .sisyphus/evidence/task-3-layout.png

  Scenario: Long answer does not break layout
    Tool: Playwright
    Preconditions: Mock long answer/candidates enabled.
    Steps:
      1. Load page with mock long answer.
      2. Assert candidate table remains within viewport/container and answer wraps.
    Expected Result: No horizontal page overflow beyond intended scroll container.
    Evidence: .sisyphus/evidence/task-3-long-answer.png
  ```

  **Commit**: YES
  - Message: `feat(ui): add thesis debug dashboard shell`

- [ ] 4. Audio recording/upload design

  **What to do**:
  - Implement browser MediaRecorder push-to-talk/click-to-record.
  - Detect supported MIME types (`audio/webm;codecs=opus`, `audio/webm`, `audio/ogg`).
  - Enforce max recording duration and show microphone permission errors.
  - Prepare multipart upload to backend endpoint.

  **Must NOT do**:
  - Do not implement live streaming audio in v1.
  - Do not allow unlimited recording length/upload size.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering` — browser audio UX.
  - **Skills**: `playwright` for browser validation.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T14, T19
  - **Blocked By**: None

  **References**:
  - User requirement: fastest/simple audio input; use MediaRecorder upload.

  **Acceptance Criteria**:
  - [ ] User can start and stop recording.
  - [ ] UI shows recording duration and selected MIME type.
  - [ ] Permission denied and unsupported browser states are clearly displayed.

  **QA Scenarios**:
  ```text
  Scenario: Recording controls enter and exit recording state
    Tool: Playwright
    Preconditions: Browser context grants microphone permission using fake media device.
    Steps:
      1. Click [data-testid="record-button"].
      2. Assert [data-testid="recording-status"] contains "Recording".
      3. Wait 2 seconds and click stop.
      4. Assert an audio blob/upload-ready state is shown.
    Expected Result: Recording lifecycle completes without page error.
    Evidence: .sisyphus/evidence/task-4-recording.png

  Scenario: Microphone permission denied shows actionable error
    Tool: Playwright
    Preconditions: Browser context denies microphone permission.
    Steps:
      1. Click [data-testid="record-button"].
      2. Assert [data-testid="audio-error"] mentions microphone permission.
    Expected Result: Clear error message, no crash.
    Evidence: .sisyphus/evidence/task-4-permission-denied.png
  ```

  **Commit**: YES
  - Message: `feat(ui): add push to talk recording controls`

- [ ] 5. Backend provider interfaces + response schema

  **What to do**:
  - Define provider interfaces/contracts for STT, normalizer, embedding, retrieval/baseline rerank, LLM selection+verbalization, TTS, and SSE event streaming.
  - Define Pydantic models for pipeline response schema.
  - Add request ID, SSE event models, and per-stage timing/error collection utilities.
  - Add provider selector fields for normalizer and TTS.

  **Must NOT do**:
  - Do not couple the orchestrator directly to one model implementation.
  - Do not lose partial stage results when later stages fail.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — API contract design.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7-T13, T16
  - **Blocked By**: None

  **References**:
  - Pipeline response schema in this plan.
  - `_MConverter.eu_test_retrieval.md:359-364` — candidate/retrieval record shape.

  **Acceptance Criteria**:
  - [ ] Pydantic response models cover transcript, normalized query, retrieval candidates, baseline rerank selection, LLM selection/verbalization, TTS, timing, SSE event types, and errors.
  - [ ] Partial error responses can include completed stages.

  **QA Scenarios**:
  ```text
  Scenario: Mock pipeline response validates against schema
    Tool: Bash
    Preconditions: Backend dependencies installed.
    Steps:
      1. Run a schema validation command or call mock endpoint.
      2. Assert JSON includes request_id, transcript, retrieval.candidates, timing, errors.
    Expected Result: Schema validation succeeds.
    Evidence: .sisyphus/evidence/task-5-schema-valid.json

  Scenario: Partial error response preserves completed stages
    Tool: Bash (curl)
    Preconditions: Mock endpoint configured to fail at TTS stage.
    Steps:
      1. POST mock request.
      2. Assert transcript/normalized/retrieval fields exist and errors includes TTS failure.
    Expected Result: Structured partial response, no blank generic 500.
    Evidence: .sisyphus/evidence/task-5-partial-error.json
  ```

  **Commit**: YES
  - Message: `feat(api): define voice pipeline provider contracts`

- [ ] 6. Deployment/README skeleton for GPU lab

  **What to do**:
  - Create README sections for prerequisites: Python, Node, CUDA/GPU, PostgreSQL/pgvector, OpenAI API, model paths.
  - Document expected local paths via env vars, not hardcoded values.
  - Document first-run steps for frontend and backend.

  **Must NOT do**:
  - Do not claim production-hardening beyond lab/demo deployment.
  - Do not include real credentials.

  **Recommended Agent Profile**:
  - **Category**: `writing` — technical docs.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T17
  - **Blocked By**: None

  **References**:
  - `_MConverter.eu_test_retrieval.md:69-89` — default model/retrieval constants.
  - Metis review environment checklist in planning context.

  **Acceptance Criteria**:
  - [ ] README includes lab server setup checklist.
  - [ ] README explains no training is performed by this app.
  - [ ] README documents push-to-talk non-streaming v1 scope.

  **QA Scenarios**:
  ```text
  Scenario: README contains required setup sections
    Tool: Bash
    Preconditions: README exists.
    Steps:
      1. Search README for "Environment", "GPU", "PostgreSQL", "OpenAI", "Models", "Run".
      2. Assert each section exists.
    Expected Result: Required sections present.
    Evidence: .sisyphus/evidence/task-6-readme-sections.txt

  Scenario: README has no credential literals
    Tool: Bash
    Preconditions: README exists.
    Steps:
      1. Scan README for obvious secret examples like real API keys or password="admin".
      2. Assert only placeholders are used.
    Expected Result: No real credentials in docs.
    Evidence: .sisyphus/evidence/task-6-readme-secrets.txt
  ```

  **Commit**: YES
  - Message: `docs: add gpu lab setup skeleton`

- [ ] 7. OpenAI Whisper STT provider

  **What to do**:
  - Implement STT provider using OpenAI Whisper API.
  - Accept browser audio uploads, validate MIME/size/duration, and convert if necessary.
  - Return transcript, language if available, latency, and error details.

  **Must NOT do**:
  - Do not implement local Whisper unless later requested.
  - Do not run downstream stages if STT fails or returns empty transcript, except returning structured error.

  **Recommended Agent Profile**:
  - **Category**: `quick` — external API provider implementation.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T13
  - **Blocked By**: T1, T5

  **References**:
  - User requirement: STT uses OpenAI Whisper API.
  - Browser MediaRecorder outputs from T4.

  **Acceptance Criteria**:
  - [ ] Valid audio upload returns non-empty transcript for spoken Indonesian sample.
  - [ ] Empty/corrupt audio returns clear STT error.
  - [ ] OpenAI key absence is reported by health checks without exposing secrets.

  **QA Scenarios**:
  ```text
  Scenario: Valid audio sample transcribes
    Tool: Bash (curl)
    Preconditions: Backend running with OPENAI_API_KEY and a small Indonesian audio sample file.
    Steps:
      1. POST multipart audio to /api/stt/transcribe.
      2. Assert HTTP 200 and transcript is non-empty.
    Expected Result: Transcript returned with stt latency.
    Evidence: .sisyphus/evidence/task-7-stt-valid.json

  Scenario: Corrupt audio fails gracefully
    Tool: Bash (curl)
    Preconditions: Backend running.
    Steps:
      1. POST a text file pretending to be audio.
      2. Assert HTTP 400/422 with error code STT_INVALID_AUDIO.
    Expected Result: Controlled error, no downstream processing.
    Evidence: .sisyphus/evidence/task-7-stt-invalid.json
  ```

  **Commit**: YES
  - Message: `feat(stt): add openai whisper transcription provider`

- [ ] 8. Alpaca normalizer provider

  **What to do**:
  - Implement default normalizer provider for Alpaca-format model from local base model and LoRA paths.
  - Preserve notebook prompt format and deterministic generation settings.
  - Clean outputs by removing `<think>` blocks, quotes, and extra whitespace.
  - Serialize GPU inference initially to reduce OOM risk.
  - Include fallback behavior: if normalization fails, return structured error and optionally allow orchestrator to use transcript as query if configured.

  **Must NOT do**:
  - Do not use chat template for Alpaca provider.
  - Do not allow normalizer to generate an answer.
  - Do not add new information beyond rewriting the query.

  **Recommended Agent Profile**:
  - **Category**: `deep` — GPU model serving adaptation from notebook.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T13, T19
  - **Blocked By**: T1, T5

  **References**:
  - `_MConverter.eu_test_retrieval.md:201-211` — exact Alpaca prompt format.
  - `_MConverter.eu_test_retrieval.md:214-224` — output cleaning pattern.
  - `_MConverter.eu_test_retrieval.md:226-265` — deterministic generation settings.
  - `_MConverter.eu_test_retrieval.md:122-170` — model/tokenizer/LoRA load pattern.

  **Acceptance Criteria**:
  - [ ] Provider loads local base model and LoRA adapter from env paths.
  - [ ] Normalization response includes provider name, normalized text, latency, and warnings.
  - [ ] Prompt format matches notebook exactly unless config explicitly overrides.
  - [ ] Generation uses deterministic settings matching notebook.

  **QA Scenarios**:
  ```text
  Scenario: Informal query normalizes via Alpaca provider
    Tool: Bash (curl)
    Preconditions: Backend running on GPU server with NORMALIZER_PROVIDER=alpaca and valid model paths.
    Steps:
      1. POST {"text":"rek, dimana bisa ngurus surat aktif kuliah?"} to /api/normalize.
      2. Assert provider="alpaca" and normalized_query is non-empty.
      3. Assert response does not contain "### Instruction" or prompt echo.
    Expected Result: Clean normalized Indonesian query.
    Evidence: .sisyphus/evidence/task-8-alpaca-normalize.json

  Scenario: Missing model path fails gracefully
    Tool: Bash (curl)
    Preconditions: Run backend with invalid NORMALIZER_LORA_PATH.
    Steps:
      1. GET /api/health/normalizer.
      2. Assert available=false and error mentions missing path without stacktrace leakage.
    Expected Result: Provider unavailable state, no crash.
    Evidence: .sisyphus/evidence/task-8-missing-model.json
  ```

  **Commit**: YES
  - Message: `feat(normalizer): serve alpaca lora provider`

- [ ] 9. vLLM chat-template normalizer provider

  **What to do**:
  - Implement optional provider that calls a vLLM OpenAI-compatible chat/completions endpoint.
  - Make provider selectable per request or config.
  - Return provider latency and errors.
  - Keep normalization instruction equivalent to Alpaca provider while using official chat format.

  **Must NOT do**:
  - Do not make vLLM provider required for the app to start.
  - Do not degrade Alpaca default priority.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — external model server integration.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T13, T19
  - **Blocked By**: T1, T5

  **References**:
  - User decision: try B model too, but A is priority.
  - `_MConverter.eu_test_retrieval.md:201-211` — semantic content of normalization instruction.

  **Acceptance Criteria**:
  - [ ] Provider health reports unavailable if vLLM URL/model not configured.
  - [ ] When configured, `/api/normalize?provider=vllm` returns normalized query.
  - [ ] UI can show provider used.

  **QA Scenarios**:
  ```text
  Scenario: vLLM provider unavailable is non-fatal
    Tool: Bash (curl)
    Preconditions: Backend running without NORMALIZER_VLLM_BASE_URL.
    Steps:
      1. GET /api/health/normalizer.
      2. Assert alpaca can be available while vllm.available=false.
    Expected Result: App remains usable with Alpaca provider.
    Evidence: .sisyphus/evidence/task-9-vllm-unconfigured.json

  Scenario: vLLM provider normalizes when configured
    Tool: Bash (curl)
    Preconditions: vLLM OpenAI-compatible server running and env configured.
    Steps:
      1. POST text to /api/normalize with provider="vllm".
      2. Assert provider="vllm" and normalized_query non-empty.
    Expected Result: Optional provider works.
    Evidence: .sisyphus/evidence/task-9-vllm-normalize.json
  ```

  **Commit**: YES
  - Message: `feat(normalizer): add optional vllm provider`

- [ ] 10. Embedding + retrieval + baseline reranking service

  **What to do**:
  - Load `BAAI/bge-m3` SentenceTransformer according to env.
  - Embed normalized query.
  - Query existing PostgreSQL/pgvector table using notebook-parity SQL to retrieve top-3 candidates.
  - Compute keyword overlap and baseline rerank score for debug/baseline comparison.
  - Preserve subtle notebook behavior: baseline threshold applies to selected candidate's original `similarity` after rerank selection.
  - Return top-3 candidates with baseline rerank scores for the LLM selection stage.

  **Must NOT do**:
  - Do not add BM25/cross-encoder reranking in v1.
  - Do not perform final answer selection in this stage; LLM selection happens in T13.
  - Do not change defaults without env override.

  **Recommended Agent Profile**:
  - **Category**: `deep` — core retrieval correctness.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T12, T13, T19
  - **Blocked By**: T1, T2, T5

  **References**:
  - `_MConverter.eu_test_retrieval.md:176-180` — embedding model load.
  - `_MConverter.eu_test_retrieval.md:300-307` — encoding normalized queries.
  - `_MConverter.eu_test_retrieval.md:313-317` — keyword overlap.
  - `_MConverter.eu_test_retrieval.md:326-357` — pgvector retrieval, rerank, threshold/fallback behavior.

  **Acceptance Criteria**:
  - [ ] Retrieval endpoint returns original candidate order and baseline reranked order.
  - [ ] Candidate fields include question, answer, similarity, keyword_score, rerank_score.
  - [ ] Baseline threshold/fallback logic matches notebook for debug comparison.
  - [ ] Embedding dimension mismatch is detected clearly.
  - [ ] Returns exactly top-3 candidates (or fewer if DB has fewer than 3 rows).

  **QA Scenarios**:
  ```text
  Scenario: Known normalized query returns candidates and scores
    Tool: Bash (curl)
    Preconditions: Backend connected to existing QA DB with embeddings.
    Steps:
      1. POST {"query":"informasi pembayaran UKT"} to /api/retrieve.
      2. Assert candidates length <= 3 and each candidate has similarity, keyword_score, rerank_score.
      3. Assert retrieval.top_k=3 and threshold=0.75 by default.
    Expected Result: Notebook-shaped retrieval output.
    Evidence: .sisyphus/evidence/task-10-retrieve-known.json

  Scenario: Out-of-domain query falls back below threshold
    Tool: Bash (curl)
    Preconditions: Backend connected to QA DB.
    Steps:
      1. POST nonsense/out-of-domain query to /api/retrieve.
      2. If selected similarity < 0.75, assert answered=false and answer equals configured fallback.
    Expected Result: Graceful fallback behavior.
    Evidence: .sisyphus/evidence/task-10-retrieve-fallback.json
  ```

  **Commit**: YES
  - Message: `feat(retrieval): implement pgvector hybrid reranking`

- [ ] 11. TTS providers: Supertonic-3 + OpenAI fallback

  **What to do**:
  - Implement Supertonic-3 TTS provider with `total_steps=5` default.
  - Implement OpenAI TTS fallback provider.
  - Return audio URL or binary response suitable for frontend audio player.
  - Capture provider used, fallback_used, latency, and errors.
  - Support disabling Supertonic to force OpenAI fallback for demo.

  **Must NOT do**:
  - Do not require native streaming in v1.
  - Do not fail text answer if all TTS providers fail; return text-only with error.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — media provider integration.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T13
  - **Blocked By**: T1, T5

  **References**:
  - Supertonic-3 research: ONNX, `supertonic[serve]`, `total_steps=5`, no native streaming.
  - User decision: fallback to OpenAI TTS if Supertonic Indonesian quality is poor.

  **Acceptance Criteria**:
  - [ ] Supertonic provider creates playable audio for Indonesian answer text when available.
  - [ ] OpenAI TTS fallback is attempted if Supertonic fails or is disabled.
  - [ ] Response includes audio URL, provider, fallback flag, and latency.
  - [ ] Text-only response still succeeds if both TTS providers fail.

  **QA Scenarios**:
  ```text
  Scenario: Supertonic synthesizes playable audio
    Tool: Bash (curl)
    Preconditions: Backend running with Supertonic installed.
    Steps:
      1. POST {"text":"Silakan menuju loket akademik untuk informasi lebih lanjut."} to /api/tts.
      2. Assert HTTP 200, provider="supertonic", audio_url present.
      3. Fetch audio_url and assert content-type audio/* and size > 1000 bytes.
    Expected Result: Playable audio generated.
    Evidence: .sisyphus/evidence/task-11-supertonic.json

  Scenario: Supertonic failure falls back to OpenAI TTS
    Tool: Bash (curl)
    Preconditions: Backend configured to disable/break Supertonic and valid OPENAI_API_KEY.
    Steps:
      1. POST TTS request.
      2. Assert provider="openai" and fallback_used=true.
    Expected Result: Audio still generated by fallback.
    Evidence: .sisyphus/evidence/task-11-openai-fallback.json
  ```

  **Commit**: YES
  - Message: `feat(tts): add supertonic with openai fallback`

- [ ] 12. Admin API for QA CRUD/import/embedding regeneration

  **What to do**:
  - Implement admin endpoints guarded by `ADMIN_TOKEN`.
  - Support list/search QA rows.
  - Support add/update/delete with safe identifier strategy from T2.
  - Generate/regenerate embeddings with `BAAI/bge-m3` when question changes or new row is added.
  - Support CSV import with required columns and per-row error reporting.
  - Reject save if embedding generation fails.

  **Must NOT do**:
  - Do not expose unauthenticated mutation endpoints.
  - Do not silently drop invalid CSV rows without reporting.
  - Do not regenerate embeddings with a different model than configured.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — admin/data mutation safety.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T15, T18, T19
  - **Blocked By**: T1, T2, T5, T10

  **References**:
  - `_MConverter.eu_test_retrieval.md:189-190` — required data columns in test CSV (`input_query`, `gt_query`, `gt_answer`) for context; admin QA likely needs at least question/answer.
  - `_MConverter.eu_test_retrieval.md:300-307` — embedding generation pattern.
  - `_MConverter.eu_test_retrieval.md:326-332` — table fields used by retrieval.

  **Acceptance Criteria**:
  - [ ] Admin mutation endpoints require token.
  - [ ] Add/update question regenerates embedding.
  - [ ] CSV import validates required columns and reports row-level failures.
  - [ ] Search/list endpoint works for debug/admin UI.

  **QA Scenarios**:
  ```text
  Scenario: Unauthorized admin mutation is rejected
    Tool: Bash (curl)
    Preconditions: Backend running with ADMIN_TOKEN set.
    Steps:
      1. POST /api/admin/qa without token.
      2. Assert HTTP 401/403.
    Expected Result: Mutation rejected.
    Evidence: .sisyphus/evidence/task-12-admin-unauthorized.json

  Scenario: Add QA row generates embedding and becomes retrievable
    Tool: Bash (curl)
    Preconditions: Backend running with admin token and DB write access.
    Steps:
      1. POST a QA row with question="di mana lokasi helpdesk akademik demo" and answer="Helpdesk akademik berada di kantor akademik.".
      2. Assert response includes row id or safe identifier and embedding_generated=true.
      3. POST retrieve query for same question and assert candidate includes the new answer.
    Expected Result: New row is retrievable immediately.
    Evidence: .sisyphus/evidence/task-12-admin-add-retrieve.json
  ```

  **Commit**: YES
  - Message: `feat(admin): manage qa rows and embeddings`

- [ ] 13. SSE end-to-end orchestrator + LLM answer selection/verbalization

  **What to do**:
  - Implement `/api/pipeline/audio-query/stream` SSE endpoint accepting audio + provider options and streaming stage events.
  - Orchestrate STT → normalization → embedding/retrieval → baseline rerank → threshold gate → LLM selection+verbalization from top-3 → TTS.
  - Implement one same-model LLM call that receives raw transcript, normalized query, top-3 retrieved candidates, similarity scores, baseline rerank scores, and domain instruction.
  - LLM selection+verbalization must output structured JSON: selected_rank (1-3 or null), selected_answer, spoken_answer, reason, refused boolean, refusal_reason.
  - If all top-3 candidates are below threshold or LLM refuses, return fallback/refusal answer: “Maaf, saya hanya dapat membantu pertanyaan seputar layanan dan informasi kampus ITS.” or configured fallback.
  - Track per-stage latency and preserve partial results on failure.
  - Implement graceful degradation matrix:
    - STT fail → stop with STT error.
    - Normalizer fail → configurable fallback to transcript or stop with error.
    - DB fail → clear retrieval error.
    - No candidate above threshold → configured fallback/refusal answer; skip LLM selection unless config explicitly allows diagnostic selection.
    - LLM selection/verbalization fail → fallback to baseline rerank selected answer if above threshold, otherwise fallback/refusal.
    - Supertonic fail → OpenAI TTS fallback.
    - All TTS fail → text-only response.

  **Must NOT do**:
  - Do not let LLM invent new answer content; it must select from top-3 candidates or refuse.
  - Do not ignore threshold gate.
  - Do not hide errors from debug response.

  **Recommended Agent Profile**:
  - **Category**: `deep` — integration and failure semantics.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: T14, T16, T19, T20
  - **Blocked By**: T7, T8, T9, T10, T11

  **References**:
  - Pipeline response schema in this plan.
  - `_MConverter.eu_test_retrieval.md:271-368` — normalizing and retrieval sequence.

  **Acceptance Criteria**:
  - [ ] Valid audio request streams SSE events for every stage and ends with `pipeline_complete`.
  - [ ] Final response includes transcript, normalized_query, top-3 retrieval candidates, baseline rerank selection, LLM selected_rank/reason/spoken_answer, timing, and TTS metadata.
  - [ ] Provider selection works for Alpaca default and vLLM if configured.
  - [ ] Partial failures return structured debug response.
  - [ ] Out-of-domain or below-threshold queries return refusal/fallback and do not hallucinate.

  **QA Scenarios**:
  ```text
  Scenario: End-to-end audio query succeeds
    Tool: Bash (curl)
    Preconditions: Backend running with OpenAI STT, Alpaca provider, DB, and at least one TTS provider configured.
    Steps:
      1. POST Indonesian audio sample to /api/pipeline/audio-query/stream.
      2. Assert SSE stream includes pipeline_start, stage_start/stage_complete for stt, normalize, retrieve, baseline_rerank, select_and_verbalize, tts.
      3. Assert pipeline_complete final_response has transcript, normalized_query, retrieval.candidates length <= 3, llm_selection.selected_rank, spoken_answer, timing.total_ms.
      4. Assert audio_url present or tts.errors explain text-only fallback.
    Expected Result: Complete streamed debug response.
    Evidence: .sisyphus/evidence/task-13-e2e-success.json

  Scenario: TTS failure still returns text answer
    Tool: Bash (curl)
    Preconditions: Disable all TTS providers but keep STT/normalizer/DB working.
    Steps:
      1. POST valid audio query to /api/pipeline/audio-query/stream.
      2. Assert answer non-empty and errors includes TTS failure.
    Expected Result: Text-only answer, no full pipeline crash.
    Evidence: .sisyphus/evidence/task-13-tts-failure-text-only.json
  ```

  **Commit**: YES
  - Message: `feat(api): orchestrate end to end voice pipeline`

- [ ] 14. Frontend SSE pipeline visualization integration

  **What to do**:
  - Wire recording/upload UI to the SSE pipeline endpoint using EventSource or fetch + ReadableStream.
  - Render agentic "thinking" UI showing each stage as it starts/completes: STT → Normalize → Retrieve → Baseline Rerank → LLM Selection+Verbalize → TTS.
  - Render all final response fields: transcript, normalized query, provider, top-3 candidates with scores, baseline rerank selected, LLM selected_rank + reason + spoken_answer, threshold decision, timing, TTS status, audio player.
  - Show loading/active states per stage and error panels for partial failures.
  - Allow selecting normalizer provider: default Alpaca, optional vLLM.

  **Must NOT do**:
  - Do not hide retrieval scores in thesis mode.
  - Do not block answer display if audio playback fails.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering` — interactive frontend integration.
  - **Skills**: `playwright` for browser QA.

  **Parallelization**:
  - **Can Run In Parallel**: YES with T15/T16 after T13
  - **Parallel Group**: Wave 3
  - **Blocks**: T20
  - **Blocked By**: T3, T4, T13

  **References**:
  - Pipeline response schema in this plan.
  - T3 layout components.

  **Acceptance Criteria**:
  - [ ] User can record/upload from UI and see SSE stage events render in agentic/thinking UI.
  - [ ] Each stage appears as "active" when started and "complete" when finished with latency badge.
  - [ ] Candidate table shows similarity, keyword_score, rerank_score for all top-3.
  - [ ] LLM selection shows selected_rank, reason, and spoken_answer.
  - [ ] Audio player plays returned audio URL when present.
  - [ ] Partial errors display completed stages.

  **QA Scenarios**:
  ```text
  Scenario: UI displays successful pipeline response
    Tool: Playwright
    Preconditions: Frontend and backend running; backend can use mock or real sample response.
    Steps:
      1. Navigate to app.
      2. Submit sample audio or trigger mock pipeline.
      3. Assert SSE stage events appear progressively (pipeline_start, stage_start, stage_complete for each stage, pipeline_complete).
      4. Assert transcript, normalized query, candidates table with top-3, LLM selection details, final spoken_answer, timing, and audio player are visible.
      5. Capture screenshot.
    Expected Result: Agentic streaming debug output rendered progressively.
    Evidence: .sisyphus/evidence/task-14-pipeline-ui.png

  Scenario: Partial backend error renders completed stages
    Tool: Playwright
    Preconditions: Backend configured to fail TTS after retrieval.
    Steps:
      1. Submit sample audio.
      2. Assert final answer text is visible and TTS error panel is visible.
    Expected Result: UI remains useful under partial failure.
    Evidence: .sisyphus/evidence/task-14-partial-error-ui.png
  ```

  **Commit**: YES
  - Message: `feat(ui): render voice pipeline debug response`

- [ ] 15. Admin panel integration

  **What to do**:
  - Build frontend admin/debug panel for QA list/search/add/edit/delete/import.
  - Add token input/storage strategy suitable for lab demo.
  - Show embedding generation status and row-level import errors.
  - Require confirmation for delete/bulk import.

  **Must NOT do**:
  - Do not expose admin mutations without token.
  - Do not build a full CMS beyond thesis needs.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering` — admin UI workflows.
  - **Skills**: `playwright` for UI QA.

  **Parallelization**:
  - **Can Run In Parallel**: YES with T14/T16 after T12
  - **Parallel Group**: Wave 3
  - **Blocks**: T20
  - **Blocked By**: T3, T12

  **References**:
  - T12 admin API contract.

  **Acceptance Criteria**:
  - [ ] Admin can search/list QA rows.
  - [ ] Admin can add a QA row and see embedding_generated status.
  - [ ] Unauthorized state is clearly shown.
  - [ ] Import UI reports successful and failed rows.

  **QA Scenarios**:
  ```text
  Scenario: Unauthorized admin UI blocks mutation
    Tool: Playwright
    Preconditions: Frontend/backend running, no admin token provided.
    Steps:
      1. Open admin panel.
      2. Attempt to add QA row.
      3. Assert UI shows authorization required and no success message.
    Expected Result: Mutation blocked.
    Evidence: .sisyphus/evidence/task-15-admin-unauthorized.png

  Scenario: Authorized admin adds QA row
    Tool: Playwright
    Preconditions: Admin token configured and DB write access available.
    Steps:
      1. Enter token.
      2. Add demo question/answer.
      3. Assert success message includes embedding generated.
      4. Search for the new question.
    Expected Result: New QA row appears in search results.
    Evidence: .sisyphus/evidence/task-15-admin-add.png
  ```

  **Commit**: YES
  - Message: `feat(ui): add qa admin panel`

- [ ] 16. Health checks + graceful degradation

  **What to do**:
  - Implement `/api/health` aggregate endpoint.
  - Report DB, pgvector, embedding model, normalizer providers, STT config, TTS providers, and admin config presence.
  - Add graceful error codes for stage failures.
  - Ensure health does not expose secrets.

  **Must NOT do**:
  - Do not print API keys or passwords.
  - Do not make optional vLLM/Supertonic unavailability fatal if fallbacks exist.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — robustness/diagnostics.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES with T14/T15 after T13
  - **Parallel Group**: Wave 3
  - **Blocks**: T17, T19
  - **Blocked By**: T2, T5, T13

  **References**:
  - Configuration contract in this plan.
  - Metis graceful degradation matrix.

  **Acceptance Criteria**:
  - [ ] Health endpoint reports readiness per provider.
  - [ ] Optional providers can be unavailable without failing the whole app.
  - [ ] Secrets are masked or absent.

  **QA Scenarios**:
  ```text
  Scenario: Aggregate health reports provider statuses
    Tool: Bash (curl)
    Preconditions: Backend running.
    Steps:
      1. GET /api/health.
      2. Assert JSON includes db, embedding, stt, normalizers, tts, admin.
    Expected Result: Structured health report.
    Evidence: .sisyphus/evidence/task-16-health.json

  Scenario: Health response contains no secrets
    Tool: Bash
    Preconditions: /api/health response saved.
    Steps:
      1. Scan response for OPENAI_API_KEY value, DB password, ADMIN_TOKEN value.
      2. Assert none are present.
    Expected Result: No secret leakage.
    Evidence: .sisyphus/evidence/task-16-health-no-secrets.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add health and graceful degradation diagnostics`

- [ ] 17. GPU lab deployment scripts/docs

  **What to do**:
  - Finalize README with lab server deployment steps.
  - Add run commands for frontend/backend.
  - Document vLLM optional server startup separately.
  - Document HTTPS/microphone caveat for remote server.
  - Include ffmpeg requirement if audio conversion is used.

  **Must NOT do**:
  - Do not promise one-click production deployment.
  - Do not include real lab credentials.

  **Recommended Agent Profile**:
  - **Category**: `writing` — deployment documentation.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Final verification
  - **Blocked By**: T6, T16

  **References**:
  - README skeleton from T6.
  - Health endpoint from T16.

  **Acceptance Criteria**:
  - [ ] README documents backend, frontend, DB, OpenAI, normalizer, vLLM, and TTS setup.
  - [ ] README has troubleshooting for mic HTTPS, GPU OOM, DB unavailable, OpenAI unavailable.

  **QA Scenarios**:
  ```text
  Scenario: Setup docs cover all service dependencies
    Tool: Bash
    Preconditions: README finalized.
    Steps:
      1. Search README for "Next.js", "FastAPI", "PostgreSQL", "pgvector", "OpenAI", "Alpaca", "vLLM", "Supertonic".
      2. Assert all terms appear in relevant setup sections.
    Expected Result: Complete setup documentation.
    Evidence: .sisyphus/evidence/task-17-docs-coverage.txt

  Scenario: Troubleshooting includes microphone HTTPS caveat
    Tool: Bash
    Preconditions: README finalized.
    Steps:
      1. Search README for "HTTPS" and "microphone".
      2. Assert caveat explains remote lab URL requirement.
    Expected Result: Browser mic deployment issue documented.
    Evidence: .sisyphus/evidence/task-17-https-mic.txt
  ```

  **Commit**: YES
  - Message: `docs: finalize gpu lab deployment guide`

- [ ] 18. Security/secret handling/admin guardrails

  **What to do**:
  - Review backend/frontend for secret exposure.
  - Ensure admin mutating endpoints require token.
  - Add CORS config limited by `FRONTEND_ORIGIN`.
  - Add max upload size/duration checks.
  - Add delete/import confirmations or server-side confirmation flags.

  **Must NOT do**:
  - Do not expose admin token to frontend bundle except user-entered runtime value.
  - Do not allow wildcard CORS in lab deployment docs unless explicitly local-only.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — security hardening.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Final verification
  - **Blocked By**: T12

  **References**:
  - Configuration contract and admin guardrails in this plan.

  **Acceptance Criteria**:
  - [ ] Mutating admin endpoints reject missing/invalid token.
  - [ ] CORS uses configured origin.
  - [ ] Oversized audio upload is rejected.
  - [ ] Health/API responses do not expose secrets.

  **QA Scenarios**:
  ```text
  Scenario: Admin delete requires token and confirmation
    Tool: Bash (curl)
    Preconditions: Backend running.
    Steps:
      1. Attempt DELETE admin QA without token.
      2. Attempt DELETE with token but missing confirmation flag if implemented.
      3. Assert both are rejected appropriately.
    Expected Result: Destructive action guarded.
    Evidence: .sisyphus/evidence/task-18-admin-delete-guards.json

  Scenario: Oversized upload rejected
    Tool: Bash (curl)
    Preconditions: Backend running with MAX_AUDIO_UPLOAD_MB configured low for test.
    Steps:
      1. Upload file larger than limit to pipeline endpoint.
      2. Assert HTTP 413 or structured upload-too-large error.
    Expected Result: Request rejected before STT.
    Evidence: .sisyphus/evidence/task-18-upload-limit.json
  ```

  **Commit**: YES
  - Message: `fix(security): guard admin and upload surfaces`

- [ ] 19. Manual QA smoke assets and sample utterances

  **What to do**:
  - Create sample QA checklist and sample utterance list for thesis demo.
  - Include clear Indonesian campus question, informal/slang query, out-of-domain query, silence/empty audio, provider comparison, admin add/retrieve, TTS fallback, DB unavailable, Whisper unavailable.
  - Prepare reusable sample audio files or instructions for generating them if possible.

  **Must NOT do**:
  - Do not create a formal automated test suite.
  - Do not rely on a human-only checklist; each scenario must specify executable tool steps.

  **Recommended Agent Profile**:
  - **Category**: `quick` — QA scenario assets.
  - **Skills**: none.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Final verification
  - **Blocked By**: T4, T8, T9, T10, T12, T13, T16

  **References**:
  - Manual QA scenarios from Metis review.
  - `_MConverter.eu_test_retrieval.md:679-687` — failed case analysis fields for debugging inspiration.

  **Acceptance Criteria**:
  - [ ] QA document lists at least 12 executable scenarios.
  - [ ] Scenarios include happy paths and failure paths.
  - [ ] Sample utterances are Indonesian and thesis-domain relevant.

  **QA Scenarios**:
  ```text
  Scenario: QA checklist includes required failure modes
    Tool: Bash
    Preconditions: QA document exists.
    Steps:
      1. Search for "silence", "out-of-domain", "Supertonic", "OpenAI TTS", "DB unavailable", "Whisper unavailable", "admin".
      2. Assert all are present.
    Expected Result: QA checklist covers key edge cases.
    Evidence: .sisyphus/evidence/task-19-checklist-coverage.txt

  Scenario: Sample utterance can be used in pipeline
    Tool: Bash (curl)
    Preconditions: Backend running and at least one sample text/audio available.
    Steps:
      1. Run pipeline or retrieve endpoint with a sample utterance/normalized query.
      2. Assert debug response includes retrieval candidates or fallback.
    Expected Result: QA sample is actionable.
    Evidence: .sisyphus/evidence/task-19-sample-utterance.json
  ```

  **Commit**: YES
  - Message: `docs(qa): add thesis demo smoke scenarios`

- [ ] 20. UX polish for thesis demo flow

  **What to do**:
  - Polish visual hierarchy for live thesis demonstration.
  - Add clear stage progress indicators: STT, normalize, retrieve, rerank, answer, TTS.
  - Add provider badges and latency badges.
  - Prevent duplicate submissions while request is running.
  - Reset stale audio when a new request begins.

  **Must NOT do**:
  - Do not remove debug details for aesthetic simplicity.
  - Do not add unrelated animations/features that distract from thesis explanation.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering` — final UI polish.
  - **Skills**: `frontend-ui-ux`, `playwright` if available.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Final verification
  - **Blocked By**: T14, T15

  **References**:
  - T14 and T15 implemented UI flows.

  **Acceptance Criteria**:
  - [ ] Stage progress is visually clear.
  - [ ] Provider/latency info is easy to read.
  - [ ] Duplicate submit is blocked during processing.
  - [ ] UI remains stable with long candidate lists and errors.

  **QA Scenarios**:
  ```text
  Scenario: Stage progress and latency badges visible
    Tool: Playwright
    Preconditions: Frontend/backend running with mock or real pipeline response.
    Steps:
      1. Submit a request.
      2. Assert stage indicators appear for STT, Normalize, Retrieve, Rerank, TTS.
      3. Assert latency badges appear after completion.
      4. Capture screenshot.
    Expected Result: Demo-ready progression visualization.
    Evidence: .sisyphus/evidence/task-20-stage-progress.png

  Scenario: Duplicate submit blocked while processing
    Tool: Playwright
    Preconditions: Backend endpoint delayed or mock delay enabled.
    Steps:
      1. Start a request.
      2. Attempt to submit again immediately.
      3. Assert submit/record control is disabled or warning displayed.
    Expected Result: No duplicate concurrent request from same UI action.
    Evidence: .sisyphus/evidence/task-20-duplicate-submit.png
  ```

  **Commit**: YES
  - Message: `style(ui): polish thesis demo workflow`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify all Must Have items exist and all Must NOT Have constraints are respected. Check notebook-parity retrieval behavior, provider configurability, admin auth, no hardcoded secrets, and evidence files. Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run available type/build/lint commands for frontend/backend. Review changed files for unsafe secrets, broad catches without structured errors, production `console.log`, unused code, hardcoded paths, and AI slop. Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`.

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill for UI)
  Execute every QA scenario from tasks, including browser recording, mock/real audio pipeline, admin add/retrieve, fallback cases, and health checks. Save evidence under `.sisyphus/evidence/final-qa/`. Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`.

- [ ] F4. **Scope Fidelity Check** — `deep`
  Compare implementation diff against this plan. Ensure no training pipeline, no generated RAG, no streaming v1 creep, no advanced reranker creep, no unauthenticated admin mutation, and no hardcoded secrets. Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`.

---

## Commit Strategy

- **Wave 1**: small atomic commits per scaffold/config/UI/DB/docs task.
- **Wave 2**: one commit per provider/service: STT, Alpaca, vLLM, retrieval, TTS, admin API.
- **Wave 3**: one commit per integration surface: orchestrator, pipeline UI, admin UI, health/degradation.
- **Wave 4**: docs/security/QA/polish commits.
- **Final**: no final merge/commit until final verification wave approves.

---

## Success Criteria

### Verification Commands
```bash
# Backend health
curl -s http://localhost:8000/api/health

# DB health
curl -s http://localhost:8000/api/health/db

# Retrieval smoke
curl -s -X POST http://localhost:8000/api/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"informasi pembayaran UKT"}'

# Frontend smoke
# Open http://localhost:3000 and run Playwright QA scenarios
```

### Final Checklist
- [ ] App runs on GPU lab server with env-based config.
- [ ] Browser push-to-talk works.
- [ ] OpenAI Whisper transcribes audio.
- [ ] Alpaca model normalizes by default.
- [ ] vLLM model can be selected when configured.
- [ ] Retrieval uses existing PostgreSQL/pgvector data and notebook defaults.
- [ ] Rerank and threshold behavior match notebook.
- [ ] Thesis UI shows every pipeline stage and latency.
- [ ] Admin add/update/import regenerates embeddings with `BAAI/bge-m3`.
- [ ] Admin mutation requires token.
- [ ] Supertonic-3 TTS works or OpenAI fallback works.
- [ ] Text answer still appears if TTS fails.
- [ ] No training/fine-tuning or generated RAG added.
- [ ] No secrets or local paths hardcoded.
- [ ] Final verification wave approves.
