# IRIS, Indonesian Campus Information Voice Assistant

A push-to-talk voice assistant for Indonesian campus information. Built as a senior thesis debug/demo web app, IRIS transcribes spoken questions, normalizes informal Indonesian into formal queries using fine-tuned LLMs, retrieves answers from a PostgreSQL database, selects the best candidate through LLM reasoning, and speaks the answer back to the user. Every pipeline stage streams to the frontend in real time for transparent, thesis-grade visibility.

> **Scope**: This is a lab/demo application (v1). It handles push-to-talk audio uploads. It does not stream audio in real time, and it does not train or fine-tune models. Model weights are loaded from local paths you configure.

---

## Architecture

```text
┌─────────┐    ┌──────────────────────┐    ┌──────────────┐    ┌───────────────────────────────┐
│ Browser │───▶│ FastAPI Backend (SSE)│───▶│ PostgreSQL   │    │   Local GPU Models            │
│ Next.js │◀───│ /api/pipeline/...    │◀───│ + pgvector   │    │   (loaded from local paths)   │
│         │    └──────────────────────┘    └──────────────┘    └───────────────────────────────┘
└─────────┘              │
                         │  Pipeline stages (streamed via SSE):
                         │
                         ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │ 1. STT          OpenAI Whisper API                               │
              │ 2. Normalize    Fine-tuned Alpaca-format LLM (or vLLM optional)  │
              │ 3. Embed        BAAI/bge-m3 via SentenceTransformer               │
              │ 4. Retrieve     pgvector cosine top-3 + keyword hybrid rerank     │
              │ 5. Select+Verb  LLM picks best candidate, rephrases for speech    │
              │ 6. TTS          Supertonic-3 (primary) / OpenAI TTS (fallback)    │
              │ 7. Stream       SSE events to frontend, stage by stage            │
              └──────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend (Next.js) |
| CUDA-capable GPU | 8 GB+ VRAM recommended | For local model inference |
| PostgreSQL | 14+ with `pgvector` extension | Stores QA data and embeddings |
| ffmpeg | Current stable package | Required for browser audio decoding and conversion paths used by STT/TTS tooling |
| OpenAI API key | Required | Whisper STT and optional TTS fallback |

---

## Environment Setup

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

### Required Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (see [Database](#database) section) |
| `OPENAI_API_KEY` | Your OpenAI API key for Whisper STT and optional TTS |
| `ADMIN_TOKEN` | Secret token for admin endpoints (pick a strong random string) |

### Optional / Configurable Variables

#### General

| Variable | Default | Description |
|---|---|---|
| `QA_TABLE` | `qa_ground_truth` | PostgreSQL table containing QA pairs with embeddings |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | CORS origin for the frontend |

#### Speech-to-Text

| Variable | Default | Description |
|---|---|---|
| `STT_PROVIDER` | `openai_whisper` | STT provider (currently only `openai_whisper`) |
| `OPENAI_WHISPER_MODEL` | `whisper-1` | Whisper model variant |

#### Embedding

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | SentenceTransformer model name |
| `EMBEDDING_DEVICE` | `cuda` | Device for embedding (`cuda` or `cpu`) |

#### Normalizer (query rewriting)

| Variable | Default | Description |
|---|---|---|
| `NORMALIZER_PROVIDER` | `alpaca` | `alpaca` (default) or `vllm` (optional) |
| `NORMALIZER_BASE_MODEL_PATH` | _(empty)_ | Local path to base LLM (e.g., Ministral-3-14B-Instruct) |
| `NORMALIZER_LORA_PATH` | _(empty)_ | Local path to LoRA adapter directory |
| `NORMALIZER_VLLM_BASE_URL` | _(empty)_ | vLLM server URL (only if using `vllm` provider) |
| `NORMALIZER_VLLM_MODEL` | _(empty)_ | vLLM model name (only if using `vllm` provider) |
| `NORMALIZER_MAX_INPUT_TOKENS` | `128` | Max input tokens for normalization prompt |
| `NORMALIZER_MAX_NEW_TOKENS` | `128` | Max generated tokens for normalization |

#### Verbalizer (spoken answer rephrasing)

| Variable | Default | Description |
|---|---|---|
| `VERBALIZER_PROVIDER` | `alpaca` | Same model as normalizer |
| `VERBALIZER_MAX_NEW_TOKENS` | `192` | Max tokens for verbalization output |
| `VERBALIZER_ENABLED` | `true` | Toggle verbalization on/off |

#### Retrieval

| Variable | Default | Description |
|---|---|---|
| `RETRIEVAL_TOP_K` | `3` | Number of candidates to retrieve |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity to accept a candidate |
| `RERANK_KEYWORD_WEIGHT` | `0.2` | Weight for keyword overlap in hybrid reranking |
| `FALLBACK_ANSWER` | `Maaf, saya belum dapat menemukan...` | Default answer when no candidate passes threshold |

#### Text-to-Speech

| Variable | Default | Description |
|---|---|---|
| `TTS_PROVIDER` | `supertonic` | Primary TTS: `supertonic` or `openai` |
| `SUPERTONIC_TOTAL_STEPS` | `5` | Supertonic-3 diffusion steps |
| `SUPERTONIC_VOICE` | `M1` | Supertonic-3 voice preset |
| `OPENAI_TTS_MODEL` | `tts-1` | OpenAI TTS model |
| `OPENAI_TTS_VOICE` | `alloy` | OpenAI TTS voice name |

#### Limits

| Variable | Default | Description |
|---|---|---|
| `AUDIO_RETENTION_MINUTES` | `60` | How long uploaded audio files are kept |
| `MAX_RECORDING_SECONDS` | `30` | Max browser recording duration |
| `MAX_AUDIO_UPLOAD_MB` | `25` | Max audio file upload size |
| `PIPELINE_STREAM_MODE` | `sse` | Pipeline output mode |

---

## Model Paths

The app loads model weights from your local filesystem. Set these in your `.env`:

```bash
# Example paths (adjust to your server layout):
NORMALIZER_BASE_MODEL_PATH=/home/user/models/Ministral-3-14B-Instruct-2512
NORMALIZER_LORA_PATH=/home/user/models/pipeline/experiments/training_r16_2e4/models/lora_adapters
```

The base model is a standard HuggingFace model directory. Download it before starting the backend, then point `NORMALIZER_BASE_MODEL_PATH` at the directory that contains files such as `config.json`, tokenizer files, and model weight shards. If your lab uses HuggingFace gated models, log in on the GPU server first:

```bash
huggingface-cli login
huggingface-cli download mistralai/Ministral-3-14B-Instruct-2512 \
  --local-dir /home/user/models/Ministral-3-14B-Instruct-2512
```

The LoRA adapter path points to a directory containing adapter weights from prior training, usually files such as `adapter_config.json` and `adapter_model.safetensors`. Training itself is done separately, not through this app. Copy the finished adapter directory to the GPU server and set `NORMALIZER_LORA_PATH` to that directory:

```bash
rsync -av ./models/lora_adapters/ user@gpu-server:/home/user/models/iris-lora/
```

For Supertonic-3 TTS, install the Python package from `backend/requirements.txt`, then predownload the model on the GPU server before demo day so the first request does not stall on model fetch. If your Supertonic setup requires a HuggingFace token, log in with `huggingface-cli login` on the server account that runs the backend. The model is cached under the normal HuggingFace cache directory, commonly `~/.cache/huggingface`, unless you set `HF_HOME`:

```bash
export HF_HOME=/home/user/.cache/huggingface
python - <<'PY'
import supertonic
print("Supertonic package import OK. Run one short TTS request from the app to warm the cache.")
PY
```

For embeddings, `BAAI/bge-m3` is downloaded automatically by HuggingFace on first run (requires internet) or cached locally at your HuggingFace cache path.

If you use the optional vLLM provider, start a vLLM server separately and point `NORMALIZER_VLLM_BASE_URL` and `NORMALIZER_VLLM_MODEL` at it.

---

## Database

### Connection String Format

```bash
DATABASE_URL=postgresql://username:password@hostname:5432/database_name
```

### pgvector Extension

The database must have the `pgvector` extension enabled. Run this once as a superuser:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verify the extension from the same database account used by the app:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

You can also confirm vector operators are available with a tiny smoke test:

```sql
SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance;
```

If either query fails, install the PostgreSQL `pgvector` package for your OS, restart PostgreSQL if needed, reconnect to the target database, and run `CREATE EXTENSION IF NOT EXISTS vector;` again.

### Expected Table

The app queries a table (default `qa_ground_truth`) with at least these columns:

| Column | Type | Description |
|---|---|---|
| `question` | `TEXT` | The question text |
| `answer` | `TEXT` | The answer text |
| `embedding` | `VECTOR(1024)` | BGE-M3 embedding vector |

An `id` column is recommended for admin CRUD operations. The app performs schema inspection at startup and reports any issues.

---

## Backend Setup

Install ffmpeg before Python dependencies. Examples:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Equivalent explicit install command for GPU lab notes
pip install fastapi "uvicorn[standard]" pydantic-settings python-multipart python-dotenv sentence-transformers httpx openai supertonic psycopg2-binary soundfile numpy torch transformers peft

# 4. Make sure your .env file is in the project root (one level up)

# 5. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The backend starts at `http://localhost:8000`. On startup it validates required environment variables and reports any missing configuration.

---

## Frontend Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the dev server
npm run dev
```

The frontend starts at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` or configure `FRONTEND_ORIGIN` in the backend `.env` to match.

---

## Running the App

Start both servers in separate terminals:

```bash
# Terminal 1, Backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2, Frontend
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser. Press the record button, speak an Indonesian campus question, and watch the pipeline stages stream in real time. If you access the frontend from another laptop by GPU server IP or hostname, use HTTPS or SSH port forwarding. Browsers only allow microphone capture on secure origins, except for `localhost`.

---

## Admin Panel

Admin endpoints are protected by the `ADMIN_TOKEN` you set in `.env`. Pass the token as a header:

```bash
# Example: list QA rows
curl http://localhost:8000/api/admin/qa \
  -H "Authorization: Bearer your-admin-token-here"

# Example: add a QA row (embedding is generated automatically)
curl -X POST http://localhost:8000/api/admin/qa \
  -H "Authorization: Bearer your-admin-token-here" \
  -H "Content-Type: application/json" \
  -d '{"question": "Di mana lokasi helpdesk akademik?", "answer": "Helpdesk akademik berada di kantor akademik lantai 1."}'
```

The frontend also includes an admin panel where you can enter your token to manage QA data, import CSVs, and regenerate embeddings.

---

## Troubleshooting

### Microphone Not Working

Browser microphone access requires HTTPS on any address other than `localhost`. If you deploy to a lab server and access it by IP address, the browser will block microphone access over plain HTTP even if the frontend loads correctly.

Solutions:

- Access the app via `localhost` (e.g., SSH port-forward: `ssh -L 3000:localhost:3000 user@gpu-server`)
- Set up HTTPS with a self-signed certificate, a local lab certificate authority, or a reverse proxy such as nginx or Caddy
- If the backend is also remote, expose it through HTTPS too and set `NEXT_PUBLIC_API_URL` to that HTTPS URL
- After changing certificates or ports, fully reload the browser tab and grant microphone permission again

### GPU Out of Memory (OOM)

Loading a 14B parameter model alongside the embedding model can exhaust VRAM on smaller GPUs. If you see CUDA OOM errors:

- Reduce GPU memory pressure by running inference serially (the app does this by default)
- Use a machine with more VRAM. 16 GB+ is recommended for smoother local 14B model demos, and more is better if other jobs share the GPU
- Set `EMBEDDING_DEVICE=cpu` to move embeddings off the GPU
- Close other GPU processes (`nvidia-smi` to check what's running)
- Restart the backend after a CUDA OOM. PyTorch may keep fragmented memory until the process exits
- Lower generation limits such as `NORMALIZER_MAX_NEW_TOKENS` and `VERBALIZER_MAX_NEW_TOKENS` for demos that only need short answers
- Keep only one backend worker on a single GPU. Do not run multiple `uvicorn` workers unless each worker has enough VRAM for its own model copy
- Warm up one request before presenting, then leave the process running so model loading does not happen during the demo
- If the GPU is still too small, use the `vllm` provider on a separate inference server or switch the embedding device to CPU first, since that is the lowest-risk memory reduction

### Database Unavailable

If the backend reports `DATABASE_URL` errors or `pgvector` extension missing:

- Verify PostgreSQL is running and accessible from the backend
- Check that the `vector` extension is installed (`SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';`)
- Confirm the table exists: `SELECT * FROM qa_ground_truth LIMIT 1;`
- Check your `DATABASE_URL` format and credentials

### OpenAI API Unavailable

If STT fails and you see OpenAI-related errors:

- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check that the server has outbound internet access
- Verify your OpenAI account has active credits and API access
- If TTS fallback also fails, the app returns a text-only answer with an error in the debug response

---

## What This App Does NOT Do

- **No model training or fine-tuning.** The app loads pre-trained model weights. Training is performed separately using external tooling.
- **No streaming audio in v1.** Audio input is push-to-talk: you press record, speak, stop, and the complete audio is uploaded for processing.
- **No production deployment guarantees.** This is a lab/demo application for thesis research and demonstration.

---

## Project Structure

```text
├── backend/           FastAPI backend
│   ├── app/
│   │   ├── main.py        Application entry point
│   │   ├── config.py      Environment settings loader
│   │   └── ...            Pipeline providers and services
│   └── requirements.txt
├── frontend/          Next.js frontend
│   ├── app/
│   └── package.json
├── .env.example       Environment variable template
└── README.md          This file
```
