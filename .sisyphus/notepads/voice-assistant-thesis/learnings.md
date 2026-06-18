T1 learnings - 2026-06-18

- Next.js scaffold command worked: npx create-next-app@latest frontend --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*".
- Backend config uses pydantic-settings BaseSettings with SettingsConfigDict and Field aliases for uppercase env vars.
- Required env vars are DATABASE_URL, OPENAI_API_KEY, and ADMIN_TOKEN; missing vars are re-raised with a clear RuntimeError message.
- FastAPI app exposes GET / returning {"status": "ok"} and enables CORS from FRONTEND_ORIGIN.
- LSP import errors are expected before backend Python dependencies are installed.
