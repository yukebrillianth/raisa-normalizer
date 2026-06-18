from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import get_db

settings = get_settings()

app = FastAPI(title="Voice Assistant Thesis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health/db")
def db_health() -> dict:
    db = get_db()
    schema = db.schema
    return {
        "connected": db.connected,
        "table_exists": schema.table_exists,
        "row_count": schema.row_count,
        "vector_dimension": schema.vector_dimension,
        "id_strategy": schema.id_strategy,
        "pgvector_available": schema.pgvector_available,
    }
