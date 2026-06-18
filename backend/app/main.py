import asyncio
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import admin
from app.routers import health
from app.routers import pipeline

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title="Voice Assistant Thesis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/api/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


def _cleanup_old_audio() -> None:
    retention_seconds = settings.audio_retention_minutes * 60
    cutoff = time.time() - retention_seconds
    for f in AUDIO_DIR.iterdir():
        if f.is_file() and os.path.getmtime(f) < cutoff:
            logger.info("Cleaning up old audio file: %s", f.name)
            try:
                f.unlink()
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", f.name, exc)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(600)
        _cleanup_old_audio()


@app.on_event("startup")
async def _on_startup() -> None:
    threading.Thread(target=lambda: asyncio.run(_cleanup_loop()), daemon=True).start()
    logger.info("Audio cleanup scheduler started (interval: 10 min, retention: %d min)", settings.audio_retention_minutes)


app.include_router(admin.router)
app.include_router(health.router)
app.include_router(pipeline.router)
