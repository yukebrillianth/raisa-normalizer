from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(..., alias="DATABASE_URL")
    qa_table: str = Field("qa_ground_truth", alias="QA_TABLE")
    frontend_origin: str = Field("http://localhost:3000", alias="FRONTEND_ORIGIN")

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    stt_provider: str = Field("openai_whisper", alias="STT_PROVIDER")
    openai_whisper_model: str = Field("whisper-1", alias="OPENAI_WHISPER_MODEL")

    embedding_model_name: str = Field("BAAI/bge-m3", alias="EMBEDDING_MODEL_NAME")
    embedding_device: str = Field("cuda", alias="EMBEDDING_DEVICE")

    normalizer_provider: str = Field("alpaca", alias="NORMALIZER_PROVIDER")
    normalizer_base_model_path: str = Field("", alias="NORMALIZER_BASE_MODEL_PATH")
    normalizer_lora_path: str = Field("", alias="NORMALIZER_LORA_PATH")
    normalizer_vllm_base_url: str = Field("", alias="NORMALIZER_VLLM_BASE_URL")
    normalizer_vllm_model: str = Field("", alias="NORMALIZER_VLLM_MODEL")
    normalizer_max_input_tokens: int = Field(128, alias="NORMALIZER_MAX_INPUT_TOKENS")
    normalizer_max_new_tokens: int = Field(128, alias="NORMALIZER_MAX_NEW_TOKENS")
    verbalizer_provider: str = Field("alpaca", alias="VERBALIZER_PROVIDER")
    verbalizer_max_new_tokens: int = Field(192, alias="VERBALIZER_MAX_NEW_TOKENS")
    verbalizer_enabled: bool = Field(True, alias="VERBALIZER_ENABLED")

    retrieval_top_k: int = Field(3, alias="RETRIEVAL_TOP_K")
    retrieval_candidate_k: int = Field(8, alias="RETRIEVAL_CANDIDATE_K")
    retrieval_bm25_k: int = Field(3, alias="RETRIEVAL_BM25_K")
    retrieval_rrf_k: int = Field(60, alias="RETRIEVAL_RRF_K")
    retrieval_intent_prior_enabled: bool = Field(True, alias="RETRIEVAL_INTENT_PRIOR_ENABLED")
    retrieval_name_intent_boost: float = Field(0.004, alias="RETRIEVAL_NAME_INTENT_BOOST")
    retrieval_name_intent_penalty: float = Field(0.003, alias="RETRIEVAL_NAME_INTENT_PENALTY")
    retrieval_similarity_threshold: float = Field(0.75, alias="RETRIEVAL_SIMILARITY_THRESHOLD")
    rerank_keyword_weight: float = Field(0.2, alias="RERANK_KEYWORD_WEIGHT")
    selection_candidate_k: int = Field(8, alias="SELECTION_CANDIDATE_K")
    fallback_answer: str = Field(
        "Maaf, saya belum dapat menemukan jawaban yang sesuai untuk pertanyaan Anda.",
        alias="FALLBACK_ANSWER",
    )

    tts_provider: str = Field("supertonic", alias="TTS_PROVIDER")
    supertonic_total_steps: int = Field(5, alias="SUPERTONIC_TOTAL_STEPS")
    supertonic_voice: str = Field("M1", alias="SUPERTONIC_VOICE")
    openai_tts_model: str = Field("tts-1", alias="OPENAI_TTS_MODEL")
    openai_tts_voice: str = Field("alloy", alias="OPENAI_TTS_VOICE")

    admin_token: str = Field(..., alias="ADMIN_TOKEN")
    audio_retention_minutes: int = Field(60, alias="AUDIO_RETENTION_MINUTES")
    max_recording_seconds: int = Field(30, alias="MAX_RECORDING_SECONDS")
    max_audio_upload_mb: int = Field(25, alias="MAX_AUDIO_UPLOAD_MB")
    pipeline_stream_mode: str = Field("sse", alias="PIPELINE_STREAM_MODE")


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = [str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variable(s): {names}") from exc
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
