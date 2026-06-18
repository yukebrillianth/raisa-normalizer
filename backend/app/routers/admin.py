import csv
import io
import logging

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.db import Database, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/qa", tags=["admin"])
security = HTTPBearer()


class QARowCreate(BaseModel):
    question: str
    answer: str


class QARowUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None


class QARowResponse(BaseModel):
    id: str
    question: str
    answer: str
    embedding_generated: bool = False


class ImportResult(BaseModel):
    total: int = 0
    imported: int = 0
    errors: list[dict[str, object]] = Field(default_factory=list)


class RegenerateResult(BaseModel):
    id: str
    question: str
    embedding_regenerated: bool


_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = SentenceTransformer(settings.embedding_model_name)
    return _embedding_model


def _generate_embedding(question: str) -> list[float]:
    model = _get_embedding_model()
    embedding = model.encode(question, show_progress_bar=False)
    return embedding.tolist()


def _get_embedding_col(settings) -> str:
    db = get_db()
    cols = db.schema.columns
    for c in cols:
        if c.lower() == "embedding":
            return c
    return "embedding"


async def _verify_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> None:
    settings = get_settings()
    if not settings.admin_token:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not configured")
    if credentials.credentials != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def _get_write_conn():
    settings = get_settings()
    return psycopg2.connect(settings.database_url)


def _id_query(db: Database, identifier: str) -> tuple[str, tuple[str]]:
    strategy = db.schema.id_strategy
    if strategy == "id":
        return f"WHERE {strategy} = %s", (identifier,)
    else:
        return "WHERE question = %s", (identifier,)


def _id_filter(db: Database, identifier: str) -> tuple[str, tuple[str, ...]]:
    strategy = db.schema.id_strategy
    if strategy == "id":
        return f"WHERE {strategy} = %s", (identifier,)
    else:
        return "WHERE question = %s", (identifier,)


def _dict_row(columns: list[str], row: tuple) -> dict[str, object]:
    result: dict[str, object] = {}
    for i, col in enumerate(columns):
        if i < len(row):
            result[col] = row[i]
    return result


def _id_value(db: Database, row: dict[str, object]) -> str:
    if db.schema.id_strategy == "id" and "id" in row:
        return str(row["id"])
    return str(row.get("question", ""))


@router.get("")
async def list_qa(
    search: str | None = Query(None),
    db: Database = Depends(get_db),
    _: None = Depends(_verify_admin),
):
    table = get_settings().qa_table
    cols = db.schema.columns
    try:
        if search:
            query = (
                f"SELECT * FROM {table} "
                f"WHERE question ILIKE %s OR answer ILIKE %s "
                f"ORDER BY question LIMIT 200"
            )
            pattern = f"%{search}%"
            rows = db.execute_query(query, (pattern, pattern))
        else:
            rows = db.execute_query(
                f"SELECT * FROM {table} ORDER BY question LIMIT 200"
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    results = [_dict_row(cols, r) for r in rows]
    return [
        {
            "id": _id_value(db, r),
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "embedding_generated": r.get("embedding") is not None,
        }
        for r in results
    ]


@router.get("/{identifier}")
async def get_qa(
    identifier: str,
    db: Database = Depends(get_db),
    _: None = Depends(_verify_admin),
):
    table = get_settings().qa_table
    where, params = _id_filter(db, identifier)
    try:
        rows = db.execute_query(
            f"SELECT * FROM {table} {where} LIMIT 1",
            params,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail="QA row not found")

    r = _dict_row(db.schema.columns, rows[0])
    return {
        "id": _id_value(db, r),
        "question": r.get("question", ""),
        "answer": r.get("answer", ""),
        "embedding_generated": r.get("embedding") is not None,
    }


@router.post("", status_code=201)
async def create_qa(
    body: QARowCreate,
    _: None = Depends(_verify_admin),
):
    settings = get_settings()
    db = get_db()
    table = settings.qa_table
    question = body.question.strip()
    answer = body.answer.strip()

    if not question or not answer:
        raise HTTPException(status_code=422, detail="question and answer are required")

    embedding = _generate_embedding(question)
    emb_col = _get_embedding_col(settings)
    columns = ["question", "answer", emb_col]
    values = ",".join(["%s"] * len(columns))

    conn = _get_write_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table} (question, answer, {emb_col}) VALUES ({values}) RETURNING *",
                (question, answer, embedding),
            )
            row = cur.fetchone()
            conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Create QA failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to create row: {exc}")
    finally:
        conn.close()

    assert row is not None
    r = _dict_row(db.schema.columns, row)
    return {
        "id": _id_value(db, r),
        "question": r.get("question", ""),
        "answer": r.get("answer", ""),
        "embedding_generated": True,
    }


@router.put("/{identifier}")
async def update_qa(
    identifier: str,
    body: QARowUpdate,
    db: Database = Depends(get_db),
    _: None = Depends(_verify_admin),
):
    settings = get_settings()
    table = settings.qa_table
    where, where_params = _id_filter(db, identifier)

    try:
        rows = db.execute_query(
            f"SELECT * FROM {table} {where} LIMIT 1", where_params
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail="QA row not found")

    existing = _dict_row(db.schema.columns, rows[0])

    new_question = body.question.strip() if body.question else None
    new_answer = body.answer.strip() if body.answer else None
    question_changed = new_question and new_question != str(existing.get("question", ""))

    conn = _get_write_conn()
    try:
        with conn.cursor() as cur:
            if question_changed:
                assert new_question is not None
                embedding = _generate_embedding(new_question)
                emb_col = _get_embedding_col(settings)
                if new_answer:
                    cur.execute(
                        f"UPDATE {table} SET question=%s, answer=%s, {emb_col}=%s {where}",
                        (new_question, new_answer, embedding) + where_params,
                    )
                else:
                    cur.execute(
                        f"UPDATE {table} SET question=%s, {emb_col}=%s {where}",
                        (new_question, embedding) + where_params,
                    )
            elif new_answer:
                cur.execute(
                    f"UPDATE {table} SET answer=%s {where}",
                    (new_answer,) + where_params,
                )
            elif new_question:
                assert isinstance(new_question, str)
                embedding = _generate_embedding(new_question)
                emb_col = _get_embedding_col(settings)
                cur.execute(
                    f"UPDATE {table} SET question=%s, {emb_col}=%s {where}",
                    (new_question, embedding) + where_params,
                )
            else:
                conn.close()
                return QARowResponse(
                    id=identifier,
                    question=str(existing.get("question", "")),
                    answer=str(existing.get("answer", "")),
                    embedding_generated=existing.get("embedding") is not None,
                )

            conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Update QA failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to update row: {exc}")
    finally:
        conn.close()

    return {
        "id": identifier,
        "question": new_question if new_question else str(existing.get("question", "")),
        "answer": new_answer if new_answer else str(existing.get("answer", "")),
        "embedding_generated": question_changed or (new_question is not None and new_question != str(existing.get("question", ""))),
    }


@router.delete("/{identifier}")
async def delete_qa(
    identifier: str,
    confirm: str | None = Query(None),
    db: Database = Depends(get_db),
    _: None = Depends(_verify_admin),
):
    if confirm != "true":
        raise HTTPException(
            status_code=400,
            detail="Deletion requires ?confirm=true query parameter",
        )

    settings = get_settings()
    table = settings.qa_table
    where, where_params = _id_filter(db, identifier)

    try:
        rows = db.execute_query(
            f"SELECT * FROM {table} {where} LIMIT 1", where_params
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail="QA row not found")

    conn = _get_write_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} {where}", where_params)
            conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Delete QA failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to delete row: {exc}")
    finally:
        conn.close()

    return {"detail": "Deleted", "id": identifier}


@router.post("/import", status_code=201)
async def import_csv(
    request: Request,
    _: None = Depends(_verify_admin),
):
    settings = get_settings()
    table = settings.qa_table
    emb_col = _get_embedding_col(settings)

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=400,
            detail="Expected multipart/form-data upload with a CSV file",
        )

    form = await request.form()
    file_field = next(iter(form.values()), None)
    if file_field is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not hasattr(file_field, "file"):
        raise HTTPException(status_code=400, detail="Invalid file upload")

    raw = await file_field.read()
    text = raw.decode("utf-8-sig")

    result = ImportResult()
    try:
        reader = csv.DictReader(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {exc}")

    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV has no header row")

    fields = [f.strip().lower() for f in reader.fieldnames]
    if "question" not in fields or "answer" not in fields:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain 'question' and 'answer' columns",
        )

    conn = _get_write_conn()
    try:
        with conn.cursor() as cur:
            for i, row in enumerate(reader, start=1):
                result.total += 1
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()

                if not question or not answer:
                    result.errors.append(
                        {"row": i, "error": "Missing question or answer"}
                    )
                    continue

                try:
                    embedding = _generate_embedding(question)
                    cur.execute(
                        f"INSERT INTO {table} (question, answer, {emb_col}) VALUES (%s, %s, %s)",
                        (question, answer, embedding),
                    )
                    result.imported += 1
                except Exception as exc:
                    result.errors.append(
                        {"row": i, "error": str(exc)}
                    )
                    conn.rollback()
                    conn = _get_write_conn()

        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("CSV import failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}")
    finally:
        conn.close()

    return result.model_dump()


@router.post("/{identifier}/regenerate-embedding")
async def regenerate_embedding(
    identifier: str,
    db: Database = Depends(get_db),
    _: None = Depends(_verify_admin),
):
    settings = get_settings()
    table = settings.qa_table
    where, where_params = _id_filter(db, identifier)
    emb_col = _get_embedding_col(settings)

    try:
        rows = db.execute_query(
            f"SELECT * FROM {table} {where} LIMIT 1", where_params
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail="QA row not found")

    r = _dict_row(db.schema.columns, rows[0])
    question = str(r.get("question", ""))

    if not question:
        raise HTTPException(status_code=422, detail="QA row has no question text")

    embedding = _generate_embedding(question)

    conn = _get_write_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table} SET {emb_col}=%s {where}",
                (embedding,) + where_params,
            )
            conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Regenerate embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to regenerate embedding: {exc}")
    finally:
        conn.close()

    return {
        "id": identifier,
        "question": question,
        "embedding_regenerated": True,
    }