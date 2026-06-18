import logging
from dataclasses import dataclass, field
from typing import Optional

import psycopg2
from psycopg2 import sql

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SchemaInfo:
    table_exists: bool = False
    row_count: int = 0
    vector_dimension: Optional[int] = None
    id_strategy: str = "question"
    pgvector_available: bool = False
    columns: list[str] = field(default_factory=list)


class Database:
    """PostgreSQL connection layer using psycopg2 (matching notebook pattern).

    Uses autocommit=True for read-only queries to avoid transaction-aborted state.
    Never logs credentials. Gracefully degrades if DB is unreachable.
    """

    def __init__(self) -> None:
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._schema: SchemaInfo = SchemaInfo()
        self._settings = get_settings()
        self._connected: bool = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def schema(self) -> SchemaInfo:
        return self._schema

    def connect(self) -> None:
        """Connect to PostgreSQL using DATABASE_URL from config."""
        try:
            self._conn = psycopg2.connect(self._settings.database_url)
            self._conn.autocommit = True
            self._connected = True
            logger.info("Database connected successfully")
        except Exception as exc:
            self._connected = False
            logger.warning("Database connection failed (graceful degradation): %s", exc)

    def inspect_schema(self) -> SchemaInfo:
        """Inspect the QA_TABLE schema and cache results.

        Returns cached SchemaInfo if inspection was already performed.
        """
        if self._schema.table_exists:
            return self._schema

        if not self._connected:
            logger.warning("Schema inspection skipped: not connected to database")
            return self._schema

        try:
            self._inspect_pgvector()
            self._inspect_table()
        except Exception as exc:
            logger.warning("Schema inspection failed: %s", exc)
            self._schema = SchemaInfo(table_exists=False)

        return self._schema

    def startup(self) -> SchemaInfo:
        """Run connect + inspect on startup. Never crashes the app."""
        self.connect()
        return self.inspect_schema()

    def _inspect_pgvector(self) -> None:
        """Check if pgvector extension is installed."""
        assert self._conn is not None
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                row = cur.fetchone()
                self._schema.pgvector_available = bool(row[0]) if row else False
            logger.info("pgvector available: %s", self._schema.pgvector_available)
        except Exception as exc:
            logger.warning("pgvector check failed: %s", exc)

    def _inspect_table(self) -> None:
        """Inspect QA_TABLE: columns, row count, vector dimension, id strategy."""
        assert self._conn is not None
        table = self._settings.qa_table
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                    (table,),
                )
                row = cur.fetchone()
                exists = bool(row[0]) if row else False

            if not exists:
                self._schema.table_exists = False
                logger.info("Table %r does not exist", table)
                return

            self._schema.table_exists = True

            with self._conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = %s ORDER BY ordinal_position"
                    ),
                    (table,),
                )
                self._schema.columns = [row[0] for row in cur.fetchall()]

            self._inspect_row_count(table)
            self._inspect_vector_dim(table)
            self._determine_id_strategy()

            logger.info(
                "Table %r inspected: %d rows, %s strategy, pgvector=%s",
                table,
                self._schema.row_count,
                self._schema.id_strategy,
                self._schema.pgvector_available,
            )

        except Exception as exc:
            logger.warning("Table inspection for %r failed: %s", table, exc)
            self._schema.table_exists = False

    def _inspect_row_count(self, table: str) -> None:
        assert self._conn is not None
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
                row = cur.fetchone()
                self._schema.row_count = row[0] if row else 0
        except Exception as exc:
            logger.warning("Row count query failed: %s", exc)

    def _inspect_vector_dim(self, table: str) -> None:
        assert self._conn is not None
        """Detect embedding vector dimension from the first row's embedding column."""
        cols_lower = {c.lower() for c in self._schema.columns}
        if "embedding" not in cols_lower:
            return

        emb_col = next(c for c in self._schema.columns if c.lower() == "embedding")
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT array_length({}::real[], 1) FROM {} WHERE {} IS NOT NULL LIMIT 1").format(
                        sql.Identifier(emb_col),
                        sql.Identifier(table),
                        sql.Identifier(emb_col),
                    )
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    self._schema.vector_dimension = int(row[0])
                    logger.info("Vector dimension: %d", self._schema.vector_dimension)
        except Exception as exc:
            # If the embedding column isn't a real[] compatible type, try pgvector dimension()
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT vector_dims({}) FROM {} WHERE {} IS NOT NULL LIMIT 1").format(
                            sql.Identifier(emb_col),
                            sql.Identifier(table),
                            sql.Identifier(emb_col),
                        )
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        self._schema.vector_dimension = int(row[0])
            except Exception as inner_exc:
                logger.warning("Vector dimension detection failed: %s / %s", exc, inner_exc)

    def _determine_id_strategy(self) -> None:
        """If 'id' column exists, use it; otherwise use 'question' column."""
        cols_lower = {c.lower() for c in self._schema.columns}
        if "id" in cols_lower:
            self._schema.id_strategy = "id"
        else:
            self._schema.id_strategy = "question"

    def execute_query(self, query: str, params: tuple = ()) -> list[tuple]:
        """Safe read-only query execution. Returns list of rows."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._connected = False
            logger.info("Database connection closed")


_db: Optional[Database] = None


def get_db() -> Database:
    """Return the singleton Database instance, performing startup on first call."""
    global _db
    if _db is None:
        _db = Database()
        _db.startup()
    return _db