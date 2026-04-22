"""Database helpers supporting SQLite and PostgreSQL backends."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras
from abc import ABC, abstractmethod

from backend import config
from backend.sql.schema import SchemaInfo, TableInfo, ColumnInfo


# ---------------------------------------------------------------------------
# Connection factory (used by startup / schema init code)
# ---------------------------------------------------------------------------

def get_db_connection():
    """Return a new DB connection — sqlite3 or psycopg2 depending on config.

    When SQLITE_PATH is set in the environment, returns a sqlite3.Connection
    with row_factory=sqlite3.Row so rows are dict-accessible.
    Otherwise returns a psycopg2 connection using DATABASE_URL or PG_* vars.

    Caller is responsible for closing the connection.
    """
    if config.SQLITE_PATH:
        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    if config.DATABASE_URL:
        return psycopg2.connect(config.DATABASE_URL)

    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        dbname=config.PG_DB,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
    )


# ---------------------------------------------------------------------------
# Hot-path connection reuse (SQLite thread-local + PostgreSQL pool)
# ---------------------------------------------------------------------------

_sqlite_local = threading.local()


def _get_sqlite_conn():
    """Return a thread-local reusable SQLite connection."""
    conn = getattr(_sqlite_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        _sqlite_local.conn = conn
    return conn


_pg_pool = None
_pg_pool_lock = threading.Lock()


def _get_pg_pool():
    """Lazy-init a simple connection pool for PostgreSQL (thread-safe)."""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is None:
            from psycopg2.pool import SimpleConnectionPool

            if config.DATABASE_URL:
                _pg_pool = SimpleConnectionPool(1, 4, dsn=config.DATABASE_URL)
            else:
                _pg_pool = SimpleConnectionPool(
                    1, 4,
                    host=config.PG_HOST,
                    port=config.PG_PORT,
                    dbname=config.PG_DB,
                    user=config.PG_USER,
                    password=config.PG_PASSWORD,
                )
    return _pg_pool


def execute_query(sql: str) -> list[dict[str, Any]]:
    """Execute a SELECT query and return rows as a list of dicts.

    Works with both SQLite and PostgreSQL backends.
    Uses thread-local connections (SQLite) or a connection pool (PostgreSQL)
    to avoid per-query connection setup overhead.
    Raises RuntimeError on any DB error.
    """
    if config.SQLITE_PATH:
        conn = _get_sqlite_conn()
        cur = conn.cursor()
        try:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            raise RuntimeError(f"SQL execution failed: {exc}\nQuery: {sql}") from exc
        finally:
            cur.close()

    pool = _get_pg_pool()
    conn = pool.getconn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rows = [dict(row) for row in cur.fetchall()]
        conn.commit()  # close the implicit transaction
        return rows
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"SQL execution failed: {exc}\nQuery: {sql}") from exc
    finally:
        pool.putconn(conn)



class BaseDatabase(ABC):
    """Abstract base class for database implementations."""

    @abstractmethod
    def get_schema(self) -> SchemaInfo:
        """Introspect the database and return the schema structure."""
        pass

    @abstractmethod
    def execute(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only query and return rows."""
        pass


def get_live_schema() -> SchemaInfo:
    """Return the live DB schema as a SchemaInfo object.

    Works with both SQLite and PostgreSQL backends.
    """
    if config.SQLITE_PATH:
        return _get_sqlite_schema()
    
    db = PostgresDatabase()
    return db.get_schema()


def _get_sqlite_schema() -> SchemaInfo:
    """Read schema from SQLite via sqlite_master + PRAGMA table_info."""
    schema_info = SchemaInfo(tables={})
    conn = sqlite3.connect(config.SQLITE_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cur.fetchall()]
        for table in tables:
            cur.execute(f"PRAGMA table_info({table});")
            # PRAGMA row: (cid, name, type, notnull, dflt_value, pk)
            columns = []
            for row in cur.fetchall():
                col_name = row[1]
                data_type = str(row[2]).lower() if row[2] else "text"
                nullable = not bool(row[3])
                columns.append(ColumnInfo(name=col_name, data_type=data_type, nullable=nullable))
            
            # Simple SQLite FK extraction using PRAGMA foreign_key_list
            fks = []
            cur.execute(f"PRAGMA foreign_key_list({table});")
            for fk_row in cur.fetchall():
                # fk_row = (id, seq, table, from, to, on_update, on_delete, match)
                fks.append((fk_row[3], fk_row[2], fk_row[4]))
                
            schema_info.tables[table] = TableInfo(name=table, columns=columns, foreign_keys=fks)
    finally:
        conn.close()
    return schema_info


@contextmanager
def get_db_cursor(commit: bool = False) -> Generator[Any, None, None]:
    """Context manager that yields a cursor and handles cleanup.

    For PostgreSQL yields a RealDictCursor; for SQLite yields a standard
    cursor whose connection already has row_factory=sqlite3.Row.

    Args:
        commit: If True, commit the transaction on clean exit.
    """
    conn = get_db_connection()
    try:
        if config.SQLITE_PATH:
            cur = conn.cursor()
            yield cur
            if commit:
                conn.commit()
        else:
            with conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                yield cur
                if commit:
                    conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class PostgresDatabase(BaseDatabase):
    """PostgreSQL database implementation with connection pooling and introspection."""

    def get_connection(self) -> psycopg2.extensions.connection:
        """Return a raw psycopg2 connection (caller must close it)."""
        return get_db_connection()
        
    def execute(self, sql: str) -> list[dict[str, Any]]:
        return execute_query(sql)

    def get_schema(self) -> SchemaInfo:
        """Read schema from PostgreSQL via information_schema."""
        schema_info = SchemaInfo(tables={})
        conn = self.get_connection()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Fetch tables
            cur.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            )
            tables = [row["table_name"] for row in cur.fetchall()]
            
            # Fetch columns and FKs for each table
            for table_name in tables:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (table_name,),
                )
                
                columns = []
                for row in cur.fetchall():
                    nullable = row["is_nullable"].upper() == "YES"
                    columns.append(ColumnInfo(
                        name=row["column_name"], 
                        data_type=row["data_type"], 
                        nullable=nullable
                    ))
                
                # Fetch Foreign Keys
                cur.execute(
                    """
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM 
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
                    """,
                    (table_name,)
                )
                
                fks = [
                    (row["column_name"], row["foreign_table_name"], row["foreign_column_name"])
                    for row in cur.fetchall()
                ]
                
                schema_info.tables[table_name] = TableInfo(
                    name=table_name, 
                    columns=columns, 
                    foreign_keys=fks
                )
                
        finally:
            conn.close()
            
        return schema_info

