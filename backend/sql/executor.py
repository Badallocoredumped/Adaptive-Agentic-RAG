"""SQL execution utilities for returning structured query outputs."""

from __future__ import annotations

from backend.sql.database import PostgresDatabase


class SQLExecutor:
    """Executes SQL queries against PostgreSQL and serializes rows into plain dicts."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def execute(self, sql_query: str) -> dict:
        """Execute a SQL query and return rows, columns, and status metadata."""
        try:
            import psycopg2.extras

            conn = self.database.get_connection()
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(sql_query)
                rows = cur.fetchall()
                # cursor.description: list of Column objects; [0] is the column name
                columns = [col[0] for col in (cur.description or [])]
                payload_rows = [dict(row) for row in rows]
            finally:
                conn.close()

            return {
                "ok": True,
                "query": sql_query,
                "columns": columns,
                "rows": payload_rows,
                "row_count": len(payload_rows),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "query": sql_query,
                "error": str(exc),
                "rows": [],
                "row_count": 0,
            }
