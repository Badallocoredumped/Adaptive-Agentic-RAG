"""SQL execution utilities for returning structured query outputs."""

from __future__ import annotations

from backend.sql.database import SQLiteDatabase


class SQLExecutor:
    """Executes SQL queries and serializes rows into plain dictionaries."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def execute(self, sql_query: str) -> dict:
        """Execute a SQL query and return rows, columns, and status metadata."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description or []]

            payload_rows = [dict(row) for row in rows]
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
