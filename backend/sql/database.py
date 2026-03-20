"""SQLite database helpers for schema creation and connections."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteDatabase:
    """Simple SQLite manager for this MVP."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)

    def get_connection(self) -> sqlite3.Connection:
        """Return a sqlite3 connection with row access by column name."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_schema(self) -> None:
        """Create minimal example tables if they do not already exist."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT,
                created_at TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount REAL,
                status TEXT,
                created_at TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            """,
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for statement in statements:
                cursor.execute(statement)

            self._seed_sample_data(cursor)
            conn.commit()

    @staticmethod
    def _seed_sample_data(cursor: sqlite3.Cursor) -> None:
        """Insert deterministic sample rows so SQL/hybrid demos return meaningful values."""
        customers = [
            (1, "Alice Johnson", "Cairo", "2026-01-10"),
            (2, "Omar Hassan", "Alexandria", "2026-01-15"),
            (3, "Mona Adel", "Giza", "2026-01-20"),
            (4, "Youssef Samir", "Cairo", "2026-02-01"),
        ]

        orders = [
            (1, 1, 120.50, "completed", "2026-02-05"),
            (2, 1, 45.00, "completed", "2026-02-07"),
            (3, 2, 88.25, "pending", "2026-02-09"),
            (4, 3, 230.00, "completed", "2026-02-11"),
            (5, 4, 15.75, "cancelled", "2026-02-13"),
            (6, 2, 310.40, "completed", "2026-02-15"),
        ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO customers (id, name, city, created_at)
            VALUES (?, ?, ?, ?);
            """,
            customers,
        )

        cursor.executemany(
            """
            INSERT OR IGNORE INTO orders (id, customer_id, amount, status, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            orders,
        )
