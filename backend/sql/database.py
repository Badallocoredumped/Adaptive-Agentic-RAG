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
        """Create tables if they do not already exist."""
        statements = [
            # ── Sales / E-commerce domain ──────────────────────────
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
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            """,
            # ── HR domain ──────────────────────────────────────────
            """
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                role TEXT,
                salary REAL,
                hire_date TEXT,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            );
            """,
            # ── Inventory / Warehouse domain ───────────────────────
            """
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY,
                product_id INTEGER,
                warehouse TEXT,
                stock_qty INTEGER,
                last_restock TEXT,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            """,
            # ── Customer Support domain ────────────────────────────
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                subject TEXT,
                priority TEXT,
                status TEXT,
                created_at TEXT,
                resolved_at TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            """,
            # ── Content / Blog domain ──────────────────────────────
            """
            CREATE TABLE IF NOT EXISTS blog_posts (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                category TEXT,
                views INTEGER,
                published_at TEXT
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

        # -- Customers (6 across 4 cities) --
        customers = [
            (1, "Alice Johnson", "Cairo", "2026-01-10"),
            (2, "Omar Hassan", "Alexandria", "2026-01-15"),
            (3, "Mona Adel", "Giza", "2026-01-20"),
            (4, "Youssef Samir", "Cairo", "2026-02-01"),
            (5, "Layla Mostafa", "Luxor", "2026-02-10"),
            (6, "Karim Nabil", "Cairo", "2026-03-01"),
        ]

        # -- Orders (12 orders, Jan–Mar, mixed statuses) --
        orders = [
            (1,  1, 120.50, "completed",  "2026-02-05"),
            (2,  1,  45.00, "completed",  "2026-02-07"),
            (3,  2,  88.25, "pending",    "2026-02-09"),
            (4,  3, 230.00, "completed",  "2026-02-11"),
            (5,  4,  15.75, "cancelled",  "2026-02-13"),
            (6,  2, 310.40, "completed",  "2026-02-15"),
            (7,  5,  75.00, "completed",  "2026-02-20"),
            (8,  6, 420.00, "completed",  "2026-03-01"),
            (9,  1, 200.00, "pending",    "2026-03-05"),
            (10, 3,  60.00, "cancelled",  "2026-03-08"),
            (11, 5, 150.00, "completed",  "2026-03-10"),
            (12, 6,  95.50, "pending",    "2026-03-15"),
        ]

        # -- Products (5 items across 3 categories) --
        products = [
            (1, "Cloud Basic",      "subscription", 29.99),
            (2, "Cloud Pro",        "subscription", 79.99),
            (3, "Premium Support",  "service",      49.99),
            (4, "Data Analytics",   "addon",        19.99),
            (5, "Enterprise Suite", "subscription", 149.99),
        ]

        # -- Order Items (line items linking orders to products) --
        order_items = [
            (1,  1, 2, 1),   # order 1 -> Cloud Pro
            (2,  1, 3, 1),   # order 1 -> Premium Support
            (3,  2, 1, 2),   # order 2 -> Cloud Basic x2
            (4,  3, 4, 3),   # order 3 -> Data Analytics x3
            (5,  4, 5, 1),   # order 4 -> Enterprise Suite
            (6,  4, 3, 1),   # order 4 -> Premium Support
            (7,  5, 1, 1),   # order 5 -> Cloud Basic
            (8,  6, 5, 2),   # order 6 -> Enterprise Suite x2
            (9,  7, 2, 1),   # order 7 -> Cloud Pro
            (10, 8, 5, 3),   # order 8 -> Enterprise Suite x3
            (11, 9, 2, 1),   # order 9 -> Cloud Pro
            (12, 9, 4, 2),   # order 9 -> Data Analytics x2
            (13, 10, 1, 1),  # order 10 -> Cloud Basic
            (14, 11, 2, 2),  # order 11 -> Cloud Pro x2
            (15, 12, 3, 1),  # order 12 -> Premium Support
        ]

        cursor.executemany(
            "INSERT OR IGNORE INTO customers (id, name, city, created_at) VALUES (?, ?, ?, ?);",
            customers,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO orders (id, customer_id, amount, status, created_at) VALUES (?, ?, ?, ?, ?);",
            orders,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO products (id, name, category, price) VALUES (?, ?, ?, ?);",
            products,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO order_items (id, order_id, product_id, quantity) VALUES (?, ?, ?, ?);",
            order_items,
        )

        # -- Departments --
        departments = [
            (1, "Engineering", 500000.00),
            (2, "Sales",       300000.00),
            (3, "Marketing",   200000.00),
            (4, "Legal",       150000.00),
        ]

        # -- Employees (8 across 4 departments) --
        employees = [
            (1, "Sara Ali",       1, "Senior Engineer",    95000.00, "2024-03-15"),
            (2, "Ahmed Fathy",    1, "Junior Engineer",    55000.00, "2025-08-01"),
            (3, "Nour Ibrahim",   2, "Sales Manager",      72000.00, "2024-06-20"),
            (4, "Hana Tamer",     2, "Sales Rep",          45000.00, "2025-11-10"),
            (5, "Tarek Zaki",     3, "Marketing Lead",     68000.00, "2024-01-05"),
            (6, "Dina Khaled",    3, "Content Specialist", 42000.00, "2025-09-22"),
            (7, "Rami Sayed",     4, "Legal Counsel",      88000.00, "2024-04-12"),
            (8, "Farida Mansour", 1, "DevOps Engineer",    78000.00, "2025-02-28"),
        ]

        # -- Inventory (stock across 3 warehouses) --
        inventory = [
            (1, 1, "Cairo-Main",   350, "2026-03-01"),
            (2, 2, "Cairo-Main",   120, "2026-02-20"),
            (3, 3, "Alex-North",    80, "2026-02-15"),
            (4, 4, "Cairo-Main",   500, "2026-03-10"),
            (5, 5, "Giza-West",     45, "2026-01-25"),
            (6, 1, "Alex-North",   200, "2026-03-05"),
        ]

        # -- Support Tickets --
        support_tickets = [
            (1, 1, "Cannot login to dashboard",     "high",   "open",     "2026-03-10", None),
            (2, 2, "Billing discrepancy",            "medium", "resolved", "2026-02-20", "2026-02-22"),
            (3, 3, "Feature request: export CSV",    "low",    "open",     "2026-03-12", None),
            (4, 1, "Slow API response times",        "high",   "resolved", "2026-02-25", "2026-02-27"),
            (5, 4, "Wrong product delivered",        "high",   "open",     "2026-03-14", None),
            (6, 6, "Need invoice copy",              "low",    "resolved", "2026-03-01", "2026-03-02"),
        ]

        # -- Blog Posts --
        blog_posts = [
            (1, "Getting Started with Cloud Pro", "Tarek Zaki",     "tutorial",     1250, "2026-01-15"),
            (2, "Q1 Sales Highlights",            "Nour Ibrahim",   "company-news", 870,  "2026-03-20"),
            (3, "Data Security Best Practices",   "Sara Ali",       "engineering",  2340, "2026-02-10"),
            (4, "Customer Success Stories",        "Dina Khaled",    "marketing",    560,  "2026-02-28"),
            (5, "Enterprise Suite Deep Dive",      "Ahmed Fathy",   "tutorial",    1890, "2026-03-05"),
        ]

        cursor.executemany(
            "INSERT OR IGNORE INTO departments (id, name, budget) VALUES (?, ?, ?);",
            departments,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO employees (id, name, department_id, role, salary, hire_date) VALUES (?, ?, ?, ?, ?, ?);",
            employees,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO inventory (id, product_id, warehouse, stock_qty, last_restock) VALUES (?, ?, ?, ?, ?);",
            inventory,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO support_tickets (id, customer_id, subject, priority, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            support_tickets,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO blog_posts (id, title, author, category, views, published_at) VALUES (?, ?, ?, ?, ?, ?);",
            blog_posts,
        )
