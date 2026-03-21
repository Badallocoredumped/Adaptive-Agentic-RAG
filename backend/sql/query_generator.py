"""Natural language to SQL generation stub for the MVP."""

from __future__ import annotations


class SQLQueryGenerator:
    """Generates SQL from user questions with simple rules and LLM placeholder."""

    def generate_sql(self, user_query: str) -> str:
        """Return a best-effort SQL query from a natural language request."""
        query = user_query.lower()

        if "revenue" in query or "sum" in query:
            return "SELECT SUM(amount) AS total_revenue FROM orders;"

        if "total" in query and "orders" in query:
            return "SELECT COUNT(*) AS total_orders FROM orders;"

        if "customers" in query and "city" in query:
            return "SELECT city, COUNT(*) AS total_customers FROM customers GROUP BY city;"

        return "SELECT id, name, city FROM customers LIMIT 5;"

    def generate_sql_with_llm(self, user_query: str) -> str:
        """Stub hook for future LLM-based NL-to-SQL generation."""
        return self.generate_sql(user_query)
