"""Structured retrieval modules for PostgreSQL execution."""

from .database import PostgresDatabase, SQLiteDatabase, get_db_connection, get_db_cursor
from .executor import SQLExecutor
from .query_generator import SQLQueryGenerator
from .react_agent import build_react_agent, run_react_sql_agent
from .sql_agent import AgentResult, run_sql_agent, run_table_rag_pipeline

__all__ = [
    "PostgresDatabase",
    "SQLiteDatabase",       # backward-compat alias → PostgresDatabase
    "get_db_connection",
    "get_db_cursor",
    "SQLQueryGenerator",
    "SQLExecutor",
    "AgentResult",
    "run_sql_agent",
    "run_table_rag_pipeline",
    # ReAct agent — public entry point and factory for external use/testing
    "run_react_sql_agent",
    "build_react_agent",
]
