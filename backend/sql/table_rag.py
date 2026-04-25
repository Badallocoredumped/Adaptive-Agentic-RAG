"""TableRAG: semantic schema pruning for SQL generation.

Architecture
------------
The index stores THREE granularities of text per table so that the
embedding space can match queries from very different angles:

  1. TABLE-LEVEL  — table name + all column names (broad structural signal)
  2. COLUMN-LEVEL — one entry per column: "table.column — description"
                    (fine-grained column-name matching)
  3. VALUE-LEVEL  — sampled distinct cell values per column
                    (domain/entity matching, e.g. "Germany", "FRPM")

After FAISS retrieval the scoring is aggregated at the TABLE level (a
table's score = max score of any of its chunks) so a single highly
relevant column or cell value can surface its whole table.

FK expansion (score-gated)
--------------------------
Once the top-K tables are selected, FK-connected neighbours are only
pulled in when they pass at least one of three gates:

  GATE 1 — BOTH ENDS RELEVANT  (neighbour score >= FK_INCLUDE_THRESHOLD)
  GATE 2 — BRIDGE TABLE  (neighbour's own FKs touch 2+ selected tables)
  GATE 3 — SMALL LOOKUP TABLE  (<=FK_LOOKUP_MAX_COLS cols, score >= FK_LOOKUP_THRESHOLD)

Tables that fail all three gates are silently dropped — they are FK
neighbours in the schema but irrelevant to this query.

Output format
-------------
`retrieve_relevant_schema()` now returns a SINGLE formatted string
(not a list of raw embedding texts) that reads naturally for an LLM:

    ### Database Schema (relevant tables only)

    Table: orders
    Columns:
      - id          INTEGER   PK
      - customer_id INTEGER   FK → customers(id)
      - status      TEXT
      - amount      REAL
    Sample values for status: 'pending', 'shipped', 'cancelled'

    Table: customers
    Columns:
      - id   INTEGER  PK
      - name TEXT
      - country TEXT
    Sample values for country: 'Germany', 'France', 'USA'

    Foreign Key Relationships:
      orders.customer_id → customers.id
"""

from __future__ import annotations

import json
import sqlite3
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend import config
from backend.models import get_shared_st_model
from backend.sql.schema import SchemaInfo, TableInfo

# ---------------------------------------------------------------------------
# Index file paths
# ---------------------------------------------------------------------------
_SCHEMA_INDEX_PATH = config.INDEX_DIR / "schema.faiss"
_SCHEMA_META_PATH  = config.INDEX_DIR / "schema_meta.json"   # replaces schema_texts.json
# Keep the old texts path for backward-compat with callers that read it directly
_SCHEMA_TEXTS_PATH = config.INDEX_DIR / "schema_texts.json"

# How many distinct cell values to sample per column for value-level chunks
_VALUE_SAMPLE_SIZE = 5
# Columns whose values are too noisy to be useful (high-cardinality IDs etc.)
_SKIP_VALUE_COLS = re.compile(
    r"(^id$|_id$|_key$|_code$|uuid|hash|token|password|timestamp|created|updated)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# FK expansion thresholds (tunable via config overrides)
# ---------------------------------------------------------------------------

# Gate 1: minimum score for a FK neighbour to be included on its own merit
FK_INCLUDE_THRESHOLD: float = getattr(config, "FK_INCLUDE_THRESHOLD", 0.30)

# Gate 3: minimum score for a small lookup/dimension table to be included
FK_LOOKUP_THRESHOLD: float = getattr(config, "FK_LOOKUP_THRESHOLD", 0.20)

# Gate 3: max column count for a table to qualify as a "lookup/dimension table"
FK_LOOKUP_MAX_COLS: int = getattr(config, "FK_LOOKUP_MAX_COLS", 5)

# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _debug(message: str) -> None:
    if getattr(config, "DEBUG_LOGGING", False):
        print(message)


# ---------------------------------------------------------------------------
# E5 prefix helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _apply_e5_prefix(text: str, is_query: bool) -> str:
    if not config.E5_PREFIX_ENABLED or "e5" not in config.EMBEDDING_MODEL_NAME.lower():
        return text
    prefix = config.E5_QUERY_PREFIX if is_query else config.E5_PASSAGE_PREFIX
    stripped = text.strip()
    if stripped.lower().startswith(prefix.strip().lower()):
        return stripped
    return f"{prefix}{stripped}"


def _embed_texts(texts: list[str], is_query: bool) -> np.ndarray:
    model = get_shared_st_model()
    payload = [_apply_e5_prefix(t, is_query=is_query) for t in texts]
    vectors = model.encode(payload, convert_to_numpy=True, normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    if config.VECTOR_DISTANCE_METRIC.lower() == "cosine" and vectors.size > 0:
        faiss.normalize_L2(vectors)
    return vectors


# ---------------------------------------------------------------------------
# Cell-value sampling
# ---------------------------------------------------------------------------

def _sample_column_values(table_name: str, col_name: str, n: int = _VALUE_SAMPLE_SIZE) -> list[str]:
    """Return up to *n* distinct non-null values from a column (SQLite only)."""
    if not config.SQLITE_PATH:
        return []
    try:
        conn = sqlite3.connect(config.SQLITE_PATH)
        cur  = conn.cursor()
        # Use double-quoted identifiers to handle spaces/special chars
        cur.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
            f'WHERE "{col_name}" IS NOT NULL LIMIT {n};'
        )
        rows = cur.fetchall()
        conn.close()
        return [str(r[0]).strip() for r in rows if str(r[0]).strip()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Multi-granularity chunk generation
# ---------------------------------------------------------------------------

def _build_chunks(schema_info: SchemaInfo) -> list[dict[str, Any]]:
    """Return a list of chunk dicts, each with keys: table, text, level."""
    chunks: list[dict[str, Any]] = []

    for table_name, table_info in schema_info.tables.items():
        col_names = [c.name for c in table_info.columns]

        # ── 1. TABLE-LEVEL chunk ──────────────────────────────────────────
        fk_hints = []
        for local_col, ref_table, ref_col in table_info.foreign_keys:
            fk_hints.append(f"{table_name}.{local_col} references {ref_table}.{ref_col}")

        table_text = (
            f"Table {table_name} contains columns: {', '.join(col_names)}."
        )
        if fk_hints:
            table_text += " Foreign keys: " + "; ".join(fk_hints) + "."

        chunks.append({"table": table_name, "text": table_text, "level": "table"})

        # ── 2. COLUMN-LEVEL chunks ────────────────────────────────────────
        for col in table_info.columns:
            col_text = (
                f"Column {col.name} in table {table_name} "
                f"stores {col.data_type} data."
            )
            chunks.append({"table": table_name, "text": col_text, "level": "column"})

        # ── 3. VALUE-LEVEL chunks ─────────────────────────────────────────
        for col in table_info.columns:
            if _SKIP_VALUE_COLS.search(col.name):
                continue
            values = _sample_column_values(table_name, col.name)
            if not values:
                continue
            val_str   = ", ".join(f"'{v}'" for v in values[:_VALUE_SAMPLE_SIZE])
            value_text = (
                f"Column {col.name} in table {table_name} "
                f"contains values such as: {val_str}."
            )
            chunks.append({"table": table_name, "text": value_text, "level": "value"})

    return chunks


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_schema_index(schema_info: SchemaInfo) -> None:
    """Build and persist a multi-granularity FAISS index for schema pruning."""
    chunks = _build_chunks(schema_info)
    if not chunks:
        raise ValueError("schema_info is empty; cannot build schema index.")

    texts   = [c["text"] for c in chunks]
    vectors = _embed_texts(texts, is_query=False)
    dim     = int(vectors.shape[1])

    if config.VECTOR_DISTANCE_METRIC.lower() == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(vectors)

    _SCHEMA_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, config.win_short_path(_SCHEMA_INDEX_PATH))

    # Persist metadata (table → chunk mapping) and plain texts for compat
    meta = [{"table": c["table"], "level": c["level"], "text": c["text"]} for c in chunks]
    with _SCHEMA_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Legacy compat: schema_texts.json still written as plain list of texts
    with _SCHEMA_TEXTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Legacy helper used by sql_agent.py
# ---------------------------------------------------------------------------

def get_schema_texts(schema_info: SchemaInfo) -> list[str]:
    """Legacy shim — returns the table-level embedding texts only."""
    return schema_info.to_embedding_texts()


# ---------------------------------------------------------------------------
# Retrieve + score + FK-expand + format
# ---------------------------------------------------------------------------

def retrieve_relevant_schema(
    query: str,
    top_k: int = config.SQL_TOP_K,
    schema_info: SchemaInfo | None = None,
) -> list[str]:
    """Return a list with ONE element: the formatted schema context string.

    The single-element list keeps the call-site API identical to the old
    version (callers do ``'\n'.join(retrieve_relevant_schema(...))``) while
    the content is now a rich, LLM-friendly block.

    Args:
        query:       Natural-language question to answer.
        top_k:       Maximum number of *tables* to include (not chunks).
        schema_info: Optional live SchemaInfo used for FK expansion and
                     formatting.  If None, loaded from the live DB.
    """
    if not _SCHEMA_INDEX_PATH.exists() or not _SCHEMA_META_PATH.exists():
        # Graceful fallback when index has not been built yet
        if schema_info is None:
            from backend.sql.database import get_live_schema
            schema_info = get_live_schema()
        return [_format_schema_context(list(schema_info.tables.keys()), schema_info)]

    index = faiss.read_index(config.win_short_path(_SCHEMA_INDEX_PATH))
    with _SCHEMA_META_PATH.open("r", encoding="utf-8") as f:
        meta: list[dict] = json.load(f)

    if not meta:
        return []

    # ── Step 1: embed query and retrieve candidates ───────────────────────
    # Retrieve more candidates than needed (×4) so aggregation has room to work
    candidate_k = min(max(top_k * 4, 20), len(meta))
    query_vector = _embed_texts([query], is_query=True)
    scores, indices = index.search(query_vector, candidate_k)

    # ── Step 2: aggregate scores to TABLE level (max-pool) ────────────────
    table_scores: dict[str, float] = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        table_name = meta[idx]["table"]
        table_scores[table_name] = max(table_scores.get(table_name, -1.0), float(score))

    if not table_scores:
        return []

    # ── Step 3: rank tables and pick top-k ───────────────────────────────
    ranked = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)

    threshold: float | None = getattr(config, "SQL_SCHEMA_THRESHOLD", None)
    selected_tables: list[str] = []
    for i, (tname, tscore) in enumerate(ranked):
        if i == 0:
            selected_tables.append(tname)  # always keep the best match
            continue
        if len(selected_tables) >= top_k:
            break
        if threshold is not None and tscore < threshold:
            break
        selected_tables.append(tname)

    print(f"[TableRAG] Scores (threshold={threshold}, top_k={top_k}):")
    for tname, tscore in ranked:
        status = "PASS" if tname in selected_tables else "FAIL"
        print(f"  [{status}] {tname:<40} score={tscore:.4f}")
    print(f"[TableRAG] Selected: {selected_tables}")

    # ── Step 4: FK expansion ──────────────────────────────────────────────
    if schema_info is None:
        try:
            from backend.sql.database import get_live_schema
            schema_info = get_live_schema()
        except Exception:
            schema_info = None

    if schema_info is not None:
        selected_tables = _expand_with_fk_tables(selected_tables, schema_info, table_scores)
        print(f"[TableRAG] After FK expansion:   {selected_tables}")

    # Sort alphabetically so table order is deterministic and consistent with
    # format_full_schema — avoids "lost in the middle" LLM degradation from
    # relevance-score ordering placing the most query-relevant table mid-schema.
    selected_tables = sorted(selected_tables)
    _debug(f"[TableRAG] Final order (sorted):         {selected_tables}")

    # ── Step 5: format output ─────────────────────────────────────────────
    formatted = _format_schema_context(selected_tables, schema_info)
    return [formatted]


# ---------------------------------------------------------------------------
# FK expansion helper
# ---------------------------------------------------------------------------

def _expand_with_fk_tables(
    selected: list[str],
    schema_info: SchemaInfo,
    table_scores: dict[str, float],
) -> list[str]:
    """Selectively add FK-connected tables using three relevance gates.

    A FK neighbour is included only when it passes at least one gate:

    GATE 1 — BOTH ENDS RELEVANT
        The neighbour's similarity score >= FK_INCLUDE_THRESHOLD.
        Both anchor and partner scored well independently.

    GATE 2 — BRIDGE TABLE
        The neighbour's own FKs touch 2+ already-selected tables
        (classic many-to-many bridge, e.g. order_items → orders + products).
        Included regardless of its own score.

    GATE 3 — SMALL LOOKUP / DIMENSION TABLE
        The neighbour has <= FK_LOOKUP_MAX_COLS columns AND a modest score
        >= FK_LOOKUP_THRESHOLD.  Tiny reference tables (status codes,
        category labels) are cheap and often critical for value matching.

    Tables that fail all three gates are silently dropped.
    """
    selected_set = set(selected)
    additions: list[str] = []

    # Pre-build reverse-FK map: ref_table → set of tables pointing to it
    reverse_fk: dict[str, set[str]] = {}
    for tname, tinfo in schema_info.tables.items():
        for _lc, ref_table, _rc in tinfo.foreign_keys:
            reverse_fk.setdefault(ref_table, set()).add(tname)

    def _passes_gates(neighbour: str, anchor: str) -> bool:
        n_info = schema_info.tables.get(neighbour)
        if n_info is None:
            return False

        n_score = table_scores.get(neighbour, 0.0)

        # Gate 1: neighbour has strong independent relevance
        if n_score >= FK_INCLUDE_THRESHOLD:
            return True

        # Gate 2: neighbour is a bridge between two already-selected tables
        neighbour_fk_targets = {ref for _lc, ref, _rc in n_info.foreign_keys}
        if len(neighbour_fk_targets & selected_set) >= 2:
            return True
        # Also: anchor points to neighbour AND neighbour points to another selected table
        anchor_fk_targets = {
            ref for _lc, ref, _rc in
            (schema_info.tables[anchor].foreign_keys if anchor in schema_info.tables else [])
        }
        if neighbour in anchor_fk_targets and len(neighbour_fk_targets & selected_set) >= 1:
            return True

        # Gate 3: small lookup/dimension table with modest score
        if len(n_info.columns) <= FK_LOOKUP_MAX_COLS and n_score >= FK_LOOKUP_THRESHOLD:
            return True

        return False

    # Outgoing FKs: selected table → neighbour
    for anchor in list(selected_set):
        anchor_info = schema_info.tables.get(anchor)
        if not anchor_info:
            continue
        for _lc, ref_table, _rc in anchor_info.foreign_keys:
            if ref_table in selected_set:
                continue
            if _passes_gates(ref_table, anchor):
                selected_set.add(ref_table)
                additions.append(ref_table)

    # Incoming FKs: neighbour → selected table (reverse direction)
    for anchor in list(selected_set):
        for neighbour in list(reverse_fk.get(anchor, set())):
            if neighbour in selected_set:
                continue
            if _passes_gates(neighbour, anchor):
                selected_set.add(neighbour)
                additions.append(neighbour)

    return selected + additions


# ---------------------------------------------------------------------------
# Human-readable schema formatter
# ---------------------------------------------------------------------------

def _format_schema_context(
    table_names: list[str],
    schema_info: SchemaInfo | None,
) -> str:
    """Produce a clean, LLM-readable schema block for the selected tables.

    Output example:
        ### Database Schema (relevant tables only)

        Table: orders
        Columns:
          - id          INTEGER   PRIMARY KEY
          - customer_id INTEGER   FK → customers(id)
          - status      TEXT
        Sample values for status: 'pending', 'shipped', 'cancelled'

        Foreign Key Relationships:
          orders.customer_id → customers.id
    """
    if not table_names:
        return ""

    lines: list[str] = ["### Database Schema (relevant tables only)\n"]
    fk_lines: list[str] = []
    seen_tables = set()

    for tname in table_names:
        if tname in seen_tables:
            continue
        seen_tables.add(tname)

        if schema_info is None or tname not in schema_info.tables:
            # Fallback: just emit the table name
            lines.append(f"Table: {tname}\n")
            continue

        table_info: TableInfo = schema_info.tables[tname]

        # Identify PK columns (we infer by name heuristic if not in schema)
        pk_cols = {c.name for c in table_info.columns if c.name.lower() in ("id", f"{tname.lower()}_id")}
        fk_col_map = {fk[0]: (fk[1], fk[2]) for fk in table_info.foreign_keys}

        lines.append(f"Table: {tname}")
        lines.append("Columns:")
        for col in table_info.columns:
            annotations = []
            if col.name in pk_cols:
                annotations.append("PRIMARY KEY")
            if col.name in fk_col_map:
                ref_t, ref_c = fk_col_map[col.name]
                annotations.append(f"FK → {ref_t}({ref_c})")
                fk_lines.append(f"  {tname}.{col.name} → {ref_t}.{ref_c}")
            ann_str = f"   [{', '.join(annotations)}]" if annotations else ""
            lines.append(f"  - {col.name:<30} {col.data_type.upper():<12}{ann_str}")

        # Sample values for non-id text/varchar columns
        value_samples: list[str] = []
        for col in table_info.columns:
            if _SKIP_VALUE_COLS.search(col.name):
                continue
            if col.data_type.lower() not in ("text", "varchar", "char", "string", "nvarchar", "real", "integer", "int", "numeric", "float"):
                continue
            vals = _sample_column_values(tname, col.name, n=3)
            if vals:
                val_str = ", ".join(f"'{v}'" for v in vals)
                value_samples.append(f"  Sample values for {col.name}: {val_str}")

        if value_samples:
            lines.extend(value_samples)

        lines.append("")  # blank line between tables

    if fk_lines:
        # Deduplicate
        seen_fk: set[str] = set()
        unique_fk = [l for l in fk_lines if l not in seen_fk and not seen_fk.add(l)]  # type: ignore[func-returns-value]
        lines.append("Foreign Key Relationships:")
        lines.extend(unique_fk)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: format ALL tables (used by bird_eval full_single_pass baseline)
# ---------------------------------------------------------------------------

def format_full_schema(schema_info: SchemaInfo) -> str:
    """Format the entire schema (no pruning) in the same LLM-readable format."""
    return _format_schema_context(list(schema_info.tables.keys()), schema_info)
