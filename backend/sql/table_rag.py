"""TableRAG: LLM-based schema selection for SQL generation.

Architecture
------------
Schema selection is the task of deciding which tables from a database are
relevant to a natural-language query before handing off to the SQL agent.

Previous approach (embedding similarity) fails for same-domain databases
because all tables score similarly — the embedding model cannot bridge the
gap between natural language ("test takers", "excellence rate") and
abbreviated column identifiers (NumTstTakr, NumGE1500).

Current approach: LLM-as-Schema-Selector
-----------------------------------------
Aligned with CHESS (Talaei et al., 2024) and XiYan-SQL, the selection is
done by a single LLM call that receives:

  1. The full compact schema  — all table names + column names + types +
     sample values + FK relationships, formatted to be token-efficient.
  2. The user query.

The LLM responds with a JSON list of relevant table names. Because the LLM
has world knowledge, it can:
  - Decode abbreviations: NumTstTakr → number of test takers
  - Understand paraphrases: "excellence rate" → NumGE1500
  - Infer joins: "female customers with credit cards" → client + card + disp
  - Handle same-domain databases: all banking tables scored equally before;
    now the LLM reads the columns and makes an informed decision.

FAISS index
-----------
The FAISS index (build_schema_index / schema.faiss) is kept for backward
compatibility with the SQL golden-query cache (SQLCache) and the
schema_lookup tool in the ReAct agent. It is no longer used for the primary
schema selection path.

Output format
-------------
`retrieve_relevant_schema()` returns a list with ONE element: the full
formatted schema string for the selected tables, ready for the SQL agent.

    ### Database Schema (relevant tables only)

    Table: orders
    Columns:
      - id          INTEGER
      - customer_id INTEGER   [FK → customers(id)]
      - status      TEXT
    Sample values for status: 'pending', 'shipped', 'cancelled'

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


def search_schema_index(query: str, top_k: int = 5) -> list[str]:
    """Retrieve relevant schema chunks directly from the FAISS index."""
    if not _SCHEMA_INDEX_PATH.exists() or not _SCHEMA_META_PATH.exists():
        return []
        
    try:
        index = faiss.read_index(config.win_short_path(_SCHEMA_INDEX_PATH))
        with _SCHEMA_META_PATH.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            
        vector = _embed_texts([query], is_query=True)
        distances, indices = index.search(vector, top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= int(idx) < len(meta):
                results.append(meta[int(idx)]["text"])
        return results
    except Exception as exc:
        _debug(f"[TableRAG] FAISS search failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Retrieve + LLM-select + format
# ---------------------------------------------------------------------------

def retrieve_relevant_schema(
    query: str,
    top_k: int = config.SQL_TOP_K,
    schema_info: SchemaInfo | None = None,
) -> list[str]:
    """Return a list with ONE element: the formatted schema context string.

    Uses a single LLM call to select relevant tables from the full schema.
    Falls back to the full schema if the LLM call fails for any reason.

    Args:
        query:       Natural-language question to answer.
        top_k:       Soft upper bound on tables passed to the LLM selector.
                     The LLM may return fewer; this only caps the compact
                     schema fed into the prompt (not the output).
        schema_info: Live SchemaInfo object.  If None, loaded from the DB.
    """
    if schema_info is None:
        try:
            from backend.sql.database import get_live_schema
            schema_info = get_live_schema()
        except Exception as exc:
            _debug(f"[TableRAG] Could not load live schema: {exc}")
            return []

    all_tables = list(schema_info.tables.keys())

    # If the database is tiny (≤ 3 tables), skip the LLM call — just return all.
    if len(all_tables) <= 3:
        _debug(f"[TableRAG] ≤3 tables — returning full schema without LLM call.")
        return [_format_schema_context(sorted(all_tables), schema_info)]

    # ── Step 1: build compact schema for the selector prompt ─────────────
    compact = _build_compact_schema(schema_info)

    # ── Step 2: call LLM to select relevant tables ────────────────────────
    selected_tables = _llm_select_tables(query, compact, all_tables)

    if not selected_tables:
        # LLM returned nothing parseable — fall back to full schema
        _debug("[TableRAG] LLM selector returned empty — falling back to full schema.")
        selected_tables = all_tables

    selected_tables = sorted(selected_tables)
    _debug(f"[TableRAG] LLM-selected tables: {selected_tables}")

    from backend.sql.join_path import compute_join_tree
    
    join_hint = ""
    # Avoid massive hints if fallback selected all tables in a huge schema
    if len(selected_tables) <= 20:
        required_set, join_hint = compute_join_tree(schema_info, selected_tables)
        expanded_tables = sorted(list(required_set))
        if set(expanded_tables) != set(selected_tables):
            _debug(f"[TableRAG] Graph expanded tables to include bridge tables: {expanded_tables}")
            selected_tables = expanded_tables

    # ── Step 3: format and return ─────────────────────────────────────────
    return [_format_schema_context(selected_tables, schema_info, join_hint=join_hint)]


# ---------------------------------------------------------------------------
# Compact schema builder  (input to the LLM selector prompt)
# ---------------------------------------------------------------------------

def _build_compact_schema(schema_info: SchemaInfo) -> str:
    """Build a token-efficient schema string for the LLM selector prompt.

    Format per table (one line per table, columns inline):
        table_name (col1: TYPE, col2: TYPE, ...) [FK: col → other_table.col, ...]
        Sample values — col: 'val1', 'val2'; col2: 'val1'

    This is intentionally more compact than _format_schema_context so that
    large schemas (50+ tables) fit in a single prompt.
    """
    lines: list[str] = []

    for tname, tinfo in schema_info.tables.items():
        # Column summary
        col_parts = []
        fk_map = {fk[0]: (fk[1], fk[2]) for fk in tinfo.foreign_keys}
        for col in tinfo.columns:
            entry = f"{col.name}: {col.data_type.upper()}"
            if col.name in fk_map:
                ref_t, ref_c = fk_map[col.name]
                entry += f" →{ref_t}.{ref_c}"
            col_parts.append(entry)

        fk_str = ""
        if tinfo.foreign_keys:
            fk_parts = [
                f"{lc}→{rt}.{rc}"
                for lc, rt, rc in tinfo.foreign_keys
            ]
            fk_str = f"  [FK: {', '.join(fk_parts)}]"

        lines.append(f"Table {tname}: {', '.join(col_parts)}{fk_str}")

        # Sample values (up to 3 per non-id text/numeric column, max 4 cols)
        sampled = 0
        val_parts: list[str] = []
        for col in tinfo.columns:
            if sampled >= 4:
                break
            if _SKIP_VALUE_COLS.search(col.name):
                continue
            vals = _sample_column_values(tname, col.name, n=3)
            if vals:
                val_parts.append(f"{col.name}: {', '.join(repr(v) for v in vals)}")
                sampled += 1
        if val_parts:
            lines.append(f"  Values — {'; '.join(val_parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM selector call
# ---------------------------------------------------------------------------

_SELECTOR_SYSTEM_PROMPT = """\
You are a database schema expert. Given a user question and a database schema, \
identify which tables are needed to answer the question.

Rules:
- Include ALL tables that contain columns needed for SELECT, WHERE, JOIN, GROUP BY, or ORDER BY.
- Include join/bridge tables that connect two relevant tables even if they have no direct column match.
- Do NOT include tables that are completely unrelated to the question.
- Respond ONLY with a valid JSON array of table name strings, e.g. ["orders", "customers"].
- No explanation, no markdown fences, just the raw JSON array.\
"""

_SELECTOR_USER_TEMPLATE = """\
Question: {query}

Database schema:
{compact_schema}

Which tables are needed? Respond with a JSON array of table names only.\
"""


def _llm_select_tables(
    query: str,
    compact_schema: str,
    all_tables: list[str],
) -> list[str]:
    """Call the LLM to select relevant tables. Returns [] on any failure."""
    try:
        from openai import OpenAI  # noqa: PLC0415

        api_key  = getattr(config, "OPENAI_API_KEY", "") or ""
        base_url = getattr(config, "LLM_BASE_URL", "") or ""
        model    = getattr(config, "SQL_OPENAI_MODEL", "gpt-4o-mini")

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        user_msg = _SELECTOR_USER_TEMPLATE.format(
            query=query,
            compact_schema=compact_schema,
        )

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SELECTOR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        _debug(f"[TableRAG] LLM selector raw response: {raw!r}")
        return _parse_table_list(raw, all_tables)

    except Exception as exc:
        _debug(f"[TableRAG] LLM selector call failed: {exc}")
        return []


def _parse_table_list(raw: str, all_tables: list[str]) -> list[str]:
    """Parse the LLM's JSON array response into a validated list of table names.

    Accepts:
      - Clean JSON array:   ["orders", "customers"]
      - With markdown:      ```json\n["orders"]\n```
      - Single table:       "orders"  (rare fallback)

    Any name not in all_tables is silently dropped. Returns [] if nothing valid.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
        text = text.strip()

    valid_set = {t.lower(): t for t in all_tables}

    # Try JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                name = str(item).strip()
                canonical = valid_set.get(name.lower())
                if canonical:
                    result.append(canonical)
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract quoted strings or bare words that match table names
    candidates = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\b\w+\b)', text)
    result = []
    seen: set[str] = set()
    for groups in candidates:
        for name in groups:
            if not name:
                continue
            canonical = valid_set.get(name.lower())
            if canonical and canonical not in seen:
                result.append(canonical)
                seen.add(canonical)

    return result


# ---------------------------------------------------------------------------
# Human-readable schema formatter
# ---------------------------------------------------------------------------

def _format_schema_context(
    table_names: list[str],
    schema_info: SchemaInfo | None,
    join_hint: str = "",
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

    if join_hint:
        lines.append(f"System Note: To connect these tables, you MUST use the following join path:\n{join_hint}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: format ALL tables (used by bird_eval full_single_pass baseline)
# ---------------------------------------------------------------------------

def format_full_schema(schema_info: SchemaInfo) -> str:
    """Format the entire schema (no pruning) in the same LLM-readable format."""
    return _format_schema_context(list(schema_info.tables.keys()), schema_info)
