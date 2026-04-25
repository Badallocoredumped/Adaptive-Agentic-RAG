"""Database schema data classes.

Changes from original
---------------------
* ``to_embedding_texts()`` now produces richer per-table strings that include
  the table name in natural prose, all column names, and FK hints — improving
  the quality of the table-level FAISS chunk while staying backward-compatible.
* New ``to_llm_context()`` produces the full LLM-readable schema block used
  by the full_single_pass baseline and cache-miss fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    foreign_keys: List[Tuple[str, str, str]] = field(default_factory=list)
    # (local_col, referenced_table, referenced_col)


@dataclass
class SchemaInfo:
    tables: Dict[str, TableInfo]

    # ------------------------------------------------------------------
    # Embedding texts  (used by FAISS index — one string per table)
    # ------------------------------------------------------------------

    def to_embedding_texts(self) -> List[str]:
        """Produce one rich text string per table for the FAISS index.

        Format:
            "Table <name> with columns <col1> (<type>), <col2> (<type>), ...
             [— foreign keys: col → ref_table(ref_col), ...]"
        """
        texts = []
        for table_name, table_info in self.tables.items():
            cols = [f"{col.name} ({col.data_type})" for col in table_info.columns]
            text = f"Table {table_name} with columns {', '.join(cols)}"
            if table_info.foreign_keys:
                fks = [
                    f"{local_col} → {ref_table}({ref_col})"
                    for local_col, ref_table, ref_col in table_info.foreign_keys
                ]
                text += f" — foreign keys: {', '.join(fks)}"
            texts.append(text)
        return texts

    # ------------------------------------------------------------------
    # LLM-readable context block  (used by SQL generation prompts)
    # ------------------------------------------------------------------

    def to_llm_context(self, table_names: List[str] | None = None) -> str:
        """Return a structured, human-readable schema block for the LLM.

        If *table_names* is provided only those tables are included;
        otherwise all tables are rendered.

        Output format::

            Table: orders
            Columns:
              - id          INTEGER
              - customer_id INTEGER   FK → customers(id)
              - status      TEXT
            ...

            Foreign Key Relationships:
              orders.customer_id → customers.id
        """
        target = table_names if table_names is not None else list(self.tables.keys())
        lines: list[str] = []
        fk_lines: list[str] = []

        for tname in target:
            info = self.tables.get(tname)
            if info is None:
                continue
            fk_map = {fk[0]: (fk[1], fk[2]) for fk in info.foreign_keys}

            lines.append(f"Table: {tname}")
            lines.append("Columns:")
            for col in info.columns:
                ann = ""
                if col.name in fk_map:
                    ref_t, ref_c = fk_map[col.name]
                    ann = f"   FK → {ref_t}({ref_c})"
                    fk_lines.append(f"  {tname}.{col.name} → {ref_t}.{ref_c}")
                lines.append(f"  - {col.name:<30} {col.data_type.upper():<12}{ann}")
            lines.append("")

        if fk_lines:
            lines.append("Foreign Key Relationships:")
            seen: set[str] = set()
            for fl in fk_lines:
                if fl not in seen:
                    lines.append(fl)
                    seen.add(fl)
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Legacy dict format
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, List[str]]:
        """Legacy format: {table_name: [col1, col2, ...]}"""
        return {
            tname: [col.name for col in info.columns]
            for tname, info in self.tables.items()
        }
