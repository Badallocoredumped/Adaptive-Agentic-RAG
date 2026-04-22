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
    foreign_keys: List[Tuple[str, str, str]] = field(default_factory=list)  # (col, ref_table, ref_col)


@dataclass
class SchemaInfo:
    tables: Dict[str, TableInfo]

    def to_embedding_texts(self) -> List[str]:
        """Produce rich text string per table for FAISS embedding."""
        texts = []
        for table_name, table_info in self.tables.items():
            cols = [f"{col.name} ({col.data_type})" for col in table_info.columns]
            text = f"Table {table_name} with columns {', '.join(cols)}"
            
            if table_info.foreign_keys:
                fks = [f"{local_col} → {ref_table}({ref_col})" for local_col, ref_table, ref_col in table_info.foreign_keys]
                text += f" — foreign keys: {', '.join(fks)}"
            
            texts.append(text)
        return texts

    def to_dict(self) -> Dict[str, List[str]]:
        """Legacy format for backward backward compatibility: {table: [col1, col2, ...]}"""
        return {
            table_name: [col.name for col in table_info.columns]
            for table_name, table_info in self.tables.items()
        }
