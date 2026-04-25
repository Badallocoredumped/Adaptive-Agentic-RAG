import difflib
import json
import re

from backend import config


def extract_literal_values(sql: str) -> list[str]:
    """Extract string literals used in the SQL query."""
    # Matches strings in single quotes (commonly used in SQL)
    literals = re.findall(r"'([^']*)'", sql)
    
    clean_literals = []
    for lit in literals:
        # Remove wildcard characters if the agent used LIKE
        clean_lit = lit.strip('%')
        # Ignore empty strings or very short strings to avoid noise
        if len(clean_lit) > 1:
            clean_literals.append(clean_lit)
            
    return list(set(clean_literals))


def get_schema_sample_values() -> set[str]:
    """Extract all sample values stored in the TableRAG schema metadata."""
    meta_path = config.INDEX_DIR / "schema_meta.json"
    if not meta_path.exists():
        return set()
        
    sample_values = set()
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        for chunk in meta:
            if chunk.get("level") == "value":
                text = chunk.get("text", "")
                # Example text: "Column col in table tab contains values such as: 'val1', 'val2'."
                if "contains values such as:" in text:
                    literals = re.findall(r"'([^']*)'", text)
                    for lit in literals:
                        if len(lit) > 1:
                            sample_values.add(lit)
    except Exception as exc:
        if getattr(config, "DEBUG_LOGGING", False):
            print(f"[Candidate Predicate] Failed to read schema_meta.json: {exc}")
        
    return sample_values


def fuzzy_match_values(value: str, sample_values: set[str]) -> list[str]:
    """Find close matches for a value in the set of sample values."""
    if not sample_values:
        return []
        
    # cutoff=0.6 provides a good balance between catching typos and avoiding false positives
    matches = difflib.get_close_matches(value, sample_values, n=3, cutoff=0.6)
    
    # Filter out exact matches (case-insensitive) since those aren't typos
    return [m for m in matches if m.lower() != value.lower()]


def generate_candidate_predicates(sql: str) -> str | None:
    """Generate a hint string if any fuzzy matches are found for literals in the SQL."""
    literals = extract_literal_values(sql)
    if not literals:
        return None
        
    sample_values = get_schema_sample_values()
    if not sample_values:
        return None
        
    hints = []
    for lit in literals:
        matches = fuzzy_match_values(lit, sample_values)
        if matches:
            match_str = ", ".join(f"'{m}'" for m in matches)
            hints.append(f"You filtered by '{lit}', but close match(es) in the schema are: {match_str}")
            
    if hints:
        return "\n".join(hints)
    return None
