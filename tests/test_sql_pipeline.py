import pytest
import sys
from pathlib import Path

# Add project root to PYTHONPATH so we can import 'backend'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.sql.sql_agent import run_table_rag_pipeline, _get_sql_cache
from backend.sql.sql_cache import SQLCache

# Use a specific fake query to test semantic hitting
TEST_QUERY = "pytest unique completely new query xyz"
MOCK_SQL_SCRIPT = "SELECT 1 as test_col;"

@pytest.fixture(autouse=True)
def isolated_sql_cache(tmp_path, monkeypatch):
    """
    Fixture that runs before every test.
    Automatically replaces the default SQLCache paths with a temporary directory
    so we don't overwrite the real production FAISS cache.
    """
    test_index = tmp_path / "test_cache.faiss"
    test_meta = tmp_path / "test_cache_texts.json"

    # Create an isolated cache instance
    test_cache = SQLCache(index_path=test_index, metadata_path=test_meta)
    test_cache.initialize_cache()
    
    # Overwrite the lazy loader in sql_agent to return our isolated cache
    monkeypatch.setattr("backend.sql.sql_agent._get_sql_cache.instance", test_cache, raising=False)
    
    # Mock the lazy loader function itself just in case it wasn't initialized
    monkeypatch.setattr("backend.sql.sql_agent._get_sql_cache", lambda: test_cache)
    
    return test_cache

@pytest.fixture
def mock_llm_agent(monkeypatch):
    """
    Mocks the run_sql_agent function inside sql_agent.py.
    This prevents the pipeline from making real OpenAI API calls during tests,
    simulating a successful LLM generation.
    """
    def fake_run_sql_agent(query, schema_context, top_k):
        return {
            "sql": MOCK_SQL_SCRIPT,
            "result": [{"test_col": 1}],
            "error": None,
            "schema_context": schema_context,
        }
    
    monkeypatch.setattr("backend.sql.sql_agent.run_sql_agent", fake_run_sql_agent)

def test_cache_miss_triggers_agent_and_saves(mock_llm_agent, isolated_sql_cache):
    """
    Test 1: A brand new query should MISS the cache, hit the Agent pipeline,
    and then successfully save the resulting SQL to the cache.
    """
    res = run_table_rag_pipeline(TEST_QUERY)
    
    # Assert pipeline took the agent path
    assert res["path"] == "agent", "Should have taken the agent path on a cache miss."
    assert res["sql"] == MOCK_SQL_SCRIPT, "Should have used the SQL from the mocked agent."
    assert res["error"] is None
    assert len(res["result"]) > 0
    assert res["result"][0]["test_col"] == 1
    
    # Assert cache saved the result
    assert isolated_sql_cache.index.ntotal == 1, "FAISS index should contain exactly 1 embedded query."
    assert len(isolated_sql_cache.metadata) == 1, "Cache metadata should contain exactly 1 entry."
    assert isolated_sql_cache.metadata[0]["sql"] == MOCK_SQL_SCRIPT
    assert isolated_sql_cache.metadata[0]["question"] == TEST_QUERY


def test_cache_hit_bypasses_agent(mock_llm_agent, isolated_sql_cache):
    """
    Test 2: A query that already exists in the cache should HIT the semantic cache,
    bypassing the agent and returning the 'fast' path.
    """
    # Seed the cache first manually or by running the pipeline once
    isolated_sql_cache.add_to_cache(TEST_QUERY, MOCK_SQL_SCRIPT, "schema")
    isolated_sql_cache.save_cache()

    # Run the query
    res = run_table_rag_pipeline(TEST_QUERY)
    
    # Assert pipeline took the fast path
    assert res["path"] == "fast", "Should have taken the fast path due to semantic cache hit."
    assert res["sql"] == MOCK_SQL_SCRIPT, "Should have directly retrieved the cached SQL."
    assert res["error"] is None
    assert res["result"][0]["test_col"] == 1, "Should have executed the cached SQL successfully."


def test_different_query_triggers_new_miss(mock_llm_agent, isolated_sql_cache):
    """
    Test 3: Different semantic queries should not trigger a false-positive cache hit.
    """
    # Seed the cache with the standard test query
    isolated_sql_cache.add_to_cache(TEST_QUERY, MOCK_SQL_SCRIPT, "schema")
    isolated_sql_cache.save_cache()

    # Run a completely unrelated query
    diff_query = "What is the capital of France?"
    res = run_table_rag_pipeline(diff_query)
    
    # Assert pipeline recognized it as a miss and took the agent path
    assert res["path"] == "agent", "A completely different semantic query should miss the cache."
    assert isolated_sql_cache.index.ntotal == 2, "The new miss should have added a second entry to FAISS."
