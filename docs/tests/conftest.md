# `conftest.py` — Global Pytest Fixtures

This file defines **project-wide pytest fixtures** for mocking external dependencies and isolating tests from real IO, model inference, or database access.

---

## Purpose
- Automatically patch ChromaDB clients, AI/LLM service layers, and the vector store
- Ensure all tests run with controlled, fast, side-effect free mocks
- Centralize test setup logic for reuse and maintainability

---

## Main Fixtures

### `patch_chromadb_client` (autouse, session)
- Patches `chromadb.Client` and `chromadb.PersistentClient` everywhere before imports
- Returns a mock collection for all `.get_or_create_collection()` calls
- Ensures all vector store/database operations in tests are no-ops

### `mock_all_services` (autouse, function)
- Patches:
  - `OllamaService.generate_response` (returns "Mocked AI response")
  - `EntityExtractor.analyze_message` (returns mock entities/intents)
  - `Summarizer.summarize_conversation` and `.generate_key_points` (return mock summary/key points)
- All service instances in tests use these safe, deterministic return values

### `mock_vector_store`
- Provides a mock vector store with no-op implementations for get/add/delete methods

---

## How to Use
- **Do not import or call these fixtures manually**—pytest auto-applies them to all test files in the suite.
- Place this file at the root of your `tests/` directory.
- Add new fixture logic here as you add new external dependencies or service layers to your backend.

---

## Example Docstring for Test File
```python
"""
tests/conftest.py

Global pytest fixtures for:
- Patching ChromaDB and all external model dependencies (Ollama, Summarizer, EntityExtractor)
- Providing a safe, mock vector store

This allows all service and API tests to run quickly and without real network/model calls.
"""
```

---

## Why Centralize in `conftest.py`?
- **Consistency:** All tests run with the same mocks, making failures reliable and reproducible.
- **Maintainability:** Update mocking logic in one place as your app evolves.
- **Performance:** No waiting for real database/model inference in tests.
- **Safety:** No risk of accidentally hitting production services or databases during testing.

---

*Update this file as you add new dependencies or service integrations requiring test isolation!*