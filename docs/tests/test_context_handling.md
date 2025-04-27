# Test File: `test_context_handling.py`

This test suite verifies **semantic context extraction and formatting** in the conversation service logic, ensuring that:
- Relevant historical conversations are retrieved from the vector store,
- Context snippets are properly included and deduplicated in the prompts sent to the LLM (Ollama),
- The aggregation and formatting of conversation context for LLM input is robust.

---

## Overview

**Purpose:**  
To ensure the `ConversationService` correctly:
- Retrieves and includes relevant context from multiple prior conversations,
- Avoids duplicate entries in context sent to the LLM,
- Properly formats the system context message for the LLM call,
- Passes correct parameters to vector store similarity search.

---

## Key Fixtures

- **`mock_vector_store`**:  
  Returns mock similar conversations for context injection and verifies calls to vector search.
- **`mock_ollama_service`**:  
  Captures calls and system context content sent to the LLM.
- **`mock_entity_extractor`, `mock_summarizer`**:  
  Provide deterministic NLP outputs for conversation analysis and summaries.

---

## Main Test Cases

- `test_context_inclusion_from_multiple_conversations`:
    - Asserts that system context sent to Ollama includes snippets from *all* relevant conversations (not just the latest).
    - Checks that vector store similarity search is called with the right parameters.
- `test_vector_store_similarity_search`:
    - Verifies the vector store mock is called with the correct query and returns multiple high-similarity conversations.
- `test_context_deduplication`:
    - Ensures no duplicate context entries are included in the system prompt.
- `test_context_formatting_in_ollama_request`:
    - Confirms that the context header and body are properly formatted in the message sent to Ollama.

---

## Sample Usage Pattern

These tests allow you to:
- Refactor context aggregation logic safely,
- Confirm the robustness of semantic retrieval and prompt-building,
- Ensure integration between ConversationService and VectorStore remains stable.

---
