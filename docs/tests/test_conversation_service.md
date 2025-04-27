# Test File: `test_conversation_service.py`

This suite tests the core behaviors, edge cases, and interactions of the `ConversationService`—the heart of your chat and context pipeline.

---

## Overview

**Purpose:**  
To ensure `ConversationService`:
- Handles conversation and message creation reliably
- Aggregates context and interacts with the vector store correctly
- Analyzes messages, produces summaries, and persists data as expected
- Handles errors and edge cases robustly

---

## Key Fixtures

- **`mock_all_services`**:  
  Autouse fixture that mocks all key backend services (ChromaDB, Ollama, entity extraction, summarizer, etc).
- **`mock_vector_store`**:  
  Simulates a vector database for conversation persistence and retrieval.
- **`conversation_service`**:  
  Main service under test, initialized with mocks.

---

## Main Test Cases

- `test_new_conversation`: Checks conversation creation and initialization.
- `test_add_message`: Validates user message + AI response flow.
- `test_get_similar_conversations`: Ensures semantic search works as expected.
- `test_conversation_persistence`: Verifies vector store persistence after each message.
- `test_message_analysis`: Asserts that entities/intents are extracted and attached to messages.
- `test_conversation_summary`: Confirms conversation summaries are generated and stored.
- `test_error_handling`: Tests handling of bad/empty inputs and not-found cases.
- `test_conversation_history`: Validates multi-message dialogue and history structure.
- `test_message_chunking`: Checks large messages and correct chunking to the store.
- `test_context_retrieval_multiple`: Ensures multiple relevant conversations are used for context.
- `test_conversation_metadata`: Asserts that Conversation objects have required metadata fields.

---

## Usage

Run this test suite to:
- Safely refactor conversation or vector store logic
- Validate that all parts of the system interact correctly
- Ensure new features (summaries, analysis, context) are regression-proof

---