# Test File: `test_vector_store.py`

This suite tests the **VectorStore** service for:
- Conversation chunking and serialization
- Storage and retrieval of message data
- Metadata preservation and numpy/edge-case serialization
- Semantic similarity search with filtering

---

## Overview

**Purpose:**
- To ensure conversations and messages are chunked, stored, and retrieved as expected
- To validate custom metadata, duplicate prevention, and numpy serialization
- To test filtering, error handling, and all-conversation retrieval

---

## Key Fixtures

- **`mock_tokenizer`**:  
  Mocks the tokenizer for predictable token counting during tests
- **`vector_store`**:  
  Fully mocked VectorStore instance, including mocked ChromaDB client and tokenization
- **`sample_conversation`**:  
  Provides a sample Conversation object with messages and analysis

---

## Main Test Cases

- `test_add_conversation`: Checks storing conversations and metadata structure
- `test_numpy_serialization`: Ensures numpy data in analysis is serialized to native types
- `test_similar_conversations`: Validates semantic search, retrieval, and structure of similar conversation results
- `test_conversation_chunking`: Ensures long conversations are properly chunked (token budget)
- `test_duplicate_handling`: Checks that duplicate conversations are detected and handled safely
- `test_metadata_preservation`: Verifies custom metadata is stored alongside conversation chunks
- `test_query_with_filters`: Tests vector search with custom filters (timestamp, type)
- `test_get_all_conversations`: Ensures all stored conversations can be retrieved and parsed
- `test_error_handling`: Covers edge-case and error-path handling for all main VectorStore operations

---

## Usage

Run this suite to:
- Refactor VectorStore chunking or storage logic safely
- Ensure robust handling of custom data and numpy types
- Validate all-vectorstore workflows for chat/semantic retrieval features

---
