# Test File: `test_entity_extractor.py`

This suite tests the **EntityExtractor** service for named entity recognition (NER), intent classification, and holistic message analysis.

---

## Overview

**Purpose:**
- To ensure EntityExtractor correctly identifies entities and classifies user intent
- To verify all expected fields are returned by the analysis
- To check behavior for valid, ambiguous, and empty inputs

---

## Key Fixture

- **`extractor`**: Pytest fixture that initializes EntityExtractor on CPU (for consistency and CI compatibility)

---

## Main Test Cases

- `test_initialization`: Checks that all NLP/model components are loaded.
- `test_extract_entities`: Confirms expected entities are detected from a sample text.
- `test_classify_intent_question`: Asserts 'question' intent is correctly recognized with high confidence.
- `test_classify_intent_command`: Ensures 'request' and 'command' intents are detected.
- `test_analyze_message`: Checks that analysis returns a complete dictionary with entities, intent, and raw_entities fields.
- `test_empty_input`: Verifies graceful handling of empty string input (should return empty structures).

---

## Usage

Run this suite to:
- Safeguard against regressions in NER or intent logic
- Confirm compatibility with real-world and edge-case input
- Validate that message analysis supports the full chat pipeline

---
