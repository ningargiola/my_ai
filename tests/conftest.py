import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock

# Ensure app is in the sys.path before importing anything
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch ChromaDB Client everywhere BEFORE any import that might cause it to load
@pytest.fixture(autouse=True, scope="session")
def patch_chromadb_client():
    with patch("chromadb.Client", autospec=True) as mock_client, \
         patch("chromadb.PersistentClient", autospec=True) as mock_persistent_client:
        # Setup the mock so any collection calls are safely mocked
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value.get_or_create_collection.return_value = mock_collection
        yield

# Now patch all service layers before each test
@pytest.fixture(autouse=True)
def mock_all_services():
    with patch('app.services.ollama_service.OllamaService') as mock_ollama, \
         patch('app.services.entity_extractor.EntityExtractor') as mock_entity, \
         patch('app.services.summarizer.Summarizer') as mock_summarizer:

        mock_ollama_instance = mock_ollama.return_value
        mock_ollama_instance.generate_response.return_value = "Mocked AI response"

        mock_entity_instance = mock_entity.return_value
        mock_entity_instance.analyze_message.return_value = {
            "entities": {"PERSON": ["Test"]},
            "intent": {"greeting": 0.9},
            "raw_entities": []
        }

        mock_summarizer_instance = mock_summarizer.return_value
        mock_summarizer_instance.summarize_conversation.return_value = "Mocked conversation summary"
        mock_summarizer_instance.generate_key_points.return_value = [
            "Mocked key point 1", "Mocked key point 2"
        ]

        yield {
            "ollama": mock_ollama_instance,
            "entity_extractor": mock_entity_instance,
            "summarizer": mock_summarizer_instance,
        }

# Mock the vector store (does not require actual DB)
@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.get_all_conversations.return_value = []
    store.get_similar_conversations.return_value = []
    store.add_conversation.return_value = None
    store.collection = Mock()
    store.collection.get.return_value = {'ids': [], 'metadatas': []}
    return store
