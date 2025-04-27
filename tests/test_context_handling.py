import pytest
from unittest.mock import Mock, patch
from app.services.conversation_service import ConversationService
from app.services.vector_store import VectorStore
from app.services.ollama_service import OllamaService
from app.services.entity_extractor import EntityExtractor
from app.services.summarizer import Summarizer
from app.models.conversation import Conversation, Message

@pytest.fixture
def mock_vector_store():
    store = Mock(spec=VectorStore)
    store.get_similar_conversations.return_value = [
        {
            "id": "test_conv_1",
            "chunks": [{
                "messages": [
                    {"role": "user", "content": "What's your favorite color?"},
                    {"role": "assistant", "content": "I like blue because it's calming."}
                ],
                "chunk_start": 0,
                "chunk_end": 1
            }],
            "similarity": 0.95
        },
        {
            "id": "test_conv_2",
            "chunks": [{
                "messages": [
                    {"role": "user", "content": "Tell me about yourself"},
                    {"role": "assistant", "content": "I am ALIAS, your AI assistant."}
                ],
                "chunk_start": 0,
                "chunk_end": 1
            }],
            "similarity": 0.85
        }
    ]
    store.collection = Mock()
    store.collection.get.return_value = {'ids': [], 'metadatas': []}
    return store

@pytest.fixture
def mock_ollama_service():
    service = Mock(spec=OllamaService)
    service.generate_response.return_value = "Test response using context"
    return service

@pytest.fixture
def mock_entity_extractor():
    extractor = Mock(spec=EntityExtractor)
    extractor.analyze_message.return_value = {
        "entities": {"PERSON": ["Nick"]},
        "intent": {"greeting": 0.9},
        "raw_entities": []
    }
    return extractor

@pytest.fixture
def mock_summarizer():
    summarizer = Mock(spec=Summarizer)
    summarizer.summarize_conversation.return_value = "Mocked conversation summary"
    return summarizer

@pytest.fixture
def conversation_service(mock_vector_store, mock_ollama_service, mock_entity_extractor, mock_summarizer):
    with patch('app.services.conversation_service.OllamaService', return_value=mock_ollama_service), \
         patch('app.services.conversation_service.EntityExtractor', return_value=mock_entity_extractor), \
         patch('app.services.conversation_service.Summarizer', return_value=mock_summarizer):
        return ConversationService(vector_store=mock_vector_store)

def extract_context_content(mock_ollama_service, marker="Context from previous relevant conversations"):
    """Helper to extract system context message sent to Ollama."""
    args, kwargs = mock_ollama_service.generate_response.call_args
    # Find system context message
    for msg in kwargs["conversation_history"]:
        if msg["role"] == "system" and marker in msg["content"]:
            return msg["content"]
    return ""

def test_context_inclusion_from_multiple_conversations(conversation_service, mock_vector_store, mock_ollama_service):
    message = "Tell me about yourself and your favorite color"
    conv_id = conversation_service.start_new_conversation().id
    conversation_service.add_message(conv_id, message)
    mock_vector_store.get_similar_conversations.assert_called_once_with(message, limit=5, min_similarity=0.7)

    # Check both key context snippets are present in context
    context = extract_context_content(mock_ollama_service)
    assert "I like blue because it's calming" in context, "First conversation context not included"
    assert "I am ALIAS, your AI assistant." in context, "Second conversation context not included"

def test_vector_store_similarity_search(mock_vector_store):
    similar = mock_vector_store.get_similar_conversations("Tell me about colors and yourself")
    mock_vector_store.get_similar_conversations.assert_called_with("Tell me about colors and yourself")
    assert len(similar) > 1, "Should retrieve multiple relevant conversations"
    assert all(conv["similarity"] > 0.7 for conv in similar), "All conversations should have high similarity"

def test_context_deduplication(conversation_service, mock_ollama_service):
    conv_id = conversation_service.start_new_conversation().id
    conversation_service.add_message(conv_id, "Tell me about yourself")
    context = extract_context_content(mock_ollama_service)
    # Split by lines, remove header and deduplicate
    lines = [l for l in context.splitlines() if l.strip() and "Context from previous" not in l]
    assert len(set(lines)) == len(lines), "Found duplicate context entries"

def test_context_formatting_in_ollama_request(conversation_service, mock_ollama_service):
    conv_id = conversation_service.start_new_conversation().id
    conversation_service.add_message(conv_id, "Tell me about colors")
    context = extract_context_content(mock_ollama_service)
    assert "Context from previous" in context, "Context was not properly included in Ollama request"
