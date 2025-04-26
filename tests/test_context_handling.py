import pytest
from unittest.mock import Mock, patch
from app.services.conversation_service import ConversationService
from app.services.vector_store import VectorStore
from app.services.ollama_service import OllamaService
from app.models.conversation import Conversation, Message
import logging

@pytest.fixture
def mock_vector_store():
    store = Mock(spec=VectorStore)
    # Mock multiple similar conversations with different contexts
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
    return store

@pytest.fixture
def mock_ollama_service():
    with patch('app.services.ollama_service.requests.post') as mock_post:
        service = OllamaService(model_name="llama3")
        mock_post.return_value.json.return_value = {
            "message": {"content": "Test response using context"}
        }
        mock_post.return_value.raise_for_status = lambda: None
        yield service

def test_context_inclusion_from_multiple_conversations(mock_vector_store, mock_ollama_service, caplog):
    caplog.set_level(logging.INFO)
    
    # Create conversation service
    conv_service = ConversationService(
        vector_store=mock_vector_store,
        ollama_service=mock_ollama_service
    )
    
    # Start a new conversation
    conv_id = conv_service.start_new_conversation().id
    
    # Add a message that should trigger context retrieval
    message = "Tell me about yourself and your favorite color"
    conv_service.add_message(conv_id, message)
    
    # Verify that similar conversations were retrieved
    mock_vector_store.get_similar_conversations.assert_called_once_with(message)
    
    # Check logs for context inclusion from both conversations
    log_messages = [record.message for record in caplog.records]
    context_logs = [msg for msg in log_messages if "Including context in response generation" in msg]
    
    # Verify both contexts are included
    assert any("I like blue because it's calming" in msg for msg in context_logs), "First conversation context not included"
    assert any("I am ALIAS, your AI assistant" in msg for msg in context_logs), "Second conversation context not included"

def test_vector_store_similarity_search(mock_vector_store):
    # Create test conversations
    conv1 = Conversation(id="test_conv_1")
    conv1.add_message(Message(role="user", content="What's your favorite color?"))
    conv1.add_message(Message(role="assistant", content="I like blue because it's calming."))
    
    conv2 = Conversation(id="test_conv_2")
    conv2.add_message(Message(role="user", content="Tell me about yourself"))
    conv2.add_message(Message(role="assistant", content="I am ALIAS, your AI assistant."))
    
    # Add conversations to vector store
    mock_vector_store.add_conversation(conv1)
    mock_vector_store.add_conversation(conv2)
    
    # Search for similar conversations
    similar = mock_vector_store.get_similar_conversations("Tell me about colors and yourself")
    
    # Verify search was performed
    mock_vector_store.get_similar_conversations.assert_called_with("Tell me about colors and yourself")
    
    # Check that we got multiple results
    assert len(similar) > 1, "Should retrieve multiple relevant conversations"
    assert all(conv["similarity"] > 0.7 for conv in similar), "All conversations should have high similarity"

def test_context_deduplication(mock_vector_store, mock_ollama_service):
    # Create conversation service
    conv_service = ConversationService(
        vector_store=mock_vector_store,
        ollama_service=mock_ollama_service
    )
    
    # Start a new conversation
    conv_id = conv_service.start_new_conversation().id
    
    # Add a message
    message = "Tell me about yourself"
    conv_service.add_message(conv_id, message)
    
    # Get the calls to the Ollama service
    mock_calls = mock_ollama_service._last_messages if hasattr(mock_ollama_service, '_last_messages') else []
    
    # Get all system messages with context
    context_messages = [
        msg["content"] for msg in mock_calls 
        if msg["role"] == "system" and "Context from previous" in msg["content"]
    ]
    
    # Verify no duplicate contexts
    if context_messages:
        context_parts = context_messages[0].split("Previous relevant conversation:")
        unique_parts = set(part.strip() for part in context_parts if part.strip())
        assert len(unique_parts) == len(context_parts) - 1, "Found duplicate context entries"

def test_context_formatting_in_ollama_request(mock_vector_store, mock_ollama_service):
    # Create conversation service
    conv_service = ConversationService(
        vector_store=mock_vector_store,
        ollama_service=mock_ollama_service
    )
    
    # Start a new conversation
    conv_id = conv_service.start_new_conversation().id
    
    # Add a message
    message = "Tell me about colors"
    conv_service.add_message(conv_id, message)
    
    # Get the calls to the Ollama service
    mock_calls = mock_ollama_service._last_messages if hasattr(mock_ollama_service, '_last_messages') else []
    
    # Verify that context was included in the messages
    assert any(
        msg for msg in mock_calls 
        if msg["role"] == "system" and "Context from previous" in msg["content"]
    ), "Context was not properly included in Ollama request" 