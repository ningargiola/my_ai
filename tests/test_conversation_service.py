import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.conversation_service import ConversationService
from app.services.vector_store import VectorStore
from app.services.ollama_service import OllamaService
from app.services.entity_extractor import EntityExtractor
from app.services.summarizer import Summarizer
from app.models.conversation import Conversation, Message
import uuid

@pytest.fixture(autouse=True)
def mock_all_services():
    with patch('chromadb.Client') as mock_chroma_client, \
         patch('ollama.chat') as mock_ollama_chat, \
         patch('app.services.conversation_service.OllamaService') as mock_ollama, \
         patch('app.services.conversation_service.EntityExtractor') as mock_entity_extractor, \
         patch('app.services.conversation_service.Summarizer') as mock_summarizer:

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['mock_id_1', 'mock_id_2'],
            'metadatas': [
                {'conversation_id': 'mock_conv_1'},
                {'conversation_id': 'mock_conv_2'}
            ]
        }
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Ollama chat
        mock_ollama_chat.return_value = Mock(
            message=Mock(content="Mocked AI response")
        )

        # Mock services
        mock_ollama_instance = Mock(
            model_name="phi",
            generate_response=Mock(return_value="Mocked AI response")
        )
        mock_ollama.return_value = mock_ollama_instance

        mock_entity_extractor_instance = Mock(
            analyze_message=Mock(return_value={
                'entities': {'PERSON': ['Test']},
                'intent': {'greeting': 0.9}
            })
        )
        mock_entity_extractor.return_value = mock_entity_extractor_instance

        mock_summarizer_instance = Mock(
            summarize_conversation=Mock(return_value="Mocked conversation summary")
        )
        mock_summarizer.return_value = mock_summarizer_instance

        yield {
            'chroma': mock_chroma_client,
            'ollama_chat': mock_ollama_chat,
            'ollama': mock_ollama_instance,
            'entity_extractor': mock_entity_extractor_instance,
            'summarizer': mock_summarizer_instance
        }

@pytest.fixture
def mock_vector_store():
    store = Mock(spec=VectorStore)
    store.get_similar_conversations.return_value = []
    store.add_conversation.return_value = None
    store.get_all_conversations.return_value = []
    store.collection = Mock()
    store.collection.get.return_value = {
        'ids': ['mock_id_1', 'mock_id_2'],
        'metadatas': [
            {'conversation_id': 'mock_conv_1'},
            {'conversation_id': 'mock_conv_2'}
        ]
    }
    return store

@pytest.fixture
def conversation_service(mock_vector_store, mock_all_services):
    # Patch Summarizer, OllamaService, EntityExtractor for ConversationService init
    with patch('app.services.conversation_service.OllamaService', return_value=mock_all_services['ollama']), \
         patch('app.services.conversation_service.EntityExtractor', return_value=mock_all_services['entity_extractor']), \
         patch('app.services.conversation_service.Summarizer', return_value=mock_all_services['summarizer']):
        return ConversationService(vector_store=mock_vector_store)

def test_new_conversation(conversation_service):
    conversation = conversation_service.start_new_conversation()
    assert conversation is not None
    assert isinstance(conversation, Conversation)
    assert conversation.id is not None
    assert len(conversation.messages) == 0

def test_add_message(conversation_service):
    conversation = conversation_service.start_new_conversation()
    conversation = conversation_service.add_message(conversation.id, "Hello, how are you?")
    assert len(conversation.messages) == 2  # User message and AI response
    assert conversation.messages[0].content == "Hello, how are you?"
    assert conversation.messages[0].role == "user"
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == "Mocked AI response"

def test_get_similar_conversations(conversation_service, mock_vector_store):
    mock_vector_store.get_similar_conversations.return_value = [
        {
            "id": "test_conv_1",
            "chunks": [{
                "messages": [
                    {"role": "user", "content": "What's your favorite color?"},
                    {"role": "assistant", "content": "I like blue."}
                ]
            }],
            "similarity": 0.95
        }
    ]
    conversation = conversation_service.start_new_conversation()
    conversation = conversation_service.add_message(conversation.id, "What's your favorite color?")
    mock_vector_store.get_similar_conversations.assert_called_once()
    assert len(conversation.messages) == 2

def test_conversation_persistence(conversation_service, mock_vector_store):
    """Test that conversations are persisted in the vector store"""
    conversation = conversation_service.start_new_conversation()
    conversation = conversation_service.add_message(conversation.id, "Test message")
    
    # Verify the vector store was called
    mock_vector_store.add_conversation.assert_called_once()
    args = mock_vector_store.add_conversation.call_args[0]
    assert isinstance(args[0], Conversation)  # Should pass the conversation object
    assert args[0].id == conversation.id

def test_message_analysis(conversation_service, mock_all_services):
    conversation = conversation_service.start_new_conversation()
    conversation = conversation_service.add_message(conversation.id, "Hello, my name is Nick")
    user_message = conversation.messages[0]
    assert user_message.analysis is not None
    assert "entities" in user_message.analysis
    assert "intent" in user_message.analysis
    assert user_message.analysis["entities"] == {"PERSON": ["Test"]}
    assert user_message.analysis["intent"] == {"greeting": 0.9}
    mock_all_services['entity_extractor'].analyze_message.assert_called_once()

def test_conversation_summary(conversation_service, mock_all_services):
    conversation = conversation_service.start_new_conversation()
    conversation = conversation_service.add_message(conversation.id, "Hello")
    conversation = conversation_service.add_message(conversation.id, "How are you?")
    assert conversation.summary is not None
    assert conversation.summary == "Mocked conversation summary"
    mock_all_services['summarizer'].summarize_conversation.assert_called()

def test_error_handling(conversation_service):
    with pytest.raises(Exception):
        conversation_service.add_message("non-existent-id", "Hello")
    conversation = conversation_service.start_new_conversation()
    with pytest.raises(ValueError):
        conversation_service.add_message(conversation.id, "")

def test_conversation_history(conversation_service):
    conversation = conversation_service.start_new_conversation()
    messages = [
        "Hello, how are you?",
        "I have a question about Python",
        "Can you help me with async/await?"
    ]
    for msg in messages:
        conversation = conversation_service.add_message(conversation.id, msg)
    assert len(conversation.messages) == len(messages) * 2  # Each user message has an AI response
    for i, msg in enumerate(messages):
        msg_index = i * 2
        assert conversation.messages[msg_index].content == msg
        assert conversation.messages[msg_index].role == "user"
        assert conversation.messages[msg_index + 1].role == "assistant"
        assert conversation.messages[msg_index + 1].content == "Mocked AI response"

def test_message_chunking(conversation_service, mock_vector_store):
    """Test proper chunking of conversations."""
    conversation = conversation_service.start_new_conversation()
    long_message = "This is a very long message that should be chunked. " * 100
    conversation = conversation_service.add_message(conversation.id, long_message)
    
    # Verify the vector store was called
    mock_vector_store.add_conversation.assert_called_once()
    args = mock_vector_store.add_conversation.call_args[0]
    assert isinstance(args[0], Conversation)
    assert len(args[0].messages) > 0

def test_context_retrieval_multiple(conversation_service, mock_vector_store):
    mock_vector_store.get_similar_conversations.return_value = [
        {
            "id": "test_conv_1",
            "chunks": [{
                "messages": [
                    {"role": "user", "content": "What's your favorite color?"},
                    {"role": "assistant", "content": "I like blue."}
                ]
            }],
            "similarity": 0.95
        },
        {
            "id": "test_conv_2",
            "chunks": [{
                "messages": [
                    {"role": "user", "content": "Tell me about yourself"},
                    {"role": "assistant", "content": "I am ALIAS."}
                ]
            }],
            "similarity": 0.85
        }
    ]
    conversation = conversation_service.start_new_conversation()
    conversation = conversation_service.add_message(conversation.id, "Tell me about yourself and your favorite color")
    mock_vector_store.get_similar_conversations.assert_called_once()
    assert len(conversation.messages) == 2

def test_conversation_metadata(conversation_service):
    conversation = conversation_service.start_new_conversation()
    assert hasattr(conversation, 'id')
    assert hasattr(conversation, 'messages')
    assert hasattr(conversation, 'summary')
    assert hasattr(conversation, 'created_at')
    assert hasattr(conversation, 'updated_at')
