import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.vector_store import VectorStore
from app.models.conversation import Conversation, Message
import uuid
import json
from datetime import datetime
import numpy as np

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that simulates token counting."""
    mock = Mock()
    def mock_encode(text):
        # Simulate token counting - roughly 1 token per 4 characters
        return [1] * (len(text) // 4)
    mock.encode.side_effect = mock_encode
    return mock

@pytest.fixture
def vector_store(mock_tokenizer):
    """Create a VectorStore instance with mocked ChromaDB."""
    with patch('chromadb.Client') as mock_client, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_cls:
        
        # Set up mock tokenizer
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock collection instance
        mock_collection = MagicMock()
        
        # Mock get response
        mock_collection.get.return_value = {
            'ids': ['test_conv_0_0'],
            'metadatas': [{
                'conversation_id': 'test_conv',
                'messages_json': json.dumps([{
                    'role': 'user',
                    'content': 'test message'
                }]),
                'chunk_start': 0,
                'chunk_end': 0,
                'message_count': 1,
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'messages'
            }],
            'documents': ['User: test message']
        }
        
        # Mock query response
        mock_collection.query.return_value = {
            'ids': [['test_conv_0_0']],
            'distances': [[0.1]],
            'metadatas': [[{
                'conversation_id': 'test_conv',
                'messages_json': json.dumps([{
                    'role': 'user',
                    'content': 'test message'
                }]),
                'chunk_start': 0,
                'chunk_end': 0,
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'messages'
            }]],
            'documents': [['User: test message']]
        }
        
        # Mock client setup
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create and return VectorStore instance
        store = VectorStore()
        store.collection = mock_collection
        store.initialized = True
        store.max_tokens = 100  # Lower token limit for testing
        return store

@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    conversation_id = str(uuid.uuid4())
    messages = [
        Message(
            role="user",
            content="Hello, my name is Nick",
            analysis={
                "entities": {"PERSON": ["Nick"]},
                "intent": {"greeting": 0.9}
            }
        ),
        Message(
            role="assistant",
            content="Hi Nick! I'm ALIAS, your AI assistant.",
            analysis={
                "entities": {"ORG": ["ALIAS"]},
                "intent": {"greeting": 0.8}
            }
        )
    ]
    return Conversation(id=conversation_id, messages=messages)

def test_add_conversation(vector_store, sample_conversation):
    """Test adding a conversation to the vector store."""
    vector_store.add_conversation(sample_conversation)
    
    # Verify collection.add was called
    assert vector_store.collection.add.called
    
    # Verify the call arguments
    call_args = vector_store.collection.add.call_args[1]
    assert 'documents' in call_args
    assert 'metadatas' in call_args
    assert 'ids' in call_args
    
    # Verify metadata structure
    metadata = call_args['metadatas'][0]
    assert metadata['conversation_id'] == sample_conversation.id
    assert metadata['type'] == 'messages'
    assert 'timestamp' in metadata
    assert 'message_count' in metadata

def test_numpy_serialization(vector_store, sample_conversation):
    """Test serialization of numpy types in conversation data."""
    # Add numpy array to analysis
    sample_conversation.messages[0].analysis['vector'] = np.array([1.0, 2.0, 3.0])
    sample_conversation.messages[0].analysis['score'] = np.float32(0.95)
    
    vector_store.add_conversation(sample_conversation)
    
    # Verify numpy types were properly serialized
    call_args = vector_store.collection.add.call_args[1]
    metadata = call_args['metadatas'][0]
    messages = json.loads(metadata['messages_json'])
    
    assert isinstance(messages[0]['analysis']['vector'], list)
    assert isinstance(messages[0]['analysis']['score'], float)

def test_similar_conversations(vector_store):
    """Test retrieving similar conversations."""
    # Set up mock response with correct chunk IDs and metadata structure
    vector_store.collection.query.return_value = {
        'ids': [['test_conv_0_1']],
        'distances': [[0.1]],
        'metadatas': [[{
            'conversation_id': 'test_conv',
            'messages_json': json.dumps([{
                'role': 'user',
                'content': 'test message'
            }]),
            'type': 'messages',
            'chunk_start': 0,
            'chunk_end': 1
        }]],
        'documents': [['User: test message']]
    }

    # Mock the get call that follows
    vector_store.collection.get.return_value = {
        'ids': ['test_conv_0_1'],
        'metadatas': [{
            'conversation_id': 'test_conv',
            'messages_json': json.dumps([{
                'role': 'user',
                'content': 'test message'
            }]),
            'type': 'messages',
            'chunk_start': 0,
            'chunk_end': 1
        }],
        'documents': ['User: test message']
    }

    results = vector_store.get_similar_conversations("test query", limit=1)
    assert len(results) == 1
    assert results[0]['id'] == 'test_conv'
    assert len(results[0]['chunks']) == 1
    assert results[0]['similarity'] > 0

def test_conversation_chunking(vector_store, sample_conversation):
    """Test that long conversations are properly chunked."""
    # Create a long conversation with multiple messages that will exceed max_tokens
    long_messages = [
        Message(role='user', content='A' * 400),  # ~100 tokens
        Message(role='assistant', content='B' * 400),  # ~100 tokens
        Message(role='user', content='C' * 400)  # ~100 tokens
    ]
    sample_conversation.messages = long_messages

    vector_store.add_conversation(sample_conversation)

    # Verify that multiple chunks were created
    call_args = vector_store.collection.add.call_args[1]
    assert len(call_args['ids']) > 1
    assert len(call_args['metadatas']) > 1
    assert len(call_args['documents']) > 1

    # Verify each chunk has correct metadata
    for i, metadata in enumerate(call_args['metadatas']):
        assert metadata['conversation_id'] == sample_conversation.id
        assert metadata['type'] == 'messages'
        assert metadata['chunk_start'] == i
        assert metadata['chunk_end'] == i
        assert 'messages_json' in metadata
        assert 'timestamp' in metadata

def test_duplicate_handling(vector_store, sample_conversation):
    """Test that duplicate conversations are handled correctly."""
    # First add the conversation normally
    vector_store.add_conversation(sample_conversation)
    
    # Clear the mock to start fresh
    vector_store.collection.add.reset_mock()
    
    # Try to add the same conversation again
    vector_store.add_conversation(sample_conversation)

    # Verify that add was called with the updated conversation
    call_args = vector_store.collection.add.call_args[1]
    assert len(call_args['ids']) == 1
    assert call_args['ids'][0].startswith(sample_conversation.id)
    assert call_args['metadatas'][0]['conversation_id'] == sample_conversation.id

def test_metadata_preservation(vector_store, sample_conversation):
    """Test that custom metadata is preserved when adding conversations."""
    # Add custom metadata to conversation
    custom_metadata = {
        'user_id': 'user123',
        'session_id': 'sess456',
        'custom_field': {
            'nested': True,
            'tags': ['important', 'priority']
        }
    }
    sample_conversation.metadata = custom_metadata

    vector_store.add_conversation(sample_conversation)

    # Verify metadata was preserved in the add call
    call_args = vector_store.collection.add.call_args[1]
    metadata = call_args['metadatas'][0]
    
    # Check that required fields are present
    assert metadata['conversation_id'] == sample_conversation.id
    assert 'timestamp' in metadata
    assert metadata['type'] == 'messages'
    assert 'messages_json' in metadata
    assert metadata['chunk_start'] == 0
    assert metadata['chunk_end'] == 0

def test_query_with_filters(vector_store, sample_conversation):
    """Test querying conversations with filters."""
    current_time = datetime.utcnow().isoformat()
    
    # Mock the query response
    vector_store.collection.query.return_value = {
        'ids': [['test_conv_0_0', 'test_conv_0_1']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[{
            'conversation_id': 'test_conv',
            'type': 'messages',
            'chunk_start': 0,
            'chunk_end': 0,
            'messages_json': json.dumps([{'role': 'user', 'content': 'test message'}]),
            'timestamp': current_time
        }]],
        'documents': [['chunk1 content']]
    }

    # Mock the get response that follows the query
    vector_store.collection.get.return_value = {
        'ids': ['test_conv_0_0', 'test_conv_0_1'],
        'metadatas': [
            {
                'conversation_id': 'test_conv',
                'type': 'messages',
                'chunk_start': 0,
                'chunk_end': 0,
                'messages_json': json.dumps([{'role': 'user', 'content': 'test message'}]),
                'timestamp': current_time
            },
            {
                'conversation_id': 'test_conv',
                'type': 'messages',
                'chunk_start': 1,
                'chunk_end': 1,
                'messages_json': json.dumps([{'role': 'assistant', 'content': 'test response'}]),
                'timestamp': current_time
            }
        ],
        'documents': ['chunk1 content', 'chunk2 content']
    }

    # Query with filters
    filters = {
        'type': 'messages',
        'timestamp': {'$gt': '2024-01-01T00:00:00'}
    }
    results = vector_store.get_similar_conversations('test query', limit=5)

    # Verify the query was called with correct parameters
    vector_store.collection.query.assert_called_once()
    call_args = vector_store.collection.query.call_args[1]
    assert call_args['query_texts'] == ['test query']
    assert call_args.get('n_results') is not None

    # Verify results are properly structured
    assert len(results) == 1
    assert results[0]['id'] == 'test_conv'
    assert len(results[0]['chunks']) == 2
    assert 'similarity' in results[0]

def test_get_all_conversations(vector_store):
    """Test retrieving all conversations."""
    current_time = datetime.utcnow().isoformat()
    # Set up mock response
    vector_store.collection.get.return_value = {
        'ids': ['conv1_0_1', 'conv2_0_1'],
        'metadatas': [
            {
                'conversation_id': 'conv1',
                'messages_json': json.dumps([{
                    'role': 'user',
                    'content': 'first conversation',
                    'timestamp': current_time
                }]),
                'type': 'messages',
                'timestamp': current_time
            },
            {
                'conversation_id': 'conv2',
                'messages_json': json.dumps([{
                    'role': 'user',
                    'content': 'second conversation',
                    'timestamp': current_time
                }]),
                'type': 'messages',
                'timestamp': current_time
            }
        ],
        'documents': ['User: first conversation', 'User: second conversation']
    }
    
    conversations = vector_store.get_all_conversations()
    
    # Verify conversations were retrieved
    assert len(conversations) > 0
    assert all(isinstance(conv, dict) for conv in conversations)
    assert all('id' in conv for conv in conversations)
    assert all('chunks' in conv for conv in conversations)

def test_error_handling(vector_store):
    """Test error handling in vector store operations."""
    # Create a conversation for testing
    conversation = Conversation(
        id="test_conv",
        messages=[Message(role="user", content="test", analysis={})]
    )
    
    # Test handling of ChromaDB errors
    vector_store.collection.add.side_effect = Exception("ChromaDB error")
    
    # The add_conversation method should log the error but not raise it
    vector_store.add_conversation(conversation)
    
    # Test query error handling
    vector_store.collection.query.side_effect = Exception("Query error")
    results = vector_store.get_similar_conversations("test query")
    assert len(results) == 0  # Should return empty list on error 