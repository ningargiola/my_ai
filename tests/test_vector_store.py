import pytest
from app.services.vector_store import VectorStore
from app.models.conversation import Conversation, Message
import uuid
import numpy as np

@pytest.fixture
def vector_store():
    return VectorStore()

@pytest.fixture
def sample_conversation():
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
    """Test adding a conversation to the vector store"""
    # Add the conversation
    vector_store.add_conversation(sample_conversation)
    
    # Verify the conversation was added
    results = vector_store.collection.get(
        where={"conversation_id": sample_conversation.id}
    )
    assert results is not None
    assert len(results['ids']) > 0

def test_similar_conversations(vector_store, sample_conversation):
    """Test retrieving similar conversations"""
    # Add the conversation
    vector_store.add_conversation(sample_conversation)
    
    # Create a similar conversation
    similar_conversation = Conversation(
        id=str(uuid.uuid4()),
        messages=[
            Message(
                role="user",
                content="Hi, I'm also Nick",
                analysis={
                    "entities": {"PERSON": ["Nick"]},
                    "intent": {"greeting": 0.9}
                }
            )
        ]
    )
    vector_store.add_conversation(similar_conversation)
    
    # Search for similar conversations
    similar = vector_store.get_similar_conversations(
        "Hello, my name is Nick",
        limit=2
    )
    
    assert len(similar) > 0
    assert any(c.id == sample_conversation.id for c in similar)

def test_duplicate_handling(vector_store, sample_conversation):
    """Test handling of duplicate conversations"""
    # Add the conversation twice
    vector_store.add_conversation(sample_conversation)
    vector_store.add_conversation(sample_conversation)
    
    # Verify no duplicate warnings in logs
    results = vector_store.collection.get(
        where={"conversation_id": sample_conversation.id}
    )
    assert len(results['ids']) == len(sample_conversation.messages)

def test_conversation_chunking(vector_store, sample_conversation):
    """Test proper chunking of conversations"""
    # Add a long conversation
    long_messages = [
        Message(
            role="user",
            content="This is a very long message " * 10,
            analysis={"entities": {}, "intent": {}}
        )
    ]
    long_conversation = Conversation(
        id=str(uuid.uuid4()),
        messages=long_messages
    )
    
    vector_store.add_conversation(long_conversation)
    
    # Verify chunks were created
    results = vector_store.collection.get(
        where={"conversation_id": long_conversation.id}
    )
    assert len(results['ids']) > 1  # Should be split into multiple chunks

def test_metadata_preservation(vector_store, sample_conversation):
    """Test preservation of conversation metadata"""
    vector_store.add_conversation(sample_conversation)
    
    results = vector_store.collection.get(
        where={"conversation_id": sample_conversation.id}
    )
    
    # Verify metadata is preserved
    for metadata in results['metadatas']:
        assert 'conversation_id' in metadata
        assert 'chunk_index' in metadata
        assert 'role' in metadata
        assert 'content' in metadata 