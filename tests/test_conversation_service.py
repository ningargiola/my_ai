import pytest
from app.services.conversation_service import ConversationService
from app.services.vector_store import VectorStore
from app.models.conversation import Conversation, Message
import uuid

@pytest.fixture
def vector_store():
    return VectorStore()

@pytest.fixture
def conversation_service(vector_store):
    return ConversationService(vector_store=vector_store)

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

def test_new_conversation(conversation_service):
    """Test starting a new conversation"""
    conversation = conversation_service.start_new_conversation()
    assert conversation is not None
    assert conversation.id is not None
    assert len(conversation.messages) == 0

def test_add_message(conversation_service):
    """Test adding a message to a conversation"""
    conversation = conversation_service.start_new_conversation()
    message = "Hello, my name is Nick"
    
    updated_conversation = conversation_service.add_message(
        conversation.id,
        message
    )
    
    assert len(updated_conversation.messages) == 1
    assert updated_conversation.messages[0].content == message
    assert updated_conversation.messages[0].role == "user"

def test_get_similar_conversations(conversation_service, sample_conversation):
    """Test retrieving similar conversations"""
    # Add the sample conversation to vector store
    conversation_service.vector_store.add_conversation(sample_conversation)
    
    # Create a new conversation with a similar message
    new_conversation = conversation_service.start_new_conversation()
    message = "Hi, I'm also Nick"
    
    updated_conversation = conversation_service.add_message(
        new_conversation.id,
        message
    )
    
    # Verify similar conversations were retrieved
    assert len(updated_conversation.messages) == 1
    assert hasattr(updated_conversation.messages[0], 'similar_conversations')
    assert len(updated_conversation.messages[0].similar_conversations) > 0

def test_conversation_persistence(conversation_service):
    """Test that conversations are persisted in vector store"""
    # Start a new conversation
    conversation = conversation_service.start_new_conversation()
    message = "Hello, my name is Nick"
    
    # Add a message
    updated_conversation = conversation_service.add_message(
        conversation.id,
        message
    )
    
    # Verify conversation was stored
    results = conversation_service.vector_store.collection.get(
        where={"conversation_id": conversation.id}
    )
    assert results is not None
    assert len(results['ids']) > 0

def test_message_analysis(conversation_service):
    """Test that messages are properly analyzed"""
    conversation = conversation_service.start_new_conversation()
    message = "Hello, my name is Nick"
    
    updated_conversation = conversation_service.add_message(
        conversation.id,
        message
    )
    
    assert len(updated_conversation.messages) == 1
    assert 'analysis' in updated_conversation.messages[0].__dict__
    analysis = updated_conversation.messages[0].analysis
    
    assert 'entities' in analysis
    assert 'intent' in analysis
    assert 'PERSON' in analysis['entities']
    assert 'Nick' in analysis['entities']['PERSON']

def test_conversation_summary(conversation_service):
    """Test that conversation summaries are generated"""
    conversation = conversation_service.start_new_conversation()
    message = "Hello, my name is Nick"
    
    updated_conversation = conversation_service.add_message(
        conversation.id,
        message
    )
    
    assert len(updated_conversation.messages) == 1
    assert hasattr(updated_conversation, 'summary')
    assert updated_conversation.summary is not None
    assert len(updated_conversation.summary) > 0 