import pytest
from app.services.summarizer import ConversationSummarizer
import torch

@pytest.fixture
def summarizer():
    # Force CPU for testing to avoid GPU memory issues
    device = "cpu"
    return ConversationSummarizer(device=device)

def test_initialization(summarizer):
    assert summarizer.summarizer is not None

def test_format_messages_for_summarization(summarizer):
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "That's great to hear!"}
    ]
    
    formatted = summarizer._format_messages_for_summarization(messages)
    assert "user: Hello, how are you?" in formatted
    assert "assistant: I'm doing well, thank you!" in formatted
    assert "user: That's great to hear!" in formatted

def test_summarize_conversation(summarizer):
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Thank you for the information."}
    ]
    
    summary = summarizer.summarize_conversation(messages)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) <= 150

def test_summarize_empty_conversation(summarizer):
    messages = []
    summary = summarizer.summarize_conversation(messages)
    assert summary == "No messages to summarize"

def test_generate_key_points(summarizer):
    messages = [
        {"role": "user", "content": "What are the main benefits of exercise?"},
        {"role": "assistant", "content": "Exercise has many benefits including improved cardiovascular health, better mental health, and increased energy levels."},
        {"role": "user", "content": "Can you elaborate on the mental health benefits?"},
        {"role": "assistant", "content": "Exercise can reduce stress, anxiety, and depression while improving mood and cognitive function."}
    ]
    
    key_points = summarizer.generate_key_points(messages)
    assert isinstance(key_points, list)
    assert len(key_points) <= 3
    for point in key_points:
        assert isinstance(point, str)
        assert len(point) > 0 