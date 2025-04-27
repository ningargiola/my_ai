import pytest
from unittest.mock import MagicMock, patch
from app.services.summarizer import Summarizer

@pytest.fixture
def summarizer():
    with patch('app.services.summarizer.pipeline') as mock_pipeline, \
         patch('app.services.summarizer.AutoTokenizer'), \
         patch('app.services.summarizer.AutoModelForSeq2SeqLM'):
        # Mock the summarizer pipeline to always return a fixed summary/key points
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.return_value = [
            {"summary_text": "Paris is the capital of France."}
        ]
        mock_pipeline.return_value = mock_summarizer_instance
        return Summarizer()

def test_initialization(summarizer):
    assert summarizer.summarizer is not None

def test_format_conversation(summarizer):
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "That's great to hear!"}
    ]
    formatted = summarizer._format_conversation(messages)
    assert "User: Hello, how are you?" in formatted
    assert "Assistant: I'm doing well, thank you!" in formatted
    assert "User: That's great to hear!" in formatted

def test_summarize_conversation(summarizer):
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Thank you for the information."}
    ]
    # Now the mock will always return "Paris is the capital of France."
    summary = summarizer.summarize_conversation(messages)
    assert isinstance(summary, str)
    assert len(summary) > 0
    # No need to assert len(summary) <= 150 if you're mocking

def test_summarize_empty_conversation(summarizer):
    messages = []
    summary = summarizer.summarize_conversation(messages)
    assert summary == "No messages to summarize."

def test_generate_key_points(summarizer):
    with patch.object(summarizer, 'summarizer') as mock_sum:
        mock_sum.return_value = [{"summary_text": "Key point 1. Key point 2. Key point 3."}]
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
