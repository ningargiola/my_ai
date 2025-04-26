import pytest
from app.services.entity_extractor import EntityExtractor

@pytest.fixture
def extractor():
    # Force CPU for testing
    return EntityExtractor(device="cpu")

def test_initialization(extractor):
    assert extractor.ner_pipeline is not None
    assert extractor.intent_classifier is not None
    assert extractor.nlp is not None

def test_extract_entities(extractor):
    text = "John works at Microsoft in Seattle and uses Python."
    entities = extractor.extract_entities(text)
    
    # Check that we got some entities
    assert len(entities) > 0
    
    # Convert to a more easily testable format
    entity_texts = {ent["word"].lower() for ent in entities}
    
    # Check for expected entities
    assert "john" in entity_texts
    assert "microsoft" in entity_texts
    assert "seattle" in entity_texts
    assert "python" in entity_texts

def test_classify_intent_question(extractor):
    text = "What is the weather like today?"
    intent = extractor.classify_intent(text)
    
    # Check that we got some intents
    assert len(intent) > 0
    assert "question" in intent
    assert intent["question"] > 0.5  # High confidence for question

def test_classify_intent_command(extractor):
    text = "Please send the report by email."
    intent = extractor.classify_intent(text)
    
    assert len(intent) > 0
    assert "request" in intent
    assert "command" in intent

def test_analyze_message(extractor):
    text = "Can you help me find John's email from Microsoft?"
    analysis = extractor.analyze_message(text)
    
    # Check structure
    assert "entities" in analysis
    assert "intent" in analysis
    assert "raw_entities" in analysis
    
    # Check entities
    entities = analysis["entities"]
    assert len(entities) > 0
    
    # Check intent
    intent = analysis["intent"]
    assert len(intent) > 0
    assert "question" in intent

def test_empty_input(extractor):
    # Test with empty string
    entities = extractor.extract_entities("")
    assert len(entities) == 0
    
    intent = extractor.classify_intent("")
    assert isinstance(intent, dict)
    
    analysis = extractor.analyze_message("")
    assert analysis["entities"] == {}
    assert analysis["intent"] == {}
    assert analysis["raw_entities"] == [] 