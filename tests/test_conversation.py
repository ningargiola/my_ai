import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import Base, engine, get_db
from sqlalchemy.orm import Session

client = TestClient(app)

def test_conversation_context():
    print("\n=== Testing Basic Context Retention ===")
    
    # Start a new conversation
    print("Sending initial message...")
    response = client.post(
        "/api/v1/chat",
        json={"message": "My name is Nick. Remember this for later."}
    )
    assert response.status_code == 200, f"Failed to get response: {response.text}"
    conversation_id = response.json()["conversation_id"]
    print(f"Initial response: {response.json()['response']}")
    
    # Ask about the remembered information
    print("\nAsking about remembered information...")
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "What is my name?",
            "conversation_id": conversation_id
        }
    )
    assert response.status_code == 200, f"Failed to get response: {response.text}"
    response_text = response.json()["response"].lower()
    print(f"AI Response: {response.json()['response']}")
    assert "nick" in response_text, f"AI failed to remember the name. Response was: {response_text}"

def test_multiple_messages_context():
    print("\n=== Testing Multiple Information Retention ===")
    
    # Start a new conversation
    print("Sending information about pets...")
    response = client.post(
        "/api/v1/chat",
        json={"message": "I have a cat named Whiskers and a dog named Rover."}
    )
    assert response.status_code == 200, f"Failed to get response: {response.text}"
    conversation_id = response.json()["conversation_id"]
    print(f"Initial response: {response.json()['response']}")
    
    # Ask about the pets
    print("\nAsking about pet names...")
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "What are the names of my pets?",
            "conversation_id": conversation_id
        }
    )
    assert response.status_code == 200, f"Failed to get response: {response.text}"
    response_text = response.json()["response"].lower()
    print(f"AI Response: {response.json()['response']}")
    
    # Check for both pet names
    whiskers_found = "whiskers" in response_text
    rover_found = "rover" in response_text
    
    if not whiskers_found:
        print("Warning: AI did not mention 'Whiskers'")
    if not rover_found:
        print("Warning: AI did not mention 'Rover'")
    
    assert whiskers_found and rover_found, f"AI failed to remember both pet names. Response was: {response_text}"

def test_complex_context():
    print("\n=== Testing Complex Context Across Multiple Messages ===")
    
    # Start a new conversation with multiple pieces of information
    messages = [
        "I live in New York City.",
        "I work as a software engineer.",
        "My favorite food is pizza."
    ]
    
    conversation_id = None
    for i, message in enumerate(messages, 1):
        print(f"\nSending message {i}: {message}")
        response = client.post(
            "/api/v1/chat",
            json={
                "message": message,
                "conversation_id": conversation_id
            }
        )
        assert response.status_code == 200, f"Failed to get response: {response.text}"
        conversation_id = response.json()["conversation_id"]
        print(f"Response {i}: {response.json()['response']}")
    
    # Ask a question that requires combining multiple pieces of context
    print("\nAsking for a summary of known information...")
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Tell me about myself based on what you know.",
            "conversation_id": conversation_id
        }
    )
    assert response.status_code == 200, f"Failed to get response: {response.text}"
    response_text = response.json()["response"].lower()
    print(f"Final AI Response: {response.json()['response']}")
    
    # Check if the AI remembered multiple pieces of information
    context_checks = {
        "new york": "new york" in response_text,
        "software": "software" in response_text,
        "pizza": "pizza" in response_text
    }
    
    # Print which pieces of information were remembered
    for key, found in context_checks.items():
        status = "✓" if found else "✗"
        print(f"{status} Mentioned {key}")
    
    remembered_count = sum(context_checks.values())
    assert remembered_count >= 2, f"AI only remembered {remembered_count} pieces of information. Response was: {response_text}" 