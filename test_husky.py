from app.services.vector_store import VectorStore
import uuid

def test_query(store, query, min_similarity=0.4):
    print(f"\nSearching for: {query}")
    similar = store.get_similar_conversations(query, limit=5, min_similarity=min_similarity)
    print(f"Found {len(similar)} similar conversations:")
    for conv in similar:
        print(f"\nConversation {conv['id']} (similarity: {conv['similarity']:.2f})")
        for chunk in conv['chunks']:
            for msg in chunk['messages']:
                print(f"{msg['role'].capitalize()}: {msg['content']}")

def main():
    # Initialize vector store
    store = VectorStore()
    
    # Create a test conversation about huskies
    conversation_id = str(uuid.uuid4())
    conversation = {
        'id': conversation_id,
        'messages': [
            {
                'role': 'user',
                'content': 'I want to get a husky',
                'conversation_id': conversation_id
            },
            {
                'role': 'assistant',
                'content': 'Huskies are beautiful dogs! They are known for their thick double coat, blue eyes, and high energy levels. They require lots of exercise and mental stimulation. Make sure you have enough space and time to care for one.',
                'conversation_id': conversation_id
            }
        ]
    }
    
    # Add the conversation
    print("Adding conversation about huskies...")
    store.add_conversation(conversation)
    
    # Test different queries
    test_query(store, "What kind of dog do you want?")
    test_query(store, "Tell me about huskies")
    test_query(store, "I'm interested in getting a dog")
    test_query(store, "What are the characteristics of a husky?")

if __name__ == "__main__":
    main()