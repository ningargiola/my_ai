from app.services.vector_store import VectorStore
import uuid

def main():
    # Initialize vector store
    store = VectorStore()
    
    # Create a test conversation
    conversation_id = str(uuid.uuid4())
    conversation = {
        'id': conversation_id,
        'messages': [
            {
                'role': 'user',
                'content': 'How do I use Python for data analysis?',
                'conversation_id': conversation_id
            },
            {
                'role': 'assistant',
                'content': 'For data analysis in Python, you can use libraries like pandas, numpy, and matplotlib. Pandas is great for data manipulation and analysis, numpy for numerical computations, and matplotlib for visualization.',
                'conversation_id': conversation_id
            }
        ]
    }
    
    # Add the conversation
    print("Adding conversation...")
    store.add_conversation(conversation)
    
    # Search for similar conversations
    print("\nSearching for similar conversations...")
    query = "What are good Python libraries for data analysis?"
    similar = store.get_similar_conversations(query, limit=5, min_similarity=0.5)  # Lower threshold for testing
    
    print(f"\nFound {len(similar)} similar conversations:")
    for conv in similar:
        print(f"\nConversation {conv['id']} (similarity: {conv['similarity']:.2f})")
        for chunk in conv['chunks']:
            for msg in chunk['messages']:
                print(f"{msg['role'].capitalize()}: {msg['content']}")

if __name__ == "__main__":
    main() 