import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "vector_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )

    def add_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]):
        """Add a conversation to the vector store"""
        try:
            # Combine all messages into a single context
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            # Convert messages to JSON string for metadata
            messages_json = json.dumps(messages)
            
            # Add to vector store
            self.collection.add(
                documents=[context],
                metadatas=[{
                    "conversation_id": conversation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_count": len(messages),
                    "messages_json": messages_json  # Store messages as JSON string
                }],
                ids=[conversation_id]
            )
            logger.debug(f"Added conversation {conversation_id} to vector store")
        except Exception as e:
            logger.error(f"Error adding conversation to vector store: {str(e)}")
            raise

    def get_similar_conversations(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar conversations based on the query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "similarity": 1 - dist  # Convert distance to similarity
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error getting similar conversations: {str(e)}")
            return []

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Retrieve all conversations from the vector store"""
        try:
            # Get all items from the collection
            results = self.collection.get()
            
            # If no results, return empty list
            if not results or not results["ids"]:
                logger.debug("No conversations found in vector store")
                return []
            
            # Format the results
            conversations = []
            for id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                # Parse the messages JSON string back into a list
                messages = json.loads(meta.get("messages_json", "[]"))
                conversations.append({
                    "id": id,
                    "messages": messages,
                    "metadata": {
                        "timestamp": meta.get("timestamp"),
                        "message_count": meta.get("message_count", 0)
                    }
                })
            
            logger.debug(f"Retrieved {len(conversations)} conversations from vector store")
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting all conversations: {str(e)}")
            return []

    def store_message(self, message_id: int, conversation_id: int, content: str, role: str):
        """Store a message in the vector database."""
        try:
            logger.debug(f"Storing message {message_id} in vector database")
            logger.debug(f"Content: {content[:100]}...")  # Log first 100 chars
            
            self.collection.add(
                documents=[content],
                metadatas=[{
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "role": role
                }],
                ids=[f"msg_{message_id}"]
            )
            
            logger.debug(f"Successfully stored message {message_id} in vector database")
            # Log the count of messages in the collection
            count = self.collection.count()
            logger.debug(f"Total messages in collection: {count}")
            
        except Exception as e:
            logger.error(f"Error storing message in vector database: {str(e)}")
            logger.error("Full traceback:")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def get_relevant_messages(self, query: str, conversation_id: Optional[int] = None, limit: int = 5) -> List[Dict]:
        """Retrieve relevant messages based on semantic similarity."""
        try:
            logger.debug(f"Searching for messages relevant to: {query[:100]}...")  # Log first 100 chars
            
            # Get total count before querying
            total_count = self.collection.count()
            logger.debug(f"Total messages in collection: {total_count}")
            
            # If there are no messages, return empty list
            if total_count == 0:
                logger.debug("No messages in collection, returning empty list")
                return []
            
            # Prepare where clause if conversation_id is provided
            where = {"conversation_id": conversation_id} if conversation_id else None
            if where:
                logger.debug(f"Filtering by conversation_id: {conversation_id}")
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit, total_count),  # Don't request more than we have
                where=where
            )
            
            # Format results
            messages = []
            if results['ids'] and results['ids'][0]:  # Check if we have any results
                for i in range(len(results['ids'][0])):
                    message = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    }
                    messages.append(message)
                    logger.debug(f"Found relevant message: {message['content'][:100]}...")  # Log first 100 chars
            
            logger.debug(f"Retrieved {len(messages)} relevant messages")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages from vector database: {str(e)}")
            logger.error("Full traceback:")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def delete_message(self, message_id: int):
        """Delete a message from the vector database."""
        try:
            self.collection.delete(ids=[f"msg_{message_id}"])
            logger.debug(f"Deleted message {message_id} from vector database")
        except Exception as e:
            logger.error(f"Error deleting message from vector database: {str(e)}")
            raise

    def delete_conversation(self, conversation_id: int):
        """Delete all messages from a conversation."""
        try:
            self.collection.delete(where={"conversation_id": conversation_id})
            logger.debug(f"Deleted conversation {conversation_id} from vector database")
        except Exception as e:
            logger.error(f"Error deleting conversation from vector database: {str(e)}")
            raise 