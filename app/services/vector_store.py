import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import logging
import os
from datetime import datetime
import json
from uuid import uuid4
import time
import numpy as np
import traceback

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = None):
        """Initialize VectorStore with optional persistence"""
        self.initialized = False
        self.client = None
        self.collection = None
        self.max_tokens = 512  # Maximum tokens per chunk
        
        try:
            # Initialize ChromaDB client
            if persist_directory:
                logger.info(f"Initializing persistent VectorStore at {persist_directory}")
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                logger.info("Initializing in-memory VectorStore")
                self.client = chromadb.Client(
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            # Use a simpler embedding function
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="conversations",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.initialized = True
            logger.info("VectorStore initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {str(e)}")
            logger.error(traceback.format_exc())
            self.initialized = False
            raise

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def _chunk_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split messages into semantic chunks based on content"""
        if not self.initialized:
            logger.warning("VectorStore not initialized, returning single chunk")
            return [{"messages": messages, "start_index": 0, "end_index": len(messages) - 1}]
            
        chunks = []
        current_chunk = []
        
        for msg in messages:
            # If adding this message would exceed max_tokens, start a new chunk
            if len(current_chunk) >= 2:  # Simple chunking: 2 messages per chunk
                chunks.append({
                    "messages": current_chunk.copy(),
                    "start_index": len(chunks),
                    "end_index": len(chunks) + len(current_chunk) - 1
                })
                current_chunk = []
            
            current_chunk.append(msg)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "messages": current_chunk.copy(),
                "start_index": len(chunks),
                "end_index": len(chunks) + len(current_chunk) - 1
            })
        
        return chunks

    def _create_chunk_context(self, chunk: Dict[str, Any]) -> str:
        """Create a rich context for a chunk of messages"""
        context_parts = []
        
        # Add message content
        for msg in chunk["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if "analysis" in msg:
                content += f"\nAnalysis: {msg['analysis']}"
            if "summary" in msg:
                content += f"\nSummary: {msg['summary']}"
            context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)

    def add_conversation(self, conversation):
        """Add a conversation to the vector store with enhanced chunk-level storage"""
        if not self.initialized:
            logger.warning("VectorStore not initialized, skipping conversation storage")
            return
            
        try:
            # Convert messages to list of dicts
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "analysis": msg.analysis if msg.analysis else {}
                }
                for msg in conversation.messages
            ]
            
            # Create semantic chunks
            chunks = self._chunk_messages(messages)
            
            # Prepare documents and metadata for each chunk
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Create a unique ID for each chunk
                chunk_id = f"{conversation.id}_{chunk['start_index']}_{chunk['end_index']}"
                
                # Create rich context for the chunk
                context = self._create_chunk_context(chunk)
                
                # Convert chunk messages to JSON string, handling numpy types
                serializable_messages = self._convert_to_serializable(chunk['messages'])
                messages_json = json.dumps(serializable_messages)
                
                # Check if this chunk already exists
                existing = self.collection.get(ids=[chunk_id])
                if existing and existing['ids']:
                    # Update existing chunk
                    self.collection.update(
                        ids=[chunk_id],
                        documents=[context],
                        metadatas=[{
                            "conversation_id": conversation.id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "message_count": len(chunk['messages']),
                            "messages_json": messages_json,
                            "chunk_start": chunk['start_index'],
                            "chunk_end": chunk['end_index']
                        }]
                    )
                    logger.debug(f"Updated existing chunk {chunk_id}")
                else:
                    # Add new chunk
                    documents.append(context)
                    metadatas.append({
                        "conversation_id": conversation.id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message_count": len(chunk['messages']),
                        "messages_json": messages_json,
                        "chunk_start": chunk['start_index'],
                        "chunk_end": chunk['end_index']
                    })
                    ids.append(chunk_id)
            
            # Add all new chunks to vector store
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.debug(f"Added {len(documents)} new chunks to vector store")
            
            logger.debug(f"Processed conversation {conversation.id} with {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error adding conversation to vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_similar_conversations(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get similar conversations from the vector store"""
        if not self.initialized:
            logger.warning("VectorStore not initialized, returning empty list")
            return []
            
        try:
            # Search for similar chunks
            results = self.collection.query(
                query_texts=[query],
                n_results=limit * 2  # Get more results initially for filtering
            )
            
            # Process and deduplicate results
            conversation_map = {}  # Map to store unique conversations
            
            for i, (chunk_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                similarity = 1 - distance  # Convert distance to similarity
                
                # Skip if similarity is too low
                if similarity < min_similarity:
                    continue
                    
                # Get metadata for the chunk
                metadata = self.collection.get(ids=[chunk_id])['metadatas'][0]
                conversation_id = metadata['conversation_id']
                
                # If we haven't seen this conversation or this chunk has higher similarity
                if conversation_id not in conversation_map or similarity > conversation_map[conversation_id]['similarity']:
                    # Get all chunks for this conversation
                    conv_chunks = self.collection.get(
                        where={"conversation_id": conversation_id}
                    )
                    
                    # Sort chunks by their position
                    chunks = []
                    for j, chunk_metadata in enumerate(conv_chunks['metadatas']):
                        chunk_data = {
                            'messages': json.loads(chunk_metadata['messages_json']),
                            'chunk_start': chunk_metadata['chunk_start'],
                            'chunk_end': chunk_metadata['chunk_end']
                        }
                        chunks.append(chunk_data)
                    
                    chunks.sort(key=lambda x: x['chunk_start'])
                    
                    # Store conversation with its best similarity score
                    conversation_map[conversation_id] = {
                        'id': conversation_id,
                        'chunks': chunks,
                        'similarity': similarity
                    }
                
                # Break if we have enough unique conversations
                if len(conversation_map) >= limit:
                    break
            
            # Convert map to sorted list
            conversations = list(conversation_map.values())
            conversations.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.debug(f"Found {len(conversations)} similar conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error getting similar conversations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Retrieve all conversations from the vector store with enhanced metadata"""
        if not self.initialized:
            logger.warning("VectorStore not initialized, returning empty list")
            return []
            
        try:
            results = self.collection.get()
            
            if not results or not results["ids"]:
                logger.debug("No conversations found in vector store")
                return []
            
            # Group chunks by conversation
            conversations = {}
            for id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                conv_id = meta["conversation_id"]
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        "id": conv_id,
                        "chunks": [],
                        "metadata": {
                            "timestamp": meta.get("timestamp"),
                            "message_count": 0
                        }
                    }
                
                # Parse messages from JSON
                messages = json.loads(meta.get("messages_json", "[]"))
                conversations[conv_id]["chunks"].append({
                    "content": doc,
                    "messages": messages,
                    "chunk_start": meta.get("chunk_start"),
                    "chunk_end": meta.get("chunk_end")
                })
                conversations[conv_id]["metadata"]["message_count"] += len(messages)
            
            # Convert to list and sort by timestamp
            conversation_list = list(conversations.values())
            conversation_list.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
            
            return conversation_list
            
        except Exception as e:
            logger.error(f"Error getting all conversations: {str(e)}")
            logger.error(traceback.format_exc())
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