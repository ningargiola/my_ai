import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import json
import hashlib
import numpy as np
import traceback
from transformers import AutoTokenizer
import os
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = None):
        """Initialize VectorStore with optional persistence and token-based chunking."""
        self.initialized = False
        self.client = None
        self.collection = None
        self.max_tokens = 256  # Tune as desired
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")  # Change model as needed

        try:
            if persist_directory is None:
                persist_directory = "data/chroma"
                os.makedirs(persist_directory, exist_ok=True)
            logger.info(f"Initializing persistent VectorStore at {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
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
        """Convert numpy/datetime types to native types for JSON serialization."""
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

    def _chunk_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        i = 0
        while i < len(messages) - 1:
            # Pair user + assistant if possible
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                chunks.append({"messages": [messages[i], messages[i+1]]})
                i += 2
            else:
                # Just take one message if not a pair
                chunks.append({"messages": [messages[i]]})
                i += 1
        if i < len(messages):
            chunks.append({"messages": [messages[i]]})
        return chunks

    def _chunk_id(self, conversation_id: str, chunk: Dict[str, Any]) -> str:
        """Create a unique chunk id based on content hash."""
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        # Convert any numpy types in the messages
        converted_messages = convert_numpy(chunk['messages'])
        chunk_json = json.dumps(converted_messages, sort_keys=True)
        chunk_hash = hashlib.sha1(chunk_json.encode('utf-8')).hexdigest()
        return f"{conversation_id}_chunk_{chunk_hash}"

    def _create_chunk_context(self, chunk: Dict[str, Any]) -> str:
        """Build a printable context string for a chunk of messages."""
        context_parts = []
        for msg in chunk["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if "analysis" in msg and msg["analysis"]:
                content += f"\nAnalysis: {msg['analysis']}"
            if "summary" in msg and msg["summary"]:
                content += f"\nSummary: {msg['summary']}"
            context_parts.append(f"{role}: {content}")
        return "\n\n".join(context_parts)

    def add_conversation(self, conversation: Dict[str, Any]) -> None:
        """Add a conversation to the vector store using unique IDs per chunk (SHA1)."""
        try:
            if not conversation or not isinstance(conversation, dict):
                raise ValueError("Invalid conversation format")
            conversation_id = conversation.get('id')
            if not conversation_id:
                raise ValueError("Conversation must have an ID")

            # Delete old conversation, if it exists (safe: prevents duplicate IDs)
            existing = self.collection.get(where={"conversation_id": conversation_id})
            if existing and existing['ids']:
                self.collection.delete(where={"conversation_id": conversation_id})

            # Chunk messages and add with unique hash IDs
            chunks = self._chunk_messages(conversation.get('messages', []))
            documents, metadatas, ids = [], [], []
            for chunk in chunks:
                chunk_id = self._chunk_id(conversation_id, chunk)
                
                # Convert numpy types in messages before JSON serialization
                converted_messages = self._convert_to_serializable(chunk['messages'])
                
                metadata = {
                    "conversation_id": conversation_id,
                    "chunk_start": chunk.get('start_index', 0),
                    "chunk_end": chunk.get('end_index', 0),
                    "message_count": len(conversation.get('messages', [])),
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "messages",
                    "messages_json": json.dumps(converted_messages),
                    "chunk_hash": chunk_id.split('_chunk_')[-1]
                }
                
                # Add any additional metadata from the conversation
                for key, value in conversation.items():
                    if key not in ['id', 'messages']:
                        metadata[key] = self._convert_to_serializable(value)
                
                context = self._create_chunk_context(chunk)
                documents.append(context)
                metadatas.append(metadata)
                ids.append(chunk_id)

            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            # Store summary with a single unique ID per conversation
            if 'summary' in conversation and conversation['summary']:
                summary_id = f"{conversation_id}_summary"
                self.collection.add(
                    documents=[conversation['summary']],
                    metadatas=[{
                        "conversation_id": conversation_id,
                        "type": "summary",
                        "timestamp": datetime.utcnow().isoformat()
                    }],
                    ids=[summary_id]
                )

        except Exception as e:
            logger.error(f"Error adding conversation to vector store: {str(e)}")
            raise

    def get_similar_conversations(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Retrieve similar conversations from the vector store."""
        if not self.initialized:
            logger.warning("VectorStore not initialized, returning empty list")
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit * 2
            )
            conversation_map = {}
            for chunk_id, distance in zip(results['ids'][0], results['distances'][0]):
                similarity = 1 - distance
                if similarity < min_similarity:
                    continue
                    
                # Get metadata and skip if no conversation_id
                metadata = self.collection.get(ids=[chunk_id])['metadatas'][0]
                if 'conversation_id' not in metadata:
                    logger.warning(f"Skipping chunk {chunk_id} without conversation_id")
                    continue
                    
                conversation_id = metadata['conversation_id']
                if conversation_id not in conversation_map or similarity > conversation_map[conversation_id]['similarity']:
                    conv_chunks = self.collection.get(
                        where={
                            "$and": [
                                {"conversation_id": {"$eq": conversation_id}},
                                {"type": {"$eq": "messages"}}
                            ]
                        }
                    )
                    chunks = []
                    for chunk_metadata in conv_chunks['metadatas']:
                        chunk_data = {
                            'messages': json.loads(chunk_metadata['messages_json']),
                            'chunk_start': chunk_metadata['chunk_start'],
                            'chunk_end': chunk_metadata['chunk_end']
                        }
                        chunks.append(chunk_data)
                    chunks.sort(key=lambda x: x['chunk_start'])
                    conversation_map[conversation_id] = {
                        'id': conversation_id,
                        'chunks': chunks,
                        'similarity': similarity
                    }
                if len(conversation_map) >= limit:
                    break
            conversations = list(conversation_map.values())
            conversations.sort(key=lambda x: x['similarity'], reverse=True)
            logger.debug(f"Found {len(conversations)} similar conversations")
            return conversations
        except Exception as e:
            logger.error(f"Error getting similar conversations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def get_conversation_summaries(self, conversation_id: str) -> List[str]:
        """Retrieve all summaries for a given conversation."""
        if not self.initialized:
            return []
        try:
            results = self.collection.get(where={"conversation_id": conversation_id, "type": "summary"})
            return results.get('documents', [])
        except Exception as e:
            logger.error(f"Error fetching summaries: {str(e)}")
            return []

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Retrieve all conversations with their chunks and summaries."""
        if not self.initialized:
            logger.warning("VectorStore not initialized, returning empty list")
            return []
        try:
            results = self.collection.get()
            if not results or not results["ids"]:
                logger.debug("No conversations found in vector store")
                return []
            conversations = {}
            for id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                # Skip entries without conversation_id
                if "conversation_id" not in meta:
                    logger.warning(f"Skipping entry {id} without conversation_id")
                    continue
                    
                conv_id = meta["conversation_id"]
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        "id": conv_id,
                        "chunks": [],
                        "summaries": [],
                        "metadata": {
                            "timestamp": meta.get("timestamp"),
                            "message_count": meta.get("message_count", 0),
                            "total_chunks": 0
                        }
                    }
                if meta.get("type") == "summary":
                    conversations[conv_id]["summaries"].append({
                        "content": doc,
                        "chunk_index": meta.get("chunk_index", 0)
                    })
                else:
                    messages = json.loads(meta.get("messages_json", "[]"))
                    conversations[conv_id]["chunks"].append({
                        "content": doc,
                        "messages": messages,
                        "chunk_start": meta.get("chunk_start", 0),
                        "chunk_end": meta.get("chunk_end", 0),
                        "chunk_index": meta.get("chunk_index", 0)
                    })
                    conversations[conv_id]["metadata"]["total_chunks"] = max(
                        conversations[conv_id]["metadata"]["total_chunks"],
                        meta.get("chunk_index", 0) + 1
                    )
            for conv in conversations.values():
                conv["chunks"].sort(key=lambda x: x["chunk_index"])
                conv["summaries"].sort(key=lambda x: x["chunk_index"])
            conversation_list = list(conversations.values())
            conversation_list.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
            return conversation_list
        except Exception as e:
            logger.error(f"Error getting all conversations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def store_message(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a single message in the vector store."""
        if not message or not isinstance(message, dict):
            raise ValueError("Invalid message format")
            
        # Convert numpy types in message
        message = self._convert_to_serializable(message)
        
        # Create unique ID for the message
        message_id = f"msg_{uuid.uuid4()}"
        
        # Prepare metadata
        msg_metadata = {
            "type": "message",
            "role": message.get("role", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_id": message.get("conversation_id", "unknown")
        }
        
        if metadata:
            msg_metadata.update(metadata)
            
        # Store the message
        self.collection.add(
            documents=[message.get("content", "")],
            metadatas=[msg_metadata],
            ids=[message_id]
        )
        
        return message_id

    def store_summary(self, summary: str, metadata: Dict[str, Any]) -> str:
        """Store a conversation summary in the vector store."""
        if not summary:
            raise ValueError("Summary cannot be empty")
            
        # Ensure conversation_id is present
        if "conversation_id" not in metadata:
            raise ValueError("conversation_id is required in metadata")
            
        # Create unique ID for the summary
        summary_id = f"summary_{uuid.uuid4()}"
        
        # Ensure required metadata fields
        metadata.update({
            "type": "summary",
            "timestamp": metadata.get("timestamp", datetime.utcnow().isoformat())
        })
        
        # Store the summary
        self.collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[summary_id]
        )
        
        return summary_id

    def get_relevant_messages(self, query: str, conversation_id: Optional[int] = None, limit: int = 5) -> List[Dict]:
        """Retrieve relevant messages from the vector database."""
        try:
            total_count = self.collection.count()
            if total_count == 0:
                return []
            where = {"conversation_id": conversation_id} if conversation_id else None
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit, total_count),
                where=where
            )
            messages = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    message = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    }
                    messages.append(message)
            return messages
        except Exception as e:
            logger.error(f"Error retrieving messages from vector database: {str(e)}")
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
        """Delete all messages and summaries from a conversation."""
        try:
            self.collection.delete(where={"conversation_id": conversation_id})
            logger.debug(f"Deleted conversation {conversation_id} from vector database")
        except Exception as e:
            logger.error(f"Error deleting conversation from vector database: {str(e)}")
            raise
