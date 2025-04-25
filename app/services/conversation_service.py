from typing import List, Dict, Any
import uuid
from datetime import datetime
from app.services.vector_store import VectorStore
from app.services.ollama_service import OllamaService
import logging
import traceback

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self):
        try:
            logger.debug("Initializing ConversationService")
            self.vector_store = VectorStore()
            self.ollama_service = OllamaService()
            self.current_conversation_id = None
            self.current_messages = []
            logger.debug("ConversationService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ConversationService: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def start_new_conversation(self) -> str:
        """Start a new conversation and return its ID"""
        try:
            conversation_id = str(uuid.uuid4())
            self.current_conversation_id = conversation_id
            self.current_messages = []
            logger.debug(f"Started new conversation with ID: {conversation_id}")
            return conversation_id
        except Exception as e:
            logger.error(f"Error starting new conversation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def send_message(self, message: str, use_history: bool = True) -> str:
        """Send a message and get a response, using conversation history as context"""
        try:
            logger.debug(f"Sending message: {message}")
            
            # Start a new conversation if none exists
            if not self.current_conversation_id:
                self.start_new_conversation()
            
            # Add user message to current conversation
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.current_messages.append(user_message)
            logger.debug("Added user message to current conversation")

            # Get relevant context from vector store
            context = ""
            if use_history:
                logger.debug("Retrieving similar conversations for context")
                similar_conversations = self.vector_store.get_similar_conversations(message)
                if similar_conversations:
                    context = "\n\nRelevant previous conversations:\n" + "\n---\n".join(
                        [conv["content"] for conv in similar_conversations]
                    )
                    logger.debug(f"Found {len(similar_conversations)} similar conversations")

            # Generate response using Ollama with context
            logger.debug("Generating response from Ollama")
            response = self.ollama_service.generate_response(
                message=message,
                context=context,
                conversation_history=self.current_messages
            )
            logger.debug("Received response from Ollama")

            # Add AI response to current conversation
            ai_message = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.current_messages.append(ai_message)
            logger.debug("Added AI response to current conversation")

            # Store the updated conversation in vector store
            logger.debug("Storing conversation in vector store")
            self.vector_store.add_conversation(
                conversation_id=self.current_conversation_id,
                messages=self.current_messages
            )
            logger.debug("Successfully stored conversation in vector store")

            return response
        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations from the vector store"""
        try:
            logger.debug("Getting all conversations from vector store")
            conversations = self.vector_store.get_all_conversations()
            logger.debug(f"Retrieved {len(conversations)} conversations")
            return conversations
        except Exception as e:
            logger.error(f"Error getting all conversations: {str(e)}")
            logger.error(traceback.format_exc())
            raise 