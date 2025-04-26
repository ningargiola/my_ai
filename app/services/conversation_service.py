from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from app.services.vector_store import VectorStore
from app.services.ollama_service import OllamaService
from app.services.entity_extractor import EntityExtractor
from app.services.summarizer import Summarizer
from app.models.conversation import Conversation, Message
import logging
import traceback

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, vector_store: VectorStore = None):
        try:
            logger.debug("Initializing ConversationService")
            # Use in-memory vector store for testing
            self.vector_store = vector_store or VectorStore(persist_directory=None)
            self.ollama_service = OllamaService()
            self.entity_extractor = EntityExtractor()
            self.summarizer = Summarizer()
            self.current_conversation = None
            logger.debug("ConversationService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ConversationService: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def start_new_conversation(self) -> Conversation:
        """Start a new conversation and return it"""
        try:
            conversation_id = str(uuid.uuid4())
            self.current_conversation = Conversation(id=conversation_id)
            logger.debug(f"Started new conversation with ID: {conversation_id}")
            return self.current_conversation
        except Exception as e:
            logger.error(f"Error starting new conversation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def add_message(self, conversation_id: str, message_content: str) -> Conversation:
        """Add a message to a conversation and get a response"""
        try:
            # Start a new conversation if needed
            if not self.current_conversation or self.current_conversation.id != conversation_id:
                self.current_conversation = Conversation(id=conversation_id)
            
            # Analyze the message for entities and intent
            logger.debug("Analyzing message for entities and intent")
            message_analysis = self.entity_extractor.analyze_message(message_content)
            logger.debug(f"Message analysis: {message_analysis}")
            
            # Create and add user message
            user_message = Message(
                role="user",
                content=message_content,
                analysis=message_analysis
            )
            self.current_conversation.add_message(user_message)
            logger.debug("Added user message to current conversation")

            # Get relevant context from vector store
            similar_conversations = self.vector_store.get_similar_conversations(
                message_content,
                limit=5,  # Get more conversations for better context
                min_similarity=0.7  # Only include reasonably similar conversations
            )
            user_message.similar_conversations = similar_conversations
            
            # Format similar conversations for context, tracking seen content
            seen_content = set()
            context_parts = []
            
            for conv in similar_conversations:
                # Log similarity score
                logger.info(f"Found similar conversation with similarity: {conv['similarity']:.2f}")
                
                for chunk in conv['chunks']:
                    messages = chunk['messages']
                    # Create a content key for deduplication
                    content_key = "|".join(msg['content'] for msg in messages)
                    
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        context_parts.append("Previous relevant conversation:")
                        for msg in messages:
                            role = "User" if msg['role'] == 'user' else "Assistant"
                            content = msg['content']
                            context_parts.append(f"{role}: {content}")
                        logger.info(f"Using context: {messages[-1]['content'][:100]}...")
                        context_parts.append("---")
            
            context = "\n".join(context_parts) if context_parts else ""
            logger.debug(f"Found {len(similar_conversations)} similar conversations")
            if context:
                logger.info("Using the following context for response generation:")
                logger.info(context[:500] + "..." if len(context) > 500 else context)

            # Generate response using Ollama with explicit context
            logger.debug("Generating response from Ollama")
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in self.current_conversation.messages
            ]
            
            # Add context as a system message if available
            messages_for_ollama = []
            messages_for_ollama.append({
                "role": "system",
                "content": "You are ALIAS (Artificial Learning and Interactive Assistive System), a helpful AI assistant. Use the context from previous conversations to inform your responses."
            })
            
            if context:
                messages_for_ollama.append({
                    "role": "system",
                    "content": f"Context from previous relevant conversations:\n{context}"
                })
            
            messages_for_ollama.extend(conversation_history)
            
            response = self.ollama_service.generate_response(
                message=message_content,
                conversation_history=messages_for_ollama
            )
            logger.debug("Received response from Ollama")

            # Create and add assistant message
            assistant_message = Message(
                role="assistant",
                content=response
            )
            self.current_conversation.add_message(assistant_message)
            logger.debug("Added AI response to current conversation")

            # Generate and update conversation summary
            logger.debug("Generating conversation summary")
            self.current_conversation.summary = self.summarizer.summarize_conversation([
                {
                    "role": msg.role,
                    "content": msg.content,
                    "analysis": msg.analysis if msg.analysis else {}
                }
                for msg in self.current_conversation.messages
            ])
            logger.debug("Generated conversation summary")

            # Store the conversation in vector store
            self.vector_store.add_conversation(self.current_conversation)

            return self.current_conversation

        except Exception as e:
            logger.error(f"Error in add_message: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations from the vector store"""
        try:
            logger.debug("Getting all conversations from vector store")
            conversations_data = self.vector_store.get_all_conversations()
            
            # Convert to Conversation objects
            conversations = []
            for data in conversations_data:
                conversation = Conversation(id=data["id"])
                
                # Add messages from chunks
                for chunk in sorted(data["chunks"], key=lambda x: x["chunk_start"]):
                    for msg_data in chunk["messages"]:
                        message = Message(
                            role=msg_data["role"],
                            content=msg_data["content"],
                            analysis=msg_data.get("analysis", {})
                        )
                        conversation.add_message(message)
                
                conversations.append(conversation)
            
            logger.debug(f"Retrieved {len(conversations)} conversations")
            return conversations
        except Exception as e:
            logger.error(f"Error getting all conversations: {str(e)}")
            logger.error(traceback.format_exc())
            raise 