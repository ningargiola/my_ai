# conversation_service.py

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

def aggregate_entities_intents(messages: List[Message]) -> (Dict[str, List[str]], Dict[str, float]):
    """Aggregate entities and intents from a list of messages."""
    entities_by_type = {}
    intents_agg = {}
    for msg in messages:
        if not hasattr(msg, 'analysis') or msg.analysis is None:
            continue
            
        # Entities
        for ent_type, ents in msg.analysis.get('entities', {}).items():
            if ent_type not in entities_by_type:
                entities_by_type[ent_type] = set()
            entities_by_type[ent_type].update(ents)
        # Intents
        for intent, score in msg.analysis.get('intent', {}).items():
            if intent not in intents_agg:
                intents_agg[intent] = 0.0
            intents_agg[intent] += score
    # Convert sets to lists
    entities_by_type = {k: list(v) for k, v in entities_by_type.items()}
    # Average intent scores if desired
    if intents_agg:
        total_msgs = len([msg for msg in messages if hasattr(msg, 'analysis') and msg.analysis is not None])
        if total_msgs > 0:
            intents_agg = {k: v / total_msgs for k, v in intents_agg.items()}
    return entities_by_type, intents_agg

class ConversationService:
    def __init__(self, vector_store: VectorStore = None):
        try:
            logger.debug("Initializing ConversationService")
            self.vector_store = vector_store or VectorStore()
            self.ollama_service = OllamaService()
            self.entity_extractor = EntityExtractor()
            self.summarizer = Summarizer(vector_store=self.vector_store)
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
            if not conversation_id:
                raise ValueError("Conversation ID cannot be empty")

            if not message_content or not message_content.strip():
                raise ValueError("Message content cannot be empty")

            # Start a new conversation if needed
            if not self.current_conversation or self.current_conversation.id != conversation_id:
                # Check if conversation exists in vector store
                existing_conversations = self.vector_store.get_all_conversations()
                conversation_exists = any(conv["id"] == conversation_id for conv in existing_conversations)
                if not conversation_exists:
                    raise ValueError(f"Conversation with ID {conversation_id} does not exist")
                self.current_conversation = Conversation(id=conversation_id)

            # Analyze the message for entities and intent
            logger.debug("Analyzing message for entities and intent")
            message_analysis = self.entity_extractor.analyze_message(message_content) or {
                "entities": {},
                "intent": {},
                "raw_entities": []
            }
            logger.debug(f"Message analysis: {message_analysis}")

            # Create and add user message
            user_message = Message(
                role="user",
                content=message_content,
                analysis=message_analysis
            )
            self.current_conversation.add_message(user_message)
            logger.debug("Added user message to current conversation")

            # Retrieve similar conversation chunks for context
            similar_conversations = self.vector_store.get_similar_conversations(
                message_content, limit=5, min_similarity=0.7
            )
            logger.debug(f"Found {len(similar_conversations)} similar conversations")

            # Build semantic context string from similar chunks
            context_parts = ["Context from previous relevant conversations:"]
            for conv in similar_conversations:
                for chunk in conv['chunks']:
                    for msg in chunk['messages']:
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        content = msg['content']
                        context_parts.append(f"{role}: {content}")
            context_str = "\n".join(context_parts)

            # Fetch conversation summary (use recent or relevant)
            recent_summary = None
            if hasattr(self.current_conversation, 'summary') and self.current_conversation.summary:
                recent_summary = self.current_conversation.summary

            # Aggregate entities and intents from the current conversation
            entities, intents = aggregate_entities_intents(self.current_conversation.messages)
            entities_text = (
                ", ".join([f"{k}: {', '.join(v)}" for k, v in entities.items()])
                if entities else "None"
            )
            intents_text = (
                ", ".join([f"{k} ({v:.2f})" for k, v in intents.items()])
                if intents else "None"
            )

            # Build enhanced prompt/context for the LLM
            messages_for_ollama = [
                {
                    "role": "system",
                    "content": (
                        "You are ALIAS, Nick's personal AI assistant. Use the provided context, conversation summary, and entities/intents "
                        "to give personal, informed answers. Reference relevant past information as if you remember the user's history."
                    )
                }
            ]

            if context_str:
                messages_for_ollama.append({
                    "role": "system",
                    "content": f"Relevant conversation snippets:\n{context_str}"
                })

            if recent_summary:
                messages_for_ollama.append({
                    "role": "system",
                    "content": f"Summary of recent discussion:\n{recent_summary}"
                })

            if entities_text or intents_text:
                messages_for_ollama.append({
                    "role": "system",
                    "content": (
                        f"Recently mentioned entities: {entities_text}.\n"
                        f"Recent user intent: {intents_text}."
                    )
                })

            # Add conversation history (last few messages for continuity)
            messages_for_ollama.extend([
                {"role": msg.role, "content": msg.content}
                for msg in self.current_conversation.messages[-5:]  # Only last 5 for brevity
            ])

            # Call the LLM with the full super-context
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
                    "analysis": msg.analysis if hasattr(msg, 'analysis') and msg.analysis else {}
                }
                for msg in self.current_conversation.messages[-5:]  # Only summarize last 5 messages for brevity
            ])

            # Format conversation for vector store
            conversation_dict = {
                "id": self.current_conversation.id,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "analysis": msg.analysis if hasattr(msg, 'analysis') and msg.analysis else {}
                    }
                    for msg in self.current_conversation.messages
                ],
                "summary": self.current_conversation.summary,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Store the conversation in vector store
            self.vector_store.add_conversation(conversation_dict)
            logger.debug("Stored conversation in vector store")
            logger.info(f"Total records in vector DB: {self.vector_store.collection.count()}")

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
