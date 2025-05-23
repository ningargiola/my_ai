from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class Message(BaseModel):
    """Represents a single message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    analysis: Optional[Dict[str, Any]] = None
    similar_conversations: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    """Represents a conversation with multiple messages"""
    id: str
    messages: List[Message] = []
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation"""
        return len(self.messages)

    def get_user_messages(self) -> List[Message]:
        """Get all user messages in the conversation"""
        return [msg for msg in self.messages if msg.role == 'user']

    def get_assistant_messages(self) -> List[Message]:
        """Get all assistant messages in the conversation"""
        return [msg for msg in self.messages if msg.role == 'assistant'] 