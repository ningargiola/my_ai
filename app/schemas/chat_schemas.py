from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class MessageBase(BaseModel):
    role: str
    content: str

class MessageCreate(MessageBase):
    pass

class Message(MessageBase):
    id: int
    conversation_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationBase(BaseModel):
    title: Optional[str] = None

class ConversationCreate(ConversationBase):
    pass

class Conversation(ConversationBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[Message] = []

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    """Request schema for chat messages."""
    message: str
    conversation_id: Optional[int] = None
    use_history: bool = True

class ChatResponse(BaseModel):
    """Response schema for chat messages."""
    conversation_id: int
    response: str
    analysis: Optional[Dict[str, Any]] = None

class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages."""
    type: str  # "message" or "analysis"
    data: Dict[str, Any]

class AnalysisResult(BaseModel):
    """Schema for analysis results."""
    entities: List[Dict[str, Any]]
    intents: List[Dict[str, Any]]
    sentiment: Optional[Dict[str, float]] = None
    summary: Optional[str] = None 