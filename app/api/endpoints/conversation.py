from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from app.services.conversation_service import ConversationService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    use_history: bool = True

class ChatResponse(BaseModel):
    conversation_id: str
    response: str

class Message(BaseModel):
    role: str
    content: str
    timestamp: str

class ConversationMetadata(BaseModel):
    timestamp: str
    message_count: int

class Conversation(BaseModel):
    id: str
    messages: List[Message]
    metadata: ConversationMetadata

# Create a single instance of ConversationService
conversation_service = ConversationService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.debug(f"Received chat request: {request.message}")
        response = conversation_service.send_message(
            message=request.message,
            use_history=request.use_history
        )
        logger.debug(f"Generated response: {response[:100]}...")
        return ChatResponse(
            conversation_id=conversation_service.current_conversation_id,
            response=response
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations", response_model=List[Conversation])
async def get_conversations():
    try:
        logger.debug("Getting all conversations")
        conversations = conversation_service.get_all_conversations()
        logger.debug(f"Retrieved {len(conversations)} conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 