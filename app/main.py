from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import traceback
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
try:
    from app.services.conversation_service import ConversationService
    from app.schemas.chat_schemas import WebSocketMessage, AnalysisResult
    conversation_service = ConversationService()
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {str(e)}")
    logger.error(traceback.format_exc())
    conversation_service = None

def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    return obj

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not conversation_service:
        await websocket.close(code=1008, reason="Service initialization failed")
        return
        
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            # Start a new conversation if needed
            if not conversation_service.current_conversation:
                conversation = conversation_service.start_new_conversation()
            else:
                conversation = conversation_service.current_conversation
            
            # Process message through conversation service
            conversation = conversation_service.add_message(conversation.id, message)
            
            # Convert numpy types to Python native types
            serializable_response = _convert_to_serializable({
                "response": conversation.messages[-1].content,
                "analysis": conversation.messages[-1].analysis,
                "summary": conversation.summary,
                "conversation_id": conversation.id
            })
            
            # Create WebSocket message
            ws_message = WebSocketMessage(
                type="message",
                data=serializable_response
            )
            
            # Send response back to client
            await websocket.send_json(ws_message.dict())
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket: {str(e)}")
        logger.error(traceback.format_exc())
        await websocket.close()

@app.get("/api/v1/conversations")
async def get_conversations() -> List[Dict[str, Any]]:
    """Get all conversations."""
    if not conversation_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return conversation_service.get_all_conversations()
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get a specific conversation by ID."""
    if not conversation_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        conversations = conversation_service.get_all_conversations()
        for conv in conversations:
            if conv["id"] == conversation_id:
                return conv
        raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat")
async def chat(message: str, conversation_id: str = None, use_history: bool = True) -> Dict[str, Any]:
    """Send a message and get a response."""
    if not conversation_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        # Start a new conversation if needed
        if not conversation_service.current_conversation:
            conversation = conversation_service.start_new_conversation()
        else:
            conversation = conversation_service.current_conversation
        
        # Process message through conversation service
        conversation = conversation_service.add_message(conversation.id, message)
        
        return {
            "response": conversation.messages[-1].content,
            "analysis": conversation.messages[-1].analysis,
            "summary": conversation.summary,
            "conversation_id": conversation.id
        }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 