import requests
import logging
from typing import List, Dict, Any
import json
import traceback
import numpy as np

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self, model_name: str = "llama3"):
        try:
            logger.debug(f"Initializing OllamaService with model: {model_name}")
            self.model_name = model_name
            self.base_url = "http://localhost:11434"
            self.system_prompt = """You are a helpful assistant that can answer questions and help with tasks. Act as though you are a personal assistant with a human tone"""
            logger.debug("OllamaService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OllamaService: {str(e)}")
            logger.error(traceback.format_exc())
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

    def generate_response(self, message: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate a response using Ollama with context from vector store"""
        try:
            # Prepare the messages array
            messages = []
            logger.debug("Preparing messages for Ollama")
            
            # Add conversation history
            if conversation_history:
                # Convert numpy types to Python native types
                serializable_history = self._convert_to_serializable(conversation_history)
                messages.extend(serializable_history)
                logger.debug(f"Added {len(conversation_history)} messages from conversation history")
                
                # Log context messages
                for msg in conversation_history:
                    if msg['role'] == 'system' and 'Context from previous' in msg['content']:
                        logger.info("Including context in response generation:")
                        logger.info(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            logger.debug("Added current message to messages")
            
            # Prepare the request payload with instructions to use context
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Slightly lower temperature for more focused responses
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "repeat_penalty": 1.1,
                    "top_k": 40,
                    "num_predict": 1000,
                    "stop": ["</s>", "User:", "Assistant:"]
                }
            }
            
            logger.debug(f"Sending request to Ollama with {len(messages)} messages")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            if "message" in result and "content" in result["message"]:
                logger.debug("Successfully extracted response content")
                response_content = result["message"]["content"]
                
                # Log that we're using context in the response
                if any(msg['role'] == 'system' and 'Context from previous' in msg['content'] for msg in messages):
                    logger.info("Response generated using context from previous conversations")
                
                return response_content
            else:
                logger.error(f"Unexpected response format: {json.dumps(result, indent=2)}")
                raise ValueError("Unexpected response format from Ollama")
                
        except requests.exceptions.ConnectionError as e:
            logger.error("Failed to connect to Ollama service. Is Ollama running?")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def list_available_models(self) -> List[str]:
        """List all available Ollama models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
            
    def set_model(self, model: str):
        """Set the model to use for generation."""
        self.model_name = model 